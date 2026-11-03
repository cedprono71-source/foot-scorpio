import os
import re
import math
import time
import sqlite3
import logging
import threading
import unicodedata
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Tuple

import requests
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Modèle ML optionnel : l'app reste utilisable sans scikit-learn.
try:
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

APP_VERSION = "33.0"
DB_PATH = os.getenv("FOOTBALL_DB_PATH", "football_history.db")
MODEL_PATH = os.getenv("MODEL_PATH", "ml_model.joblib")
PARIS_TZ = ZoneInfo("Europe/Paris")
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
REQUEST_TIMEOUT = 8
BACKFILL_DAYS = int(os.getenv("BACKFILL_DAYS", "90"))
BACKFILL_MAX_DAYS = 1095          # ~3 saisons
BACKFILL_WORKERS = int(os.getenv("BACKFILL_WORKERS", "6"))
MIN_ML_ROWS = 80
MODEL_TTL = 3600                  # durée de vie du modèle ML (secondes)
INSUFFICIENT_RETRY = 300          # délai avant de retenter un entraînement impossible
ML_FEATURE_VERSION = 2            # à incrémenter si les features changent
HOME_ELO_ADV = 55.0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("signal-fort")

LEAGUES = {
    "eng.1": "Premier League",
    "eng.2": "EFL Championship",
    "esp.1": "La Liga",
    "esp.2": "Segunda División",
    "ita.1": "Serie A",
    "ita.2": "Serie B",
    "ger.1": "Bundesliga",
    "ger.2": "2. Bundesliga",
    "fra.1": "Ligue 1",
    "fra.2": "Ligue 2",
    "ned.1": "Eredivisie",
    "por.1": "Liga Portugal",
    "bel.1": "Jupiler Pro League",
    "tur.1": "Süper Lig",
    "sco.1": "Scottish Premiership",
    "uefa.champions": "UEFA Champions League",
    "uefa.europa": "UEFA Europa League",
    "uefa.conf": "UEFA Conference League",
}

# Caches et verrous (TTLCache n'est pas thread-safe : toujours passer par cache_get / cache_set)
matches_cache = TTLCache(maxsize=200, ttl=300)
prediction_cache = TTLCache(maxsize=500, ttl=900)
ml_model_cache = {"model": None, "trained_at": 0.0, "rows": 0, "retry_after": 0.0}
elo_cache = {"version": -1, "ratings": {}}
data_version = {"v": 0}  # incrémenté dès que l'historique change réellement
backfill_status = {"running": False, "days": 0, "total": 0, "completed": 0,
                   "errors": 0, "started_at": None, "finished_at": None}
cache_lock = threading.Lock()
train_lock = threading.Lock()
backfill_lock = threading.Lock()


def cache_get(cache, key):
    with cache_lock:
        return cache.get(key)


def cache_set(cache, key, value):
    with cache_lock:
        cache[key] = value


def invalidate_data():
    """Appelé quand l'historique change : invalide les prédictions et force le recalcul des Elo."""
    with cache_lock:
        data_version["v"] += 1
        prediction_cache.clear()


session = requests.Session()
session.headers.update({"User-Agent": f"SignalFort/{APP_VERSION} (football statistics app)"})


# --------------------------------------------------------------------------- #
# Noms d'équipes et identifiants
# --------------------------------------------------------------------------- #
_STOP_TOKENS = {"fc", "cf", "afc", "sc", "ac"}


def normalize_name(name: str) -> str:
    """'Atlético de Madrid' -> 'atletico de madrid' ; 'AC Milan' -> 'milan'."""
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    tokens = re.findall(r"[a-z0-9]+", s)
    kept = [t for t in tokens if t not in _STOP_TOKENS]
    return " ".join(kept or tokens)


# --------------------------------------------------------------------------- #
# Base de données
# --------------------------------------------------------------------------- #
def db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=20, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=20000")
    return conn


@contextmanager
def db():
    """Connexion avec commit/rollback ET fermeture (le 'with' de sqlite3 ne ferme pas)."""
    conn = db_connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS matches_history (
                event_id TEXT PRIMARY KEY,
                league TEXT,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_score INTEGER,
                away_score INTEGER,
                date TEXT NOT NULL,
                home_id TEXT,
                away_id TEXT
            )
        """)
        # Migration des anciennes bases (sans identifiants d'équipes)
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(matches_history)")}
        if "home_id" not in columns:
            conn.execute("ALTER TABLE matches_history ADD COLUMN home_id TEXT")
        if "away_id" not in columns:
            conn.execute("ALTER TABLE matches_history ADD COLUMN away_id TEXT")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                team_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                norm_name TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backfill_log (
                league_code TEXT NOT NULL,
                day TEXT NOT NULL,
                PRIMARY KEY (league_code, day)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_matches_date ON matches_history(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_matches_home ON matches_history(home_team, date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_matches_away ON matches_history(away_team, date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_matches_home_id ON matches_history(home_id, date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_matches_away_id ON matches_history(away_id, date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_teams_norm ON teams(norm_name)")
    link_legacy_team_ids()


def link_legacy_team_ids():
    """Rattache les anciens matchs (sans id) à l'id de l'équipe quand le nom est connu."""
    try:
        with db() as conn:
            changed = 0
            for side in ("home", "away"):
                cur = conn.execute(f"""
                    UPDATE matches_history
                    SET {side}_id = (SELECT t.team_id FROM teams t
                                     WHERE t.name = matches_history.{side}_team LIMIT 1)
                    WHERE {side}_id IS NULL
                      AND EXISTS (SELECT 1 FROM teams t WHERE t.name = matches_history.{side}_team)
                """)
                changed += max(cur.rowcount, 0)
        if changed:
            logger.info("Matchs rattachés à un id d'équipe : %s", changed)
            invalidate_data()
    except Exception:
        logger.exception("link_legacy_team_ids failed")


init_db()


def save_match_to_db(event_id, league, home_team, away_team, home_score, away_score, date,
                     home_id=None, away_id=None):
    if not event_id or not home_team or not away_team:
        return
    if home_score is None or away_score is None:
        return
    home_id = str(home_id).strip() if home_id not in (None, "") else None
    away_id = str(away_id).strip() if away_id not in (None, "") else None
    try:
        with db() as conn:
            cur = conn.execute("""
                INSERT INTO matches_history
                (event_id, league, home_team, away_team, home_score, away_score, date, home_id, away_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    league = excluded.league,
                    home_team = excluded.home_team,
                    away_team = excluded.away_team,
                    home_score = excluded.home_score,
                    away_score = excluded.away_score,
                    date = excluded.date,
                    home_id = excluded.home_id,
                    away_id = excluded.away_id
                WHERE league IS NOT excluded.league
                   OR home_team IS NOT excluded.home_team
                   OR away_team IS NOT excluded.away_team
                   OR home_score IS NOT excluded.home_score
                   OR away_score IS NOT excluded.away_score
                   OR date IS NOT excluded.date
                   OR home_id IS NOT excluded.home_id
                   OR away_id IS NOT excluded.away_id
            """, (str(event_id), league, home_team, away_team,
                  int(home_score), int(away_score), str(date)[:10], home_id, away_id))
            changed = cur.rowcount > 0
            for team_id, name in ((home_id, home_team), (away_id, away_team)):
                if team_id:
                    conn.execute("""
                        INSERT INTO teams (team_id, name, norm_name) VALUES (?, ?, ?)
                        ON CONFLICT(team_id) DO UPDATE SET
                            name = excluded.name, norm_name = excluded.norm_name
                        WHERE name IS NOT excluded.name
                    """, (team_id, name, normalize_name(name)))
        if changed:
            # Le modèle ML n'est pas invalidé ici : il est ré-entraîné toutes les MODEL_TTL secondes.
            invalidate_data()
    except Exception:
        logger.exception("Unable to save match %s", event_id)


def resolve_team_key(name: str, team_id: Optional[str] = None) -> str:
    """Clé interne d'une équipe : son id ESPN si connu, sinon 'n:<nom>' (historique non rattaché)."""
    with db() as conn:
        if team_id:
            tid = str(team_id).strip()
            if conn.execute("SELECT 1 FROM teams WHERE team_id = ?", (tid,)).fetchone():
                return tid
        norm = normalize_name(name)
        row = conn.execute("SELECT team_id FROM teams WHERE norm_name = ? LIMIT 1", (norm,)).fetchone()
        if row:
            return row["team_id"]
    return f"n:{name}"


# --------------------------------------------------------------------------- #
# ESPN
# --------------------------------------------------------------------------- #
def fetch_scoreboard(league_code: str, date_yyyymmdd: str, retries: int = 1) -> List[dict]:
    """Appel ESPN avec nouvelles tentatives (429 et erreurs 5xx uniquement)."""
    url = f"{ESPN_BASE}/{league_code}/scoreboard"
    last_exc: Optional[Exception] = None
    for attempt in range(max(1, retries)):
        try:
            response = session.get(url, params={"dates": date_yyyymmdd}, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json().get("events", [])
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and status < 500 and status != 429:
                raise
            last_exc = exc
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
        if attempt < retries - 1:
            time.sleep(0.6 * (2 ** attempt))
    raise last_exc  # type: ignore[misc]


def parse_event(event: dict, league_code: str) -> Optional[dict]:
    competitions = event.get("competitions") or []
    if not competitions:
        return None
    comp = competitions[0]
    competitors = comp.get("competitors") or []
    if len(competitors) < 2:
        return None

    home = away = None
    for index, item in enumerate(competitors):
        side = item.get("homeAway")
        if side == "home" or (not side and index == 0):
            home = item
        elif side == "away" or (not side and index == 1):
            away = item
    if not home or not away:
        return None

    event_date = event.get("date")
    if not event_date:
        return None
    try:
        dt = datetime.fromisoformat(event_date.replace("Z", "+00:00"))
        local_dt = dt.astimezone(PARIS_TZ)
    except Exception:
        return None

    status = (comp.get("status") or {}).get("type") or {}
    state = str(status.get("state", "")).lower()
    status_name = str(status.get("name", "")).upper()
    if state == "post" or status_name == "STATUS_FINAL":
        game_state = "OFF"
    elif state in ("in", "live") or "IN_PROGRESS" in status_name or "LIVE" in status_name:
        game_state = "En cours"
    else:
        game_state = local_dt.strftime("%H:%M")

    def team_data(item):
        team = item.get("team") or {}
        score = item.get("score")
        try:
            score = int(score) if score is not None and str(score).strip() != "" else None
        except (ValueError, TypeError):
            score = None
        logo = team.get("logo")
        if not logo and team.get("logos"):
            logo = (team["logos"][0] or {}).get("href", "")
        team_id = team.get("id") or item.get("id")
        return {
            "id": str(team_id) if team_id not in (None, "") else None,
            "name": team.get("displayName") or team.get("name") or "Équipe inconnue",
            "logo": logo or "",
            "score": score,
        }

    h, a = team_data(home), team_data(away)
    parsed = {
        "event_id": str(event.get("id", "")),
        "date": local_dt.strftime("%Y-%m-%d"),
        "_utc_date": event_date,
        "league": LEAGUES.get(league_code, league_code),
        "league_code": league_code,
        "home_team": h["name"],
        "away_team": a["name"],
        "home_id": h["id"],
        "away_id": a["id"],
        "home_logo": h["logo"],
        "away_logo": a["logo"],
        "home_score": h["score"],
        "away_score": a["score"],
        "game_state": game_state,
    }
    if game_state == "OFF":
        save_match_to_db(parsed["event_id"], parsed["league"], h["name"], a["name"],
                         h["score"], a["score"], parsed["date"], h["id"], a["id"])
    return parsed


def backfill_history(days_back: int = BACKFILL_DAYS, leagues: Optional[List[str]] = None):
    """Charge l'historique ESPN, jour par jour et ligue par ligue.
    Reprenable : les couples (ligue, jour) déjà terminés sont mémorisés dans backfill_log
    et ignorés aux lancements suivants. Les 2 derniers jours ne sont jamais marqués comme terminés."""
    league_codes = [c for c in (leagues or list(LEAGUES)) if c in LEAGUES]
    today = datetime.now(PARIS_TZ).date()
    logger.info("Starting ESPN historical backfill (%s days, %s leagues)", days_back, len(league_codes))

    with db() as conn:
        done = {(r["league_code"], r["day"]) for r in conn.execute("SELECT league_code, day FROM backfill_log")}

    tasks = []
    for day_offset in range(days_back):
        day = today - timedelta(days=day_offset)
        key = day.strftime("%Y%m%d")
        for code in league_codes:
            if (code, key) not in done:
                tasks.append((code, key, day))

    with cache_lock:
        backfill_status.update(running=True, days=days_back, total=len(tasks), completed=0,
                               errors=0, started_at=datetime.now(PARIS_TZ).isoformat(),
                               finished_at=None)

    pending_done: List[Tuple[str, str]] = []

    def flush():
        if not pending_done:
            return
        with db() as conn:
            conn.executemany("INSERT OR IGNORE INTO backfill_log (league_code, day) VALUES (?, ?)",
                             pending_done)
        pending_done.clear()

    try:
        with ThreadPoolExecutor(max_workers=BACKFILL_WORKERS) as pool:
            futures = {pool.submit(fetch_scoreboard, code, key, 3): (code, key, day)
                       for code, key, day in tasks}
            # Cette boucle tourne dans le thread appelant : une seule écriture SQLite à la fois.
            for future in as_completed(futures):
                code, key, day = futures[future]
                ok = False
                try:
                    for event in future.result():
                        parse_event(event, code)
                    ok = True
                except Exception as exc:
                    logger.debug("Backfill failed for %s %s: %s", code, key, exc)
                with cache_lock:
                    backfill_status["completed"] += 1
                    if not ok:
                        backfill_status["errors"] += 1
                if ok and (today - day).days >= 2:
                    pending_done.append((code, key))
                    if len(pending_done) >= 200:
                        flush()
        flush()
    finally:
        link_legacy_team_ids()
        with cache_lock:
            backfill_status.update(running=False, finished_at=datetime.now(PARIS_TZ).isoformat())
    logger.info("Historical backfill finished")


def run_backfill(days_back: int, leagues: Optional[List[str]] = None):
    if not backfill_lock.acquire(blocking=False):
        logger.info("Backfill already running, skipped")
        return
    try:
        backfill_history(days_back, leagues)
    except Exception:
        logger.exception("Backfill crashed")
    finally:
        backfill_lock.release()


# --------------------------------------------------------------------------- #
# Statistiques (toutes les équipes sont identifiées par une "clé" : id ESPN ou 'n:<nom>')
# --------------------------------------------------------------------------- #
KEY_COLUMNS = """
    COALESCE(home_id, 'n:' || home_team) AS home_key,
    COALESCE(away_id, 'n:' || away_team) AS away_key
"""


def load_finished_matches():
    with db() as conn:
        return conn.execute(f"""
            SELECT home_team, away_team, home_score, away_score, date, {KEY_COLUMNS}
            FROM matches_history
            WHERE home_score IS NOT NULL AND away_score IS NOT NULL
            ORDER BY date ASC, event_id ASC
        """).fetchall()


def get_history(team_key: str, limit=10) -> List[dict]:
    if team_key.startswith("n:"):
        name = team_key[2:]
        where = "(home_id IS NULL AND home_team = ?) OR (away_id IS NULL AND away_team = ?)"
    else:
        name = team_key
        where = "home_id = ? OR away_id = ?"
    with db() as conn:
        rows = conn.execute(f"""
            SELECT league, home_team, away_team, home_score, away_score, date, {KEY_COLUMNS}
            FROM matches_history
            WHERE home_score IS NOT NULL AND away_score IS NOT NULL AND ({where})
            ORDER BY date DESC, event_id DESC LIMIT ?
        """, (name, name, limit)).fetchall()
    return [dict(r) for r in rows]


def team_profile(team_key: str, limit=10) -> dict:
    games = get_history(team_key, limit)
    home_gf, home_ga, away_gf, away_ga = [], [], [], []
    points, goals_for, goals_against, recent_form = 0, 0, 0, []
    for game in games:
        is_home = game["home_key"] == team_key
        gf = game["home_score"] if is_home else game["away_score"]
        ga = game["away_score"] if is_home else game["home_score"]
        if gf is None or ga is None:
            continue
        goals_for += gf
        goals_against += ga
        if is_home:
            home_gf.append(gf)
            home_ga.append(ga)
        else:
            away_gf.append(gf)
            away_ga.append(ga)
        result = "W" if gf > ga else "D" if gf == ga else "L"
        recent_form.append(result)
        points += 3 if result == "W" else 1 if result == "D" else 0

    def avg(values, default):
        return sum(values) / len(values) if values else default

    n = len(recent_form)
    return {
        "matches": n,
        "home_gf": avg(home_gf, avg([goals_for / n] if n else [], 1.35)),
        "home_ga": avg(home_ga, avg([goals_against / n] if n else [], 1.15)),
        "away_gf": avg(away_gf, avg([goals_for / n] if n else [], 1.15)),
        "away_ga": avg(away_ga, avg([goals_against / n] if n else [], 1.35)),
        "goals_for_avg": goals_for / n if n else 1.35,
        "goals_against_avg": goals_against / n if n else 1.35,
        "points_per_game": points / n if n else 1.0,
        "form": recent_form,
    }


def league_goal_averages(league_code: str) -> Tuple[float, float]:
    league_name = LEAGUES.get(league_code, league_code)
    with db() as conn:
        rows = conn.execute("""
            SELECT home_score, away_score FROM matches_history
            WHERE home_score IS NOT NULL AND away_score IS NOT NULL
            AND league = ?
            ORDER BY date DESC LIMIT 1000
        """, (league_name,)).fetchall()
        if len(rows) < 20:
            rows = conn.execute("""
                SELECT home_score, away_score FROM matches_history
                WHERE home_score IS NOT NULL AND away_score IS NOT NULL
                ORDER BY date DESC LIMIT 1000
            """).fetchall()
    if not rows:
        return 1.45, 1.20
    return (
        max(0.65, sum(r["home_score"] for r in rows) / len(rows)),
        max(0.55, sum(r["away_score"] for r in rows) / len(rows)),
    )


def elo_update(ratings: Dict[str, float], h: str, a: str, hs: int, as_: int) -> None:
    rh, ra = ratings.get(h, 1500.0), ratings.get(a, 1500.0)
    expected_h = 1.0 / (1.0 + 10 ** ((ra - (rh + HOME_ELO_ADV)) / 400.0))
    expected_a = 1.0 - expected_h
    actual_h = 1.0 if hs > as_ else 0.5 if hs == as_ else 0.0
    margin = abs(hs - as_)
    k = 20.0 * min(1.5, 1.0 + 0.15 * margin)
    ratings[h] = rh + k * (actual_h - expected_h)
    ratings[a] = ra + k * ((1.0 - actual_h) - expected_a)


def elo_ratings() -> Dict[str, float]:
    """Elo par clé d'équipe, mis en cache et recalculé seulement si l'historique a changé.
    Ne pas modifier le dict retourné."""
    with cache_lock:
        if elo_cache["version"] == data_version["v"]:
            return elo_cache["ratings"]
        version = data_version["v"]
    ratings: Dict[str, float] = {}
    for row in load_finished_matches():
        elo_update(ratings, row["home_key"], row["away_key"],
                   row["home_score"], row["away_score"])
    with cache_lock:
        elo_cache["version"] = version
        elo_cache["ratings"] = ratings
    return ratings


def poisson_p(lam: float, goals: int) -> float:
    return math.exp(-lam) * (lam ** goals) / math.factorial(goals)


def poisson_model(home_xg: float, away_xg: float) -> dict:
    matrix = {}
    total = home_win = draw = away_win = btts = over25 = 0.0
    for hg in range(8):
        for ag in range(8):
            p = poisson_p(home_xg, hg) * poisson_p(away_xg, ag)
            matrix[(hg, ag)] = p
            total += p
            if hg > ag:
                home_win += p
            elif hg == ag:
                draw += p
            else:
                away_win += p
            if hg > 0 and ag > 0:
                btts += p
            if hg + ag >= 3:
                over25 += p
    if total <= 0:
        total = 1.0
    ranked = sorted(matrix.items(), key=lambda x: x[1], reverse=True)
    return {
        "home_win": home_win / total * 100,
        "draw": draw / total * 100,
        "away_win": away_win / total * 100,
        "btts": btts / total * 100,
        "over25": over25 / total * 100,
        "scores": [{"score": f"{s[0]} - {s[1]}", "probability": round(p / total * 100, 2)}
                   for s, p in ranked[:5]],
    }


def build_features(home_key: str, away_key: str, league_code: str) -> dict:
    hp, ap = team_profile(home_key), team_profile(away_key)
    ratings = elo_ratings()
    home_elo = ratings.get(home_key, 1500.0)
    away_elo = ratings.get(away_key, 1500.0)
    home_avg, away_avg = league_goal_averages(league_code)

    def shrink(value, n, prior, weight=5.0):
        return (value * n + prior * weight) / (n + weight)

    h_attack = shrink(hp["goals_for_avg"], hp["matches"], home_avg)
    h_defence = shrink(hp["goals_against_avg"], hp["matches"], away_avg)
    a_attack = shrink(ap["goals_for_avg"], ap["matches"], away_avg)
    a_defence = shrink(ap["goals_against_avg"], ap["matches"], home_avg)

    home_xg = max(0.20, min(4.0, (h_attack * a_defence / max(0.5, away_avg)) * 1.08))
    away_xg = max(0.20, min(4.0, (a_attack * h_defence / max(0.5, home_avg)) * 0.94))
    return {
        "home_profile": hp, "away_profile": ap,
        "home_elo": round(home_elo, 1), "away_elo": round(away_elo, 1),
        "home_xg": round(home_xg, 2), "away_xg": round(away_xg, 2),
        "league_home_goals_avg": round(home_avg, 2),
        "league_away_goals_avg": round(away_avg, 2),
    }


# --------------------------------------------------------------------------- #
# Machine learning
# --------------------------------------------------------------------------- #
def rolling_profile(games: List[Tuple[int, int]]) -> dict:
    """Profil glissant (buts pour, buts contre) -> mêmes clés que team_profile."""
    if not games:
        return {"goals_for_avg": 1.35, "goals_against_avg": 1.35, "points_per_game": 1.0}
    n = len(games)
    return {
        "goals_for_avg": sum(g[0] for g in games) / n,
        "goals_against_avg": sum(g[1] for g in games) / n,
        "points_per_game": sum(3 if g[0] > g[1] else 1 if g[0] == g[1] else 0 for g in games) / n,
    }


def ml_row(hp: dict, ap: dict, elo_diff: float) -> List[float]:
    """Unique définition des features : utilisée à l'entraînement ET à la prédiction."""
    return [
        hp["goals_for_avg"], hp["goals_against_avg"],
        ap["goals_for_avg"], ap["goals_against_avg"],
        hp["points_per_game"], ap["points_per_game"],
        hp["goals_for_avg"] - ap["goals_against_avg"],
        ap["goals_for_avg"] - hp["goals_against_avg"],
        elo_diff,
    ]


def build_ml_dataset() -> Tuple[List[List[float]], List[int]]:
    """Dataset chronologique : features calculées uniquement avec le passé du match (pas de fuite)."""
    rolling: Dict[str, List[Tuple[int, int]]] = {}
    ratings: Dict[str, float] = {}
    X, y = [], []
    for row in load_finished_matches():
        h, a = row["home_key"], row["away_key"]
        hg, ag = int(row["home_score"]), int(row["away_score"])
        hp = rolling_profile(rolling.get(h, [])[-10:])
        ap = rolling_profile(rolling.get(a, [])[-10:])
        elo_diff = ratings.get(h, 1500.0) - ratings.get(a, 1500.0)
        X.append(ml_row(hp, ap, elo_diff))
        y.append(0 if hg > ag else 1 if hg == ag else 2)
        rolling[h] = (rolling.get(h, []) + [(hg, ag)])[-10:]
        rolling[a] = (rolling.get(a, []) + [(ag, hg)])[-10:]
        elo_update(ratings, h, a, hg, ag)
    return X, y


def make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


def train_ml_model():
    if not SKLEARN_AVAILABLE:
        return None, 0

    # Un seul entraînement à la fois ; les autres requêtes attendent puis réutilisent le résultat.
    with train_lock:
        now = time.time()
        with cache_lock:
            cached = dict(ml_model_cache)
        if cached["model"] is not None and now - cached["trained_at"] < MODEL_TTL:
            return cached["model"], cached["rows"]
        if now < cached["retry_after"]:
            return None, cached["rows"]

        if os.path.exists(MODEL_PATH):
            try:
                disk = joblib.load(MODEL_PATH)  # fichier de confiance uniquement (joblib = pickle)
                if (disk.get("feature_version") == ML_FEATURE_VERSION
                        and now - disk.get("trained_at", 0) < MODEL_TTL):
                    with cache_lock:
                        ml_model_cache.update(model=disk["model"],
                                              trained_at=disk["trained_at"],
                                              rows=disk["rows"])
                    return disk["model"], disk["rows"]
            except Exception:
                logger.exception("Impossible de charger le modèle ML depuis le disque")

        X, y = build_ml_dataset()
        if len(X) < MIN_ML_ROWS or len(set(y)) < 3:
            with cache_lock:
                ml_model_cache.update(model=None, rows=len(X), retry_after=now + INSUFFICIENT_RETRY)
            return None, len(X)
        try:
            model = make_model()
            model.fit(X, y)
        except Exception:
            logger.exception("ML training failed")
            with cache_lock:
                ml_model_cache.update(retry_after=now + INSUFFICIENT_RETRY)
            return None, len(X)

        with cache_lock:
            ml_model_cache.update(model=model, trained_at=now, rows=len(X), retry_after=0.0)
        try:
            tmp_path = MODEL_PATH + ".tmp"
            joblib.dump({"model": model, "trained_at": now, "rows": len(X),
                         "feature_version": ML_FEATURE_VERSION}, tmp_path)
            os.replace(tmp_path, MODEL_PATH)  # écriture atomique
        except Exception:
            logger.exception("Impossible d'enregistrer le modèle ML sur disque")
        return model, len(X)


def ml_prediction(home_key: str, away_key: str, hp: dict, ap: dict) -> Optional[dict]:
    model, rows = train_ml_model()
    if model is None:
        return None
    ratings = elo_ratings()
    elo_diff = ratings.get(home_key, 1500.0) - ratings.get(away_key, 1500.0)
    x = [ml_row(hp, ap, elo_diff)]
    try:
        p = model.predict_proba(x)[0]
        classes = list(model.named_steps["logisticregression"].classes_)
        values = {int(c): float(v) for c, v in zip(classes, p)}
        return {
            "home_win": values.get(0, 0.0) * 100,
            "draw": values.get(1, 0.0) * 100,
            "away_win": values.get(2, 0.0) * 100,
            "training_matches": rows,
        }
    except Exception:
        logger.exception("ML prediction failed")
        return None


def calculate_surgical_consensus(home_key: str, away_key: str, league_code="eng.1") -> dict:
    details = build_features(home_key, away_key, league_code)
    poisson = poisson_model(details["home_xg"], details["away_xg"])

    elo_diff = details["home_elo"] + HOME_ELO_ADV - details["away_elo"]
    home_elo_prob = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
    draw_base = max(0.16, min(0.30, 0.27 - abs(details["home_xg"] - details["away_xg"]) * 0.035))
    elo_probs = {
        "home_win": home_elo_prob * (1 - draw_base) * 100,
        "draw": draw_base * 100,
        "away_win": (1 - home_elo_prob) * (1 - draw_base) * 100,
    }

    ml = ml_prediction(home_key, away_key, details["home_profile"], details["away_profile"])
    if ml:
        weights = (0.55, 0.20, 0.25)
        method = "Poisson + Elo + LogisticRegression entraînée"
        ml_probs = ml
    else:
        weights = (0.72, 0.28, 0.0)
        method = "Poisson + Elo (historique ML insuffisant)"
        ml_probs = None

    final = {}
    for key in ("home_win", "draw", "away_win"):
        final[key] = (poisson[key] * weights[0] + elo_probs[key] * weights[1] +
                      ((ml_probs[key] if ml_probs else 0.0) * weights[2]))
    total = sum(final.values()) or 1.0
    final = {k: round(v * 100 / total, 1) for k, v in final.items()}

    return {
        **final,
        "home_exp": details["home_xg"],
        "away_exp": details["away_xg"],
        "btts_prob": round(poisson["btts"], 1),
        "over_25_prob": round(poisson["over25"], 1),
        "top_exact_scores": poisson["scores"][:3],
        "all_exact_scores": poisson["scores"],
        "model": method,
        "ml_active": bool(ml),
        "ml_training_matches": ml["training_matches"] if ml else 0,
        "data": details,
    }


def implied_value(probability: float, decimal_odds: Optional[float]) -> dict:
    if not decimal_odds or decimal_odds <= 1.0:
        return {"available": False, "message": "Cote non fournie"}
    implied = 100.0 / decimal_odds
    edge = probability - implied
    return {
        "available": True,
        "implied_probability": round(implied, 1),
        "model_probability": round(probability, 1),
        "edge_percentage_points": round(edge, 1),
        "positive_value": edge > 0,
        "expected_return_per_1_unit": round((probability / 100.0) * decimal_odds - 1.0, 3),
    }


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    if os.getenv("DISABLE_STARTUP_BACKFILL", "0") != "1":
        threading.Thread(target=run_backfill, args=(BACKFILL_DAYS,), daemon=True).start()
    yield


app = FastAPI(
    title=f"Signal Fort - Version {APP_VERSION} | Real Data + Predictive Models",
    version=APP_VERSION,
    description="Football predictions based on ESPN results, SQLite history, Elo, Poisson and optional trained machine learning.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # incompatible avec allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)


class MatchRequest(BaseModel):
    home_team: str = Field(min_length=1, max_length=100)
    away_team: str = Field(min_length=1, max_length=100)
    home_id: Optional[str] = None
    away_id: Optional[str] = None
    event_id: Optional[str] = None
    league_code: Optional[str] = "eng.1"


class ValueRequest(BaseModel):
    probability: float = Field(ge=0, le=100)
    odds: float = Field(gt=1.0)


@app.get("/", response_class=HTMLResponse)
def serve_home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"""<html><head><meta charset='utf-8'><title>Signal Fort {APP_VERSION}</title></head>
        <body style='font-family:Arial;padding:30px'><h1>Signal Fort {APP_VERSION}</h1>
        <p>Backend actif. Place ton fichier index.html dans le même dossier.</p>
        <p><a href='/docs'>Ouvrir la documentation API</a></p></body></html>"""


@app.get("/health")
def health():
    with db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM matches_history").fetchone()[0]
        teams = conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0]
        unlinked = conn.execute(
            "SELECT COUNT(*) FROM matches_history WHERE home_id IS NULL OR away_id IS NULL"
        ).fetchone()[0]
    with cache_lock:
        model_in_memory = ml_model_cache["model"] is not None
    return {
        "status": "ok", "version": APP_VERSION,
        "historical_matches": count,
        "known_teams": teams,
        "matches_without_team_id": unlinked,
        "sklearn_available": SKLEARN_AVAILABLE,
        "ml_active": model_in_memory or os.path.exists(MODEL_PATH),
        "backfill_running": backfill_lock.locked(),
    }


@app.get("/api/history")
def history(limit: int = 50):
    limit = max(1, min(limit, 500))
    with db() as conn:
        rows = conn.execute("""
            SELECT event_id, league, home_team, away_team, home_id, away_id,
                   home_score, away_score, date
            FROM matches_history ORDER BY date DESC LIMIT ?
        """, (limit,)).fetchall()
    return {"count": len(rows), "matches": [dict(r) for r in rows]}


@app.get("/api/teams")
def search_teams(q: str = "", limit: int = 20):
    """Recherche d'équipes connues (id ESPN + nom), pour vérifier le rattachement."""
    limit = max(1, min(limit, 100))
    norm = normalize_name(q)
    with db() as conn:
        rows = conn.execute(
            "SELECT team_id, name FROM teams WHERE norm_name LIKE ? ORDER BY name LIMIT ?",
            (f"%{norm}%", limit),
        ).fetchall()
    return {"count": len(rows), "teams": [dict(r) for r in rows]}


@app.get("/api/matches/foot")
def get_foot_matches(date: str):
    try:
        requested = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Date invalide. Format attendu : YYYY-MM-DD")

    cached = cache_get(matches_cache, date)
    if cached is not None:
        return cached

    dates_to_check = [
        (requested - timedelta(days=1)).strftime("%Y%m%d"),
        requested.strftime("%Y%m%d"),
        (requested + timedelta(days=1)).strftime("%Y%m%d"),
    ]
    found = []
    seen = set()
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            pool.submit(fetch_scoreboard, league_code, date_key): (league_code, date_key)
            for league_code in LEAGUES for date_key in dates_to_check
        }
        for future in as_completed(futures):
            league_code, _ = futures[future]
            try:
                events = future.result()
            except Exception as exc:
                logger.warning("ESPN request failed for %s: %s", league_code, exc)
                continue
            for event in events:
                item = parse_event(event, league_code)
                if item and item["date"] == date and item["event_id"] not in seen:
                    seen.add(item["event_id"])
                    found.append(item)

    found.sort(key=lambda x: x.get("_utc_date", ""))
    for item in found:
        item.pop("_utc_date", None)
    result = {"date": date, "matches": found, "source": "ESPN"}
    cache_set(matches_cache, date, result)
    return result


@app.post("/predict/match/foot")
def predict_foot_match(req: MatchRequest):
    home_team = req.home_team.strip()
    away_team = req.away_team.strip()
    league_code = req.league_code or "eng.1"
    if home_team.casefold() == away_team.casefold():
        raise HTTPException(status_code=400, detail="Les deux équipes doivent être différentes.")

    home_key = resolve_team_key(home_team, req.home_id)
    away_key = resolve_team_key(away_team, req.away_id)
    if home_key == away_key:
        raise HTTPException(status_code=400, detail="Les deux équipes doivent être différentes.")

    cache_key = f"{home_key}|{away_key}|{league_code}"
    cached = cache_get(prediction_cache, cache_key)
    if cached is not None:
        return cached

    try:
        consensus = calculate_surgical_consensus(home_key, away_key, league_code)
        probs = {
            "home_win": consensus["home_win"],
            "draw": consensus["draw"],
            "away_win": consensus["away_win"],
        }

        predicted_score = consensus["top_exact_scores"][0]["score"]
        max_prob = max(probs.values())
        if max_prob >= 60:
            signal_strength, signal_badge = "FORT", "🔥 Signal Fort"
        elif max_prob >= 48:
            signal_strength, signal_badge = "MODÉRÉ", "⚡ Tendance modérée"
        else:
            signal_strength, signal_badge = "INCERTAIN", "⚠️ Match indécis"

        hp = consensus["data"]["home_profile"]
        ap = consensus["data"]["away_profile"]
        total_matches = hp["matches"] + ap["matches"]
        data_quality = "bonne" if min(hp["matches"], ap["matches"]) >= 8 else (
            "moyenne" if min(hp["matches"], ap["matches"]) >= 3 else "limitée"
        )

        result = {
            "match": f"{home_team} vs {away_team}",
            "teams": {
                "home": {"name": home_team, "espn_id": None if home_key.startswith("n:") else home_key},
                "away": {"name": away_team, "espn_id": None if away_key.startswith("n:") else away_key},
            },
            "probabilities": probs,
            "signal_analysis": {
                "strength": signal_strength,
                "badge": signal_badge,
                "confidence_index": max_prob,
                "data_quality": data_quality,
                "historical_matches_available": total_matches,
                "model": consensus["model"],
            },
            "predicted_score": predicted_score,
            "top_exact_scores": consensus["top_exact_scores"],
            "btts": {"yes_prob": consensus["btts_prob"]},
            "over_under_2_5": {"over_prob": consensus["over_25_prob"]},
            "goalscorers": {
                "home_player": None,
                "home_player_prob": None,
                "away_player": None,
                "away_player_prob": None,
                "available": False,
                "message": "Aucune statistique fiable de buteurs disponible pour ce match via la source actuelle.",
            },
            "first_half": {
                "over_0_5_prob": None,
                "predicted_ht_score": None,
                "action_conseillee": "Données de première mi-temps non disponibles : ne pas extrapoler.",
                "available": False,
            },
            "yellow_cards": {
                "expected_total": None,
                "over_4_5_prob": None,
                "home_team_avg": None,
                "away_team_avg": None,
                "available": False,
                "message": "Statistiques de cartons non disponibles de façon homogène via ESPN.",
            },
            "expected_goals": {
                "home": consensus["home_exp"],
                "away": consensus["away_exp"],
                "total": round(consensus["home_exp"] + consensus["away_exp"], 2),
            },
            "team_statistics": {
                "home": hp,
                "away": ap,
                "elo_home": consensus["data"]["home_elo"],
                "elo_away": consensus["data"]["away_elo"],
            },
            "value_betting": {
                "home_win": {"available": False, "message": "Fournir la cote décimale pour calculer la value."},
                "draw": {"available": False, "message": "Fournir la cote décimale pour calculer la value."},
                "away_win": {"available": False, "message": "Fournir la cote décimale pour calculer la value."},
            },
            "model_details": {
                "name": consensus["model"],
                "machine_learning_active": consensus["ml_active"],
                "machine_learning_training_matches": consensus["ml_training_matches"],
                "data_quality": data_quality,
                "source": "ESPN + historique SQLite local",
            },
        }
        cache_set(prediction_cache, cache_key, result)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/match/foot/value")
def predict_value(req: ValueRequest):
    return implied_value(req.probability, req.odds)


@app.get("/api/backtest")
def backtest(test_ratio: float = 0.2):
    """Backtest du modèle ML : entraînement sur le passé, test sur la période suivante.
    Compare au baseline 'fréquences historiques' (plus bas = mieux pour log-loss et Brier).
    Note : class_weight='balanced' décale les probabilités ; elles ne sont pas calibrées."""
    if not SKLEARN_AVAILABLE:
        raise HTTPException(status_code=503, detail="scikit-learn n'est pas installé.")
    test_ratio = max(0.1, min(test_ratio, 0.5))

    X, y = build_ml_dataset()
    n = len(X)
    split = int(n * (1 - test_ratio))
    if split < MIN_ML_ROWS or n - split < 10 or len(set(y[:split])) < 3:
        raise HTTPException(
            status_code=400,
            detail=f"Historique insuffisant pour un backtest ({n} matchs). Lance /api/history/backfill avec plus de jours.",
        )

    model = make_model()
    model.fit(X[:split], y[:split])
    classes = [int(c) for c in model.named_steps["logisticregression"].classes_]

    y_test = y[split:]
    probs = []
    for p in model.predict_proba(X[split:]):
        full = [0.0, 0.0, 0.0]
        for c, v in zip(classes, p):
            full[c] = float(v)
        probs.append(full)

    freq = [y[:split].count(c) / split for c in range(3)]
    baseline = [freq] * len(y_test)

    def log_loss(ps):
        return -sum(math.log(max(1e-12, p[t])) for p, t in zip(ps, y_test)) / len(y_test)

    def brier(ps):
        return sum(sum((p[c] - (1.0 if c == t else 0.0)) ** 2 for c in range(3))
                   for p, t in zip(ps, y_test)) / len(y_test)

    accuracy = sum(1 for p, t in zip(probs, y_test)
                   if max(range(3), key=lambda c: p[c]) == t) / len(y_test)
    baseline_accuracy = sum(1 for t in y_test if t == max(range(3), key=lambda c: freq[c])) / len(y_test)

    return {
        "train_matches": split,
        "test_matches": len(y_test),
        "model": {"log_loss": round(log_loss(probs), 4), "brier": round(brier(probs), 4),
                  "accuracy": round(accuracy, 4)},
        "baseline_frequencies": {"log_loss": round(log_loss(baseline), 4),
                                 "brier": round(brier(baseline), 4),
                                 "accuracy": round(baseline_accuracy, 4)},
        "uniform_log_loss_reference": round(math.log(3), 4),
    }


@app.post("/api/history/backfill")
def start_backfill(days: int = BACKFILL_DAYS, leagues: Optional[str] = None):
    """days : 1 à 1095. leagues : codes séparés par des virgules (ex. 'eng.1,fra.1'), toutes par défaut.
    Reprenable : relancer la route ne retélécharge pas les jours déjà terminés."""
    days = max(1, min(days, BACKFILL_MAX_DAYS))
    selected = None
    if leagues:
        selected = [c.strip() for c in leagues.split(",") if c.strip() in LEAGUES]
        if not selected:
            raise HTTPException(status_code=400, detail=f"Aucune ligue valide. Codes possibles : {', '.join(LEAGUES)}")
    if backfill_lock.locked():
        return {"started": False, "days": days, "message": "Un chargement historique est déjà en cours."}
    threading.Thread(target=run_backfill, args=(days, selected), daemon=True).start()
    return {"started": True, "days": days, "leagues": selected or list(LEAGUES),
            "message": "Chargement historique lancé en arrière-plan. Suivi : /api/history/backfill/status"}


@app.get("/api/history/backfill/status")
def backfill_progress():
    with cache_lock:
        status = dict(backfill_status)
    total = status["total"]
    status["progress_percent"] = round(status["completed"] * 100 / total, 1) if total else 100.0
    return status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
