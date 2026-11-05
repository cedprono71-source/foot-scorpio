import os
import re
import bisect
import json
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

# PostgreSQL (Neon) : base permanente. Sans DATABASE_URL, l'app utilise SQLite (effacé au redémarrage sur Render gratuit).
try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool
    PSYCOPG2_AVAILABLE = True
except Exception:
    PSYCOPG2_AVAILABLE = False

APP_VERSION = "36.0"
DB_PATH = os.getenv("FOOTBALL_DB_PATH", "football_history.db")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DB_POOL_MAX = int(os.getenv("DB_POOL_MAX", "10"))
MODEL_PATH = os.getenv("MODEL_PATH", "ml_model.joblib")
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "ensemble_weights.json")
PARIS_TZ = ZoneInfo("Europe/Paris")
UTC_TZ = ZoneInfo("UTC")
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
REQUEST_TIMEOUT = 8
BACKFILL_DAYS = int(os.getenv("BACKFILL_DAYS", "90"))
BACKFILL_MAX_DAYS = 1095          # ~3 saisons
BACKFILL_WORKERS = int(os.getenv("BACKFILL_WORKERS", "6"))
MIN_ML_ROWS = 80
MODEL_TTL = 3600                  # durée de vie du modèle ML (secondes)
INSUFFICIENT_RETRY = 300          # délai avant de retenter un entraînement impossible
ML_FEATURE_VERSION = 3            # à incrémenter si les features ou le modèle changent
HOME_ELO_ADV = 55.0
DC_RHO = float(os.getenv("DC_RHO", "-0.08"))   # correction Dixon-Coles (négatif = plus de 0-0 / 1-1)
LEAGUE_PRIOR_WEIGHT = 20          # pseudo-matchs pour la moyenne de ligue du backtest walk-forward
# Poids par défaut de l'ensemble (Dixon-Coles, Poisson forme, Elo, ML). Remplacés par ceux de /api/backtest/ensemble?save=true.
DEFAULT_WEIGHTS = {
    "with_ml": {"dc": 0.45, "poisson": 0.10, "elo": 0.15, "ml": 0.30},
    "without_ml": {"dc": 0.60, "poisson": 0.15, "elo": 0.25},
}
DC_HALF_LIFE_DAYS = float(os.getenv("DC_HALF_LIFE_DAYS", "365"))  # un match perd la moitié de son poids en ~1 an
DC_REG = float(os.getenv("DC_REG", "4.0"))                        # ridge : ramène les équipes peu vues vers la moyenne
DC_TTL = 1800                     # ré-ajustement Dixon-Coles au plus toutes les 30 min
MIN_DC_MATCHES = 150              # en dessous, Dixon-Coles est désactivé

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
weights_cache = {"at": 0.0, "weights": DEFAULT_WEIGHTS}
dc_cache = {"model": None, "version": -1, "fitted_at": 0.0, "retry_after": 0.0}
data_version = {"v": 0}  # incrémenté dès que l'historique change réellement
backfill_status = {"running": False, "days": 0, "total": 0, "completed": 0,
                   "errors": 0, "started_at": None, "finished_at": None}
cache_lock = threading.Lock()
train_lock = threading.Lock()
backfill_lock = threading.Lock()
dc_lock = threading.Lock()


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
USE_PG = False
DB_STATUS = "sqlite"
_pg_pool = None
_pool_sem = threading.BoundedSemaphore(DB_POOL_MAX)


def _setup_backend():
    """Choisit PostgreSQL si DATABASE_URL est défini et joignable, sinon SQLite (avec un message clair)."""
    global USE_PG, DB_STATUS, _pg_pool
    if not DATABASE_URL:
        return
    if not PSYCOPG2_AVAILABLE:
        logger.error("DATABASE_URL défini mais psycopg2 absent : ajoute psycopg2-binary à requirements.txt. "
                     "Repli sur SQLite (données NON permanentes).")
        DB_STATUS = "sqlite (secours : psycopg2 manquant)"
        return
    try:
        pool = psycopg2.pool.ThreadedConnectionPool(1, DB_POOL_MAX, DATABASE_URL, connect_timeout=15)
        raw = pool.getconn()
        try:
            with raw.cursor() as cur:
                cur.execute("SELECT 1")
            raw.rollback()
        finally:
            pool.putconn(raw)
        _pg_pool, USE_PG, DB_STATUS = pool, True, "postgresql"
        logger.info("Base PostgreSQL connectée")
    except Exception:
        logger.exception("Connexion PostgreSQL impossible : repli sur SQLite (données NON permanentes)")
        DB_STATUS = "sqlite (secours : connexion PostgreSQL échouée)"


class PgConn:
    """Même interface que sqlite3 pour le code de l'appli : execute() avec des ? et lignes accessibles par nom ou index."""

    def __init__(self, raw):
        self.raw = raw

    def execute(self, sql, params=()):
        cur = self.raw.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(sql.replace("?", "%s"), params)
        return cur

    def executemany(self, sql, seq):
        cur = self.raw.cursor()
        cur.executemany(sql.replace("?", "%s"), list(seq))
        return cur


def _pg_checkout():
    """Prend une connexion du pool et vérifie qu'elle répond (Neon coupe les connexions inactives)."""
    last_exc = None
    for _ in range(3):
        raw = _pg_pool.getconn()
        try:
            with raw.cursor() as cur:
                cur.execute("SELECT 1")
            raw.rollback()
            return raw
        except Exception as exc:
            last_exc = exc
            try:
                _pg_pool.putconn(raw, close=True)
            except Exception:
                pass
    raise last_exc  # type: ignore[misc]


_setup_backend()
# "IS NOT" (SQLite) et "IS DISTINCT FROM" (PostgreSQL) : comparaison qui tient compte des NULL.
NE = "IS DISTINCT FROM" if USE_PG else "IS NOT"


def db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=20, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=20000")
    return conn


@contextmanager
def db():
    """Connexion avec commit/rollback ET restitution (pool PostgreSQL) ou fermeture (SQLite)."""
    if USE_PG:
        with _pool_sem:
            raw = _pg_checkout()
            try:
                yield PgConn(raw)
                raw.commit()
            except Exception:
                try:
                    raw.rollback()
                except Exception:
                    pass
                raise
            finally:
                try:
                    _pg_pool.putconn(raw)
                except Exception:
                    pass
        return
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
        if not USE_PG:
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions_log (
                event_id TEXT PRIMARY KEY,
                created_at TEXT, updated_at TEXT,
                league_code TEXT, home_team TEXT, away_team TEXT, home_key TEXT, away_key TEXT,
                p_home DOUBLE PRECISION, p_draw DOUBLE PRECISION, p_away DOUBLE PRECISION,
                model TEXT, app_version TEXT, components TEXT,
                outcome INTEGER, home_score INTEGER, away_score INTEGER, settled_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_settings (
                setting_key TEXT PRIMARY KEY,
                setting_value TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predlog_outcome ON predictions_log(outcome)")
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
            cur = conn.execute(f"""
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
                WHERE matches_history.league {NE} excluded.league
                   OR matches_history.home_team {NE} excluded.home_team
                   OR matches_history.away_team {NE} excluded.away_team
                   OR matches_history.home_score {NE} excluded.home_score
                   OR matches_history.away_score {NE} excluded.away_score
                   OR matches_history.date {NE} excluded.date
                   OR matches_history.home_id {NE} excluded.home_id
                   OR matches_history.away_id {NE} excluded.away_id
            """, (str(event_id), league, home_team, away_team,
                  int(home_score), int(away_score), str(date)[:10], home_id, away_id))
            changed = cur.rowcount > 0
            for team_id, name in ((home_id, home_team), (away_id, away_team)):
                if team_id:
                    conn.execute(f"""
                        INSERT INTO teams (team_id, name, norm_name) VALUES (?, ?, ?)
                        ON CONFLICT(team_id) DO UPDATE SET
                            name = excluded.name, norm_name = excluded.norm_name
                        WHERE teams.name {NE} excluded.name
                    """, (team_id, name, normalize_name(name)))
            # Règle la prédiction enregistrée avant le match, s'il y en a une.
            hs, as_ = int(home_score), int(away_score)
            conn.execute("""
                UPDATE predictions_log SET outcome = ?, home_score = ?, away_score = ?, settled_at = ?
                WHERE event_id = ? AND outcome IS NULL
            """, (0 if hs > as_ else 1 if hs == as_ else 2, hs, as_,
                  datetime.now(UTC_TZ).isoformat(), str(event_id)))
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
            conn.executemany("INSERT INTO backfill_log (league_code, day) VALUES (?, ?) ON CONFLICT DO NOTHING",
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
            SELECT league, home_team, away_team, home_score, away_score, date, {KEY_COLUMNS}
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


def dixon_coles_tau(hg: int, ag: int, home_xg: float, away_xg: float, rho: float) -> float:
    """Correction Dixon-Coles : Poisson indépendant sous-estime 0-0 et 1-1, surestime 1-0 et 0-1."""
    if hg == 0 and ag == 0:
        tau = 1.0 - home_xg * away_xg * rho
    elif hg == 0 and ag == 1:
        tau = 1.0 + home_xg * rho
    elif hg == 1 and ag == 0:
        tau = 1.0 + away_xg * rho
    elif hg == 1 and ag == 1:
        tau = 1.0 - rho
    else:
        return 1.0
    return max(0.0, tau)


def poisson_model(home_xg: float, away_xg: float, rho: float = DC_RHO) -> dict:
    matrix = {}
    total = home_win = draw = away_win = btts = over25 = 0.0
    for hg in range(8):
        for ag in range(8):
            p = (poisson_p(home_xg, hg) * poisson_p(away_xg, ag)
                 * dixon_coles_tau(hg, ag, home_xg, away_xg, rho))
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


def _shrink(value: float, n: int, prior: float, weight: float = 5.0) -> float:
    return (value * n + prior * weight) / (n + weight)


def match_components(hp: dict, ap: dict, home_elo: float, away_elo: float,
                     home_avg: float, away_avg: float) -> dict:
    """Unique définition des modèles Poisson et Elo : utilisée en prédiction ET en backtest."""
    n_h, n_a = hp.get("matches", 0), ap.get("matches", 0)
    h_attack = _shrink(hp["goals_for_avg"], n_h, home_avg)
    h_defence = _shrink(hp["goals_against_avg"], n_h, away_avg)
    a_attack = _shrink(ap["goals_for_avg"], n_a, away_avg)
    a_defence = _shrink(ap["goals_against_avg"], n_a, home_avg)

    home_xg = max(0.20, min(4.0, (h_attack * a_defence / max(0.5, away_avg)) * 1.08))
    away_xg = max(0.20, min(4.0, (a_attack * h_defence / max(0.5, home_avg)) * 0.94))
    poisson = poisson_model(home_xg, away_xg)

    elo_diff = home_elo + HOME_ELO_ADV - away_elo
    home_elo_prob = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
    draw_base = max(0.16, min(0.30, 0.27 - abs(home_xg - away_xg) * 0.035))
    elo_probs = {
        "home_win": home_elo_prob * (1 - draw_base) * 100,
        "draw": draw_base * 100,
        "away_win": (1 - home_elo_prob) * (1 - draw_base) * 100,
    }
    return {"home_xg": home_xg, "away_xg": away_xg, "poisson": poisson, "elo": elo_probs}


# --------------------------------------------------------------------------- #
# Dixon-Coles complet : attaque / défense par équipe, matchs récents plus pesants
# --------------------------------------------------------------------------- #
class DCModel:
    """log(buts domicile)    = mu[ligue] + ha[ligue] + attaque[dom] - défense[ext]
       log(buts extérieur)   = mu[ligue] + attaque[ext] - défense[dom]"""

    def __init__(self, mu, ha, att, de, rho, n_matches):
        self.mu, self.ha, self.att, self.de, self.rho = mu, ha, att, de, rho
        self.n_matches = n_matches
        self.n_teams = len(att)
        self.mu_default = sum(mu.values()) / len(mu) if mu else math.log(1.2)
        self.ha_default = sum(ha.values()) / len(ha) if ha else 0.2

    def expected_goals(self, home_key: str, away_key: str, league_name: str) -> Tuple[float, float]:
        mu = self.mu.get(league_name, self.mu_default)
        ha = self.ha.get(league_name, self.ha_default)
        lh = math.exp(mu + ha + self.att.get(home_key, 0.0) - self.de.get(away_key, 0.0))
        la = math.exp(mu + self.att.get(away_key, 0.0) - self.de.get(home_key, 0.0))
        return max(0.15, min(4.5, lh)), max(0.15, min(4.5, la))

    def predict(self, home_key: str, away_key: str, league_name: str) -> dict:
        lh, la = self.expected_goals(home_key, away_key, league_name)
        result = poisson_model(lh, la, self.rho)
        result["home_xg"], result["away_xg"] = lh, la
        return result


def to_dc_tuples(rows) -> List[tuple]:
    """(jour, ligue, clé dom, clé ext, buts dom, buts ext), dans l'ordre des lignes reçues."""
    out, last_day = [], 0
    for r in rows:
        try:
            day = datetime.fromisoformat(str(r["date"])[:10]).toordinal()
        except ValueError:
            day = last_day
        last_day = day
        out.append((day, r["league"], r["home_key"], r["away_key"],
                    int(r["home_score"]), int(r["away_score"])))
    return out


def fit_dixon_coles(data: List[tuple], ref_day: int, half_life: float = DC_HALF_LIFE_DAYS,
                    reg: float = DC_REG, max_sweeps: int = 40) -> Optional[DCModel]:
    """Régression de Poisson pondérée par l'ancienneté (demi-vie en jours), ajustée par blocs.
    Chaque taux attendu ne dépend que d'un seul paramètre d'attaque et d'un seul de défense :
    les mises à jour par bloc sont donc exactes. La régularisation ridge (reg) ramène les équipes
    peu observées vers la moyenne de leur ligue."""
    xi = math.log(2) / max(1.0, half_life)
    H, A, L, W, HG, AG = [], [], [], [], [], []
    team_idx: Dict[str, int] = {}
    league_idx: Dict[str, int] = {}
    for day, league, hk, ak, hg, ag in data:
        w = math.exp(-xi * max(0, ref_day - day))
        if w < 0.01:
            continue
        H.append(team_idx.setdefault(hk, len(team_idx)))
        A.append(team_idx.setdefault(ak, len(team_idx)))
        L.append(league_idx.setdefault(league, len(league_idx)))
        W.append(w)
        HG.append(hg)
        AG.append(ag)
    N, T, NL = len(W), len(team_idx), len(league_idx)
    if N < MIN_DC_MATCHES:
        return None

    sw = [0.0] * NL
    sh = [0.0] * NL
    sa = [0.0] * NL
    for i in range(N):
        sw[L[i]] += W[i]
        sh[L[i]] += W[i] * HG[i]
        sa[L[i]] += W[i] * AG[i]
    mu = [math.log(max(0.3, sa[l] / sw[l])) if sw[l] > 0 else 0.0 for l in range(NL)]
    tot_h, tot_a = sum(sh), sum(sa)
    ha0 = math.log(max(0.5, tot_h) / max(0.5, tot_a))
    mu0 = list(mu)
    ha = [ha0] * NL
    att = [0.0] * T
    de = [0.0] * T
    ha_reg = 5.0   # une ligue peu fournie garde un avantage du domicile proche de la moyenne globale
    mu_reg = 1.0

    def rates():
        lh, la = [0.0] * N, [0.0] * N
        for i in range(N):
            base = mu[L[i]]
            lh[i] = math.exp(base + ha[L[i]] + att[H[i]] - de[A[i]])
            la[i] = math.exp(base + att[A[i]] - de[H[i]])
        return lh, la

    def clip(x, lim=0.5):
        return max(-lim, min(lim, x))

    for _ in range(max_sweeps):
        max_delta = 0.0

        lh, la = rates()
        g, hs = [0.0] * T, [0.0] * T
        for i in range(N):
            w = W[i]
            g[H[i]] += w * (HG[i] - lh[i])
            hs[H[i]] += w * lh[i]
            g[A[i]] += w * (AG[i] - la[i])
            hs[A[i]] += w * la[i]
        for t in range(T):
            step = clip((g[t] - reg * att[t]) / (hs[t] + reg))
            att[t] += step
            max_delta = max(max_delta, abs(step))

        lh, la = rates()
        g, hs = [0.0] * T, [0.0] * T
        for i in range(N):
            w = W[i]
            g[A[i]] += w * (lh[i] - HG[i])
            hs[A[i]] += w * lh[i]
            g[H[i]] += w * (la[i] - AG[i])
            hs[H[i]] += w * la[i]
        for t in range(T):
            step = clip((g[t] - reg * de[t]) / (hs[t] + reg))
            de[t] += step
            max_delta = max(max_delta, abs(step))

        lh, la = rates()
        g, hs = [0.0] * NL, [0.0] * NL
        for i in range(N):
            w = W[i]
            g[L[i]] += w * (HG[i] - lh[i] + AG[i] - la[i])
            hs[L[i]] += w * (lh[i] + la[i])
        for l in range(NL):
            step = clip((g[l] - mu_reg * (mu[l] - mu0[l])) / (hs[l] + mu_reg), 0.3)
            mu[l] += step
            max_delta = max(max_delta, abs(step))

        lh, la = rates()
        g, hs = [0.0] * NL, [0.0] * NL
        for i in range(N):
            w = W[i]
            g[L[i]] += w * (HG[i] - lh[i])
            hs[L[i]] += w * lh[i]
        for l in range(NL):
            step = clip((g[l] - ha_reg * (ha[l] - ha0)) / (hs[l] + ha_reg), 0.3)
            ha[l] += step
            max_delta = max(max_delta, abs(step))

        if max_delta < 1e-4:
            break

    # rho (corrélation des petits scores) : balayage sur la vraisemblance des scores 0-0, 0-1, 1-0, 1-1.
    lh, la = rates()
    low = [(W[i], lh[i], la[i], HG[i], AG[i]) for i in range(N) if HG[i] <= 1 and AG[i] <= 1]
    best_rho, best_ll = 0.0, -float("inf")
    for k in range(-30, 11):
        rho = k / 100.0
        ll, ok = 0.0, True
        for w, a, b, x, y in low:
            tau = dixon_coles_tau(x, y, a, b, rho)
            if tau <= 0:
                ok = False
                break
            ll += w * math.log(tau)
        if ok and ll > best_ll:
            best_rho, best_ll = rho, ll

    inv_team = {v: k for k, v in team_idx.items()}
    inv_league = {v: k for k, v in league_idx.items()}
    return DCModel(
        mu={inv_league[l]: mu[l] for l in range(NL)},
        ha={inv_league[l]: ha[l] for l in range(NL)},
        att={inv_team[t]: att[t] for t in range(T)},
        de={inv_team[t]: de[t] for t in range(T)},
        rho=best_rho, n_matches=N,
    )


def get_dc_model() -> Optional[DCModel]:
    """Modèle Dixon-Coles en cache : réajusté au plus une fois par DC_TTL secondes quand l'historique change."""
    with dc_lock:
        now = time.time()
        with cache_lock:
            version = data_version["v"]
        if dc_cache["model"] is not None and (dc_cache["version"] == version
                                               or now - dc_cache["fitted_at"] < DC_TTL):
            return dc_cache["model"]
        if dc_cache["model"] is None and now < dc_cache["retry_after"]:
            return None
        try:
            data = to_dc_tuples(load_finished_matches())
            model = fit_dixon_coles(data, datetime.now(PARIS_TZ).date().toordinal())
        except Exception:
            logger.exception("Dixon-Coles fit failed")
            model = None
        if model is None:
            dc_cache.update(retry_after=now + INSUFFICIENT_RETRY, version=version, fitted_at=now)
            return dc_cache["model"]
        dc_cache.update(model=model, version=version, fitted_at=now, retry_after=0.0)
        logger.info("Dixon-Coles ajusté : %s matchs, %s équipes, rho=%.2f",
                    model.n_matches, model.n_teams, model.rho)
        return model


def dc_prediction(home_key: str, away_key: str, league_code: str) -> Optional[dict]:
    model = get_dc_model()
    if model is None:
        return None
    return model.predict(home_key, away_key, LEAGUES.get(league_code, league_code))


def dc_walk_forward(tuples: List[tuple], start_idx: int, refit_days: int = 30) -> Dict[int, List[float]]:
    """Prédictions Dixon-Coles hors échantillon à partir de start_idx : le modèle est ré-ajusté tous les
    refit_days jours, uniquement sur les matchs joués AVANT le premier match du bloc."""
    n = len(tuples)
    days = [t[0] for t in tuples]
    preds: Dict[int, List[float]] = {}
    i = start_idx
    while i < n:
        j = i
        while j < n and days[j] < days[i] + refit_days:
            j += 1
        cut = bisect.bisect_left(days, days[i])
        model = fit_dixon_coles(tuples[:cut], days[i]) if cut > 0 else None
        if model is not None:
            for k in range(i, j):
                p = model.predict(tuples[k][2], tuples[k][3], tuples[k][1])
                preds[k] = [p[o] / 100.0 for o in OUTCOMES]
        i = j
    return preds


def build_features(home_key: str, away_key: str, league_code: str) -> dict:
    hp, ap = team_profile(home_key), team_profile(away_key)
    ratings = elo_ratings()
    home_elo = ratings.get(home_key, 1500.0)
    away_elo = ratings.get(away_key, 1500.0)
    home_avg, away_avg = league_goal_averages(league_code)
    comp = match_components(hp, ap, home_elo, away_elo, home_avg, away_avg)
    return {
        "home_profile": hp, "away_profile": ap,
        "home_elo": round(home_elo, 1), "away_elo": round(away_elo, 1),
        "home_xg": round(comp["home_xg"], 2), "away_xg": round(comp["away_xg"], 2),
        "poisson": comp["poisson"], "elo_probs": comp["elo"],
        "league_home_goals_avg": round(home_avg, 2),
        "league_away_goals_avg": round(away_avg, 2),
    }


# --------------------------------------------------------------------------- #
# Machine learning
# --------------------------------------------------------------------------- #
def rolling_profile(games: List[Tuple[int, int]]) -> dict:
    """Profil glissant (buts pour, buts contre) -> mêmes clés que team_profile."""
    if not games:
        return {"matches": 0, "goals_for_avg": 1.35, "goals_against_avg": 1.35, "points_per_game": 1.0}
    n = len(games)
    return {
        "matches": n,
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


OUTCOMES = ("home_win", "draw", "away_win")


def build_walk_forward(with_components: bool = False, matches=None) -> List[dict]:
    """Parcours chronologique : pour chaque match, features et prédictions Poisson/Elo calculées
    uniquement avec le passé (pas de fuite). Chaque ligne : {x, y, [poisson, elo]} (probas en 0-1)."""
    rolling: Dict[str, List[Tuple[int, int]]] = {}
    ratings: Dict[str, float] = {}
    league_sums: Dict[str, Tuple[int, float, float]] = {}
    rows = []
    for row in (matches if matches is not None else load_finished_matches()):
        h, a = row["home_key"], row["away_key"]
        hg, ag = int(row["home_score"]), int(row["away_score"])
        hp = rolling_profile(rolling.get(h, [])[-10:])
        ap = rolling_profile(rolling.get(a, [])[-10:])
        home_elo, away_elo = ratings.get(h, 1500.0), ratings.get(a, 1500.0)
        item = {"x": ml_row(hp, ap, home_elo - away_elo),
                "y": 0 if hg > ag else 1 if hg == ag else 2}
        if with_components:
            n, hs, as_ = league_sums.get(row["league"], (0, 0.0, 0.0))
            home_avg = max(0.65, (hs + 1.45 * LEAGUE_PRIOR_WEIGHT) / (n + LEAGUE_PRIOR_WEIGHT))
            away_avg = max(0.55, (as_ + 1.20 * LEAGUE_PRIOR_WEIGHT) / (n + LEAGUE_PRIOR_WEIGHT))
            comp = match_components(hp, ap, home_elo, away_elo, home_avg, away_avg)
            item["poisson"] = [comp["poisson"][k] / 100.0 for k in OUTCOMES]
            item["elo"] = [comp["elo"][k] / 100.0 for k in OUTCOMES]
        rows.append(item)
        rolling[h] = (rolling.get(h, []) + [(hg, ag)])[-10:]
        rolling[a] = (rolling.get(a, []) + [(ag, hg)])[-10:]
        elo_update(ratings, h, a, hg, ag)
        n, hs, as_ = league_sums.get(row["league"], (0, 0.0, 0.0))
        league_sums[row["league"]] = (n + 1, hs + hg, as_ + ag)
    return rows


def build_ml_dataset() -> Tuple[List[List[float]], List[int]]:
    rows = build_walk_forward(False)
    return [r["x"] for r in rows], [r["y"] for r in rows]


def make_model():
    # Pas de class_weight="balanced" : il décale les probabilités et les rend moins fiables.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )


def model_probas(model, X: List[List[float]]) -> List[List[float]]:
    """Probabilités (0-1) ordonnées [domicile, nul, extérieur], même si une classe manque."""
    classes = [int(c) for c in model.named_steps["logisticregression"].classes_]
    out = []
    for p in model.predict_proba(X):
        full = [0.0, 0.0, 0.0]
        for c, v in zip(classes, p):
            full[c] = float(v)
        out.append(full)
    return out


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
    try:
        p = model_probas(model, [ml_row(hp, ap, elo_diff)])[0]
        return {
            "home_win": p[0] * 100,
            "draw": p[1] * 100,
            "away_win": p[2] * 100,
            "training_matches": rows,
        }
    except Exception:
        logger.exception("ML prediction failed")
        return None


# --------------------------------------------------------------------------- #
# Poids de l'ensemble (ajustables par backtest)
# --------------------------------------------------------------------------- #
COMPONENTS_FULL = ["dc", "poisson", "elo", "ml"]
COMPONENTS_NO_ML = ["dc", "poisson", "elo"]


def _valid_weights(w, names: List[str]) -> bool:
    return (isinstance(w, dict) and set(w) == set(names)
            and all(isinstance(x, (int, float)) and x >= 0 for x in w.values())
            and abs(sum(w.values()) - 1.0) < 1e-6)


def settings_get(key: str) -> Optional[str]:
    try:
        with db() as conn:
            row = conn.execute("SELECT setting_value FROM app_settings WHERE setting_key = ?", (key,)).fetchone()
        return row["setting_value"] if row else None
    except Exception:
        logger.exception("settings_get failed")
        return None


def settings_set(key: str, value: str) -> None:
    with db() as conn:
        conn.execute("""
            INSERT INTO app_settings (setting_key, setting_value) VALUES (?, ?)
            ON CONFLICT(setting_key) DO UPDATE SET setting_value = excluded.setting_value
        """, (key, value))


def load_weights() -> dict:
    """Poids de l'ensemble : lus dans la base (relus au plus toutes les 60 s), sinon dans WEIGHTS_PATH,
    sinon valeurs par défaut. Un format d'une ancienne version est ignoré."""
    now = time.time()
    with cache_lock:
        if now - weights_cache["at"] < 60:
            return weights_cache["weights"]
    data = None
    try:
        raw = settings_get("ensemble_weights")
        if raw:
            data = json.loads(raw)
        elif os.path.exists(WEIGHTS_PATH):
            with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
    except Exception:
        logger.exception("Poids d'ensemble illisibles, valeurs par défaut utilisées")
    weights = {k: dict(v) for k, v in DEFAULT_WEIGHTS.items()}
    if isinstance(data, dict):
        if _valid_weights(data.get("with_ml"), COMPONENTS_FULL):
            weights["with_ml"] = {k: float(v) for k, v in data["with_ml"].items()}
        if _valid_weights(data.get("without_ml"), COMPONENTS_NO_ML):
            weights["without_ml"] = {k: float(v) for k, v in data["without_ml"].items()}
    with cache_lock:
        weights_cache.update(at=now, weights=weights)
    return weights


def combine_components(components: Dict[str, dict], wdict: Dict[str, float]) -> Tuple[dict, Dict[str, float]]:
    """Moyenne pondérée des modèles disponibles (poids renormalisés si l'un d'eux manque)."""
    names = [n for n in wdict if n in components]
    total = sum(wdict[n] for n in names)
    used = ({n: wdict[n] / total for n in names} if total > 0
            else {n: 1.0 / len(names) for n in names})
    final = {k: sum(used[n] * components[n][k] for n in names) for k in OUTCOMES}
    s = sum(final.values()) or 1.0
    return {k: round(v * 100 / s, 1) for k, v in final.items()}, used


def calculate_surgical_consensus(home_key: str, away_key: str, league_code="eng.1") -> dict:
    details = build_features(home_key, away_key, league_code)
    dc = dc_prediction(home_key, away_key, league_code)
    ml = ml_prediction(home_key, away_key, details["home_profile"], details["away_profile"])

    components = {"poisson": details["poisson"], "elo": details["elo_probs"]}
    if dc:
        components["dc"] = dc
    if ml:
        components["ml"] = ml
    weights = load_weights()
    final, used = combine_components(components, weights["with_ml"] if ml else weights["without_ml"])

    # Le modèle Dixon-Coles (s'il est disponible) fournit aussi xG, scores exacts, BTTS et over 2.5.
    dist = dc or details["poisson"]
    home_exp = round(dc["home_xg"], 2) if dc else details["home_xg"]
    away_exp = round(dc["away_xg"], 2) if dc else details["away_xg"]
    labels = {"dc": "Dixon-Coles", "poisson": "Poisson (forme)", "elo": "Elo", "ml": "LogisticRegression"}
    method = " + ".join(labels[n] for n in used)

    return {
        **final,
        "home_exp": home_exp,
        "away_exp": away_exp,
        "btts_prob": round(dist["btts"], 1),
        "over_25_prob": round(dist["over25"], 1),
        "top_exact_scores": dist["scores"][:3],
        "all_exact_scores": dist["scores"],
        "model": method,
        "ml_active": bool(ml),
        "dc_active": bool(dc),
        "ml_training_matches": ml["training_matches"] if ml else 0,
        "weights": {k: round(v, 2) for k, v in used.items()},
        "components": {n: {k: round(components[n][k], 1) for k in OUTCOMES} for n in used},
        "data": details,
    }


def log_prediction(event_id, league_code, home_team, away_team, home_key, away_key,
                   probs: dict, model_name: str, components: dict) -> bool:
    """Enregistre la prédiction d'un match PAS ENCORE terminé (suivi en conditions réelles).
    Elle est réécrite à chaque nouvel appel tant que le résultat est inconnu, puis réglée
    automatiquement quand le match est enregistré dans l'historique."""
    if not event_id:
        return False
    event_id = str(event_id)
    try:
        with db() as conn:
            if conn.execute("SELECT 1 FROM matches_history WHERE event_id = ?", (event_id,)).fetchone():
                return False  # résultat déjà connu : la prédiction ne serait plus honnête
            now = datetime.now(UTC_TZ).isoformat()
            comp_json = json.dumps({n: [c["home_win"], c["draw"], c["away_win"]]
                                    for n, c in components.items()})
            conn.execute("""
                INSERT INTO predictions_log
                (event_id, created_at, updated_at, league_code, home_team, away_team, home_key, away_key,
                 p_home, p_draw, p_away, model, app_version, components)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    p_home = excluded.p_home, p_draw = excluded.p_draw, p_away = excluded.p_away,
                    model = excluded.model, app_version = excluded.app_version,
                    components = excluded.components
                WHERE predictions_log.outcome IS NULL
            """, (event_id, now, now, league_code, home_team, away_team, home_key, away_key,
                  probs["home_win"], probs["draw"], probs["away_win"], model_name, APP_VERSION, comp_json))
        return True
    except Exception:
        logger.exception("Unable to log prediction %s", event_id)
        return False


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
    description="Football predictions based on ESPN results, SQLite history, Elo, Poisson (Dixon-Coles) and optional trained machine learning.",
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
        tracked = conn.execute("SELECT COUNT(*) FROM predictions_log").fetchone()[0]
    with cache_lock:
        model_in_memory = ml_model_cache["model"] is not None
    return {
        "status": "ok", "version": APP_VERSION,
        "historical_matches": count,
        "known_teams": teams,
        "matches_without_team_id": unlinked,
        "sklearn_available": SKLEARN_AVAILABLE,
        "ml_active": model_in_memory or os.path.exists(MODEL_PATH),
        "database": DB_STATUS,
        "tuned_weights": settings_get("ensemble_weights") is not None,
        "dixon_coles_active": dc_cache["model"] is not None,
        "tracked_predictions": tracked,
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
    failures = 0
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            pool.submit(fetch_scoreboard, league_code, date_key, 3): (league_code, date_key)
            for league_code in LEAGUES for date_key in dates_to_check
        }
        for future in as_completed(futures):
            league_code, _ = futures[future]
            try:
                events = future.result()
            except Exception as exc:
                failures += 1
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
    result = {"date": date, "matches": found, "source": "ESPN", "failed_requests": failures}
    # Un résultat partiel (requêtes ESPN en échec) n'est pas mis en cache : le prochain appel réessaie.
    if failures == 0:
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
        # Même affiche déjà calculée : on enregistre quand même la prédiction pour CET event_id.
        md = cached["model_details"]
        tracked = log_prediction(req.event_id, league_code, home_team, away_team, home_key, away_key,
                                 cached["probabilities"], md["name"], md["components"])
        return {**cached, "model_details": {**md, "prediction_tracked": tracked}}

    try:
        consensus = calculate_surgical_consensus(home_key, away_key, league_code)
        probs = {
            "home_win": consensus["home_win"],
            "draw": consensus["draw"],
            "away_win": consensus["away_win"],
        }

        tracked = log_prediction(req.event_id, league_code, home_team, away_team, home_key, away_key,
                                 probs, consensus["model"], consensus["components"])
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
                "ensemble_weights": consensus["weights"],
                "components": consensus["components"],
                "dixon_coles_active": consensus["dc_active"],
                "prediction_tracked": tracked,
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


# --------------------------------------------------------------------------- #
# Backtests
# --------------------------------------------------------------------------- #
def _log_loss(ps: List[List[float]], ys: List[int]) -> float:
    return -sum(math.log(max(1e-12, p[t])) for p, t in zip(ps, ys)) / len(ys)


def _brier(ps: List[List[float]], ys: List[int]) -> float:
    return sum(sum((p[c] - (1.0 if c == t else 0.0)) ** 2 for c in range(3))
               for p, t in zip(ps, ys)) / len(ys)


def _accuracy(ps: List[List[float]], ys: List[int]) -> float:
    return sum(1 for p, t in zip(ps, ys) if max(range(3), key=lambda c: p[c]) == t) / len(ys)


def _metrics(ps: List[List[float]], ys: List[int]) -> dict:
    return {"log_loss": round(_log_loss(ps, ys), 4), "brier": round(_brier(ps, ys), 4),
            "accuracy": round(_accuracy(ps, ys), 4)}


def _calibration(ps: List[List[float]], ys: List[int]) -> List[dict]:
    """Pour l'issue la plus probable : confiance annoncée vs fréquence réelle, par tranche."""
    bins = [(0, 40), (40, 50), (50, 60), (60, 70), (70, 101)]
    out = []
    for lo, hi in bins:
        sel = [(p, t) for p, t in zip(ps, ys) if lo <= max(p) * 100 < hi]
        if not sel:
            continue
        hits = sum(1 for p, t in sel if max(range(3), key=lambda c: p[c]) == t)
        out.append({"range": f"{lo}-{min(hi, 100)}%", "matches": len(sel),
                    "mean_predicted": round(sum(max(p) for p, _ in sel) / len(sel) * 100, 1),
                    "actual_hit_rate": round(hits / len(sel) * 100, 1)})
    return out


@app.get("/api/backtest")
def backtest(test_ratio: float = 0.2):
    """Backtest du modèle ML seul : entraînement sur le passé, test sur la période suivante.
    Compare au baseline 'fréquences historiques' (plus bas = mieux pour log-loss et Brier)."""
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
    y_test = y[split:]
    probs = model_probas(model, X[split:])
    freq = [y[:split].count(c) / split for c in range(3)]

    return {
        "train_matches": split,
        "test_matches": len(y_test),
        "model": _metrics(probs, y_test),
        "baseline_frequencies": _metrics([freq] * len(y_test), y_test),
        "uniform_log_loss_reference": round(math.log(3), 4),
    }


def _compositions(k: int, total: int):
    """Toutes les répartitions de `total` unités entre k poids (entiers)."""
    if k == 1:
        yield (total,)
        return
    for i in range(total + 1):
        for rest in _compositions(k - 1, total - i):
            yield (i,) + rest


@app.get("/api/backtest/ensemble")
def backtest_ensemble(save: bool = False, refit_days: int = 30):
    """Backtest walk-forward de l'ensemble complet (Dixon-Coles + Poisson forme + Elo + ML), en 3 périodes :
    60 % entraînement du ML, 20 % réglage des poids, 20 % test final (jamais utilisé pour régler).
    Dixon-Coles est ré-ajusté tous les refit_days jours sur le seul passé (peut prendre un moment).
    save=true enregistre les poids retenus dans WEIGHTS_PATH ; ils sont utilisés par /predict."""
    refit_days = max(7, min(refit_days, 120))
    matches = load_finished_matches()
    rows = build_walk_forward(True, matches)
    n = len(rows)
    t1, t2 = int(n * 0.6), int(n * 0.8)
    train, tune, test = rows[:t1], rows[t1:t2], rows[t2:]
    if len(train) < MIN_ML_ROWS or len(tune) < 10 or len(test) < 10 or len({r["y"] for r in train}) < 3:
        raise HTTPException(
            status_code=400,
            detail=f"Historique insuffisant ({n} matchs). Lance /api/history/backfill avec plus de jours.",
        )

    dc_preds = dc_walk_forward(to_dc_tuples(matches), t1, refit_days)
    for k in range(t1, n):
        rows[k]["dc"] = dc_preds.get(k, rows[k]["poisson"])  # repli sur Poisson si pas assez d'historique
    y_tune, y_test = [r["y"] for r in tune], [r["y"] for r in test]
    freq = [sum(1 for r in train if r["y"] == c) / len(train) for c in range(3)]

    model = None
    if SKLEARN_AVAILABLE:
        try:
            model = make_model()
            model.fit([r["x"] for r in train], [r["y"] for r in train])
        except Exception:
            logger.exception("Backtest ML training failed")
            model = None

    base = ["dc", "poisson", "elo"]
    tune_c = {nm: [r[nm] for r in tune] for nm in base}
    test_c = {nm: [r[nm] for r in test] for nm in base}
    if model is not None:
        tune_c["ml"] = model_probas(model, [r["x"] for r in tune])
        test_c["ml"] = model_probas(model, [r["x"] for r in test])

    def tune_grid(names):
        best_w, best_ll = None, float("inf")
        for combo in _compositions(len(names), 10):
            ws = [c / 10.0 for c in combo]
            ll = 0.0
            for i in range(len(tune)):
                p = sum(w * tune_c[nm][i][y_tune[i]] for w, nm in zip(ws, names) if w)
                ll -= math.log(max(1e-12, p))
            ll /= len(tune)
            if ll < best_ll:
                best_w, best_ll = ws, ll
        return best_w, best_ll

    def evaluate(names, ws):
        return [[sum(w * test_c[nm][i][k] for w, nm in zip(ws, names)) for k in range(3)]
                for i in range(len(test))]

    result = {
        "train_matches": len(train), "tune_matches": len(tune), "test_matches": len(test),
        "dixon_coles_out_of_sample_coverage_percent": round(len(dc_preds) / (n - t1) * 100, 1),
        "baseline_frequencies": _metrics([freq] * len(test), y_test),
        "uniform_log_loss_reference": round(math.log(3), 4),
        "components_on_test": {
            "dixon_coles": _metrics(test_c["dc"], y_test),
            "poisson_forme": _metrics(test_c["poisson"], y_test),
            "elo": _metrics(test_c["elo"], y_test),
        },
    }
    if model is not None:
        result["components_on_test"]["logistic_regression"] = _metrics(test_c["ml"], y_test)

    new_weights = {}
    final_probs = None
    for key, names in (("without_ml", COMPONENTS_NO_ML), ("with_ml", COMPONENTS_FULL)):
        if key == "with_ml" and model is None:
            continue
        ws, tune_ll = tune_grid(names)
        probs = evaluate(names, ws)
        new_weights[key] = dict(zip(names, ws))
        result[f"ensemble_{key}"] = {"weights": new_weights[key], "tune_log_loss": round(tune_ll, 4),
                                     "test": _metrics(probs, y_test)}
        final_probs = probs

    result["calibration_on_test"] = _calibration(final_probs, y_test)
    ens_ll = _log_loss(final_probs, y_test)
    base_ll = _log_loss([freq] * len(test), y_test)
    result["beats_baseline"] = ens_ll < base_ll
    if save:
        if not result["beats_baseline"]:
            result["saved"] = False
            result["message"] = "Poids non enregistrés : l'ensemble ne bat pas le baseline sur la période de test."
        else:
            try:
                merged = {k: dict(v) for k, v in load_weights().items()}
                merged.update({k: {nm: round(w, 4) for nm, w in v.items()} for k, v in new_weights.items()})
                settings_set("ensemble_weights", json.dumps(merged))
                with cache_lock:
                    weights_cache["at"] = 0.0
                invalidate_data()
                result["saved"] = True
            except Exception:
                logger.exception("Impossible d'enregistrer les poids")
                result["saved"] = False
    return result


@app.get("/api/tracking")
def tracking(days: int = 0, limit: int = 20):
    """Suivi en conditions réelles : prédictions faites AVANT le match, comparées au résultat réel.
    days > 0 limite aux matchs réglés depuis N jours. C'est la seule mesure vraiment honnête du modèle."""
    limit = max(1, min(limit, 200))
    where, params = "outcome IS NOT NULL", []
    if days > 0:
        where += " AND settled_at >= ?"
        params.append((datetime.now(UTC_TZ) - timedelta(days=days)).isoformat())
    with db() as conn:
        rows = [dict(r) for r in conn.execute(f"""
            SELECT event_id, league_code, home_team, away_team, p_home, p_draw, p_away, model,
                   components, outcome, home_score, away_score, settled_at
            FROM predictions_log WHERE {where} ORDER BY settled_at DESC
        """, params).fetchall()]
        pending = conn.execute("SELECT COUNT(*) FROM predictions_log WHERE outcome IS NULL").fetchone()[0]
        hist = conn.execute("""
            SELECT COALESCE(SUM(CASE WHEN home_score > away_score THEN 1 ELSE 0 END), 0),
                   COALESCE(SUM(CASE WHEN home_score = away_score THEN 1 ELSE 0 END), 0),
                   COALESCE(SUM(CASE WHEN home_score < away_score THEN 1 ELSE 0 END), 0), COUNT(*)
            FROM matches_history WHERE home_score IS NOT NULL AND away_score IS NOT NULL
        """).fetchone()
    if not rows:
        return {"settled": 0, "pending": pending,
                "message": "Aucune prédiction réglée pour l'instant. Les prédictions envoyées avec un event_id "
                           "sont enregistrées avant le match puis réglées quand le résultat arrive."}

    ps = [[r["p_home"] / 100, r["p_draw"] / 100, r["p_away"] / 100] for r in rows]
    ys = [r["outcome"] for r in rows]
    total = hist[3]
    freq = [float(hist[c]) / total for c in range(3)] if total else [1 / 3] * 3

    per_component: Dict[str, Tuple[list, list]] = {}
    for r in rows:
        try:
            comps = json.loads(r["components"] or "{}")
        except ValueError:
            continue
        for name, p in comps.items():
            per_component.setdefault(name, ([], []))
            per_component[name][0].append([x / 100 for x in p])
            per_component[name][1].append(r["outcome"])

    labels = ["Domicile", "Nul", "Extérieur"]
    recent = [{
        "match": f"{r['home_team']} vs {r['away_team']}",
        "probabilities": {"home_win": r["p_home"], "draw": r["p_draw"], "away_win": r["p_away"]},
        "predicted": labels[max(range(3), key=lambda c: ps[i][c])],
        "actual": labels[r["outcome"]],
        "score": f"{r['home_score']} - {r['away_score']}",
        "correct": max(range(3), key=lambda c: ps[i][c]) == r["outcome"],
    } for i, r in enumerate(rows[:limit])]

    return {
        "settled": len(rows),
        "pending": pending,
        "model": _metrics(ps, ys),
        "baseline_frequencies": _metrics([freq] * len(rows), ys),
        "components": {name: {"matches": len(y), **_metrics(p, y)} for name, (p, y) in per_component.items()},
        "calibration": _calibration(ps, ys),
        "reliability_note": ("Moins de 200 matchs réglés : les écarts observés peuvent être du hasard."
                             if len(rows) < 200 else "Échantillon suffisant pour une première lecture."),
        "recent": recent,
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
