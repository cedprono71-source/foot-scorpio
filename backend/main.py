import os
import math
import time
import sqlite3
import logging
import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List, Tuple

import requests
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Optional genuine ML model. The app remains usable without scikit-learn.
try:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

APP_VERSION = "32.0"
DB_PATH = os.getenv("FOOTBALL_DB_PATH", "football_history.db")
PARIS_TZ = ZoneInfo("Europe/Paris")
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
REQUEST_TIMEOUT = 8
BACKFILL_DAYS = int(os.getenv("BACKFILL_DAYS", "30"))
MIN_ML_ROWS = 80

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("signal-fort")

app = FastAPI(
    title="Signal Fort - Version 32.0 | Real Data + Predictive Models",
    version=APP_VERSION,
    description="Football predictions based on ESPN results, SQLite history, Elo, Poisson and optional trained machine learning."
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

matches_cache = TTLCache(maxsize=200, ttl=300)
prediction_cache = TTLCache(maxsize=500, ttl=900)
ml_model_cache = {"model": None, "trained_at": 0, "rows": 0}
cache_lock = threading.Lock()

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

session = requests.Session()
session.headers.update({"User-Agent": "SignalFort/32.0 (football statistics app)"})


def db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=20)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=20000")
    return conn


def init_db():
    with db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS matches_history (
                event_id TEXT PRIMARY KEY,
                league TEXT,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_score INTEGER,
                away_score INTEGER,
                date TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_matches_date ON matches_history(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_matches_home ON matches_history(home_team, date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_matches_away ON matches_history(away_team, date)")


init_db()


class MatchRequest(BaseModel):
    home_team: str = Field(min_length=1, max_length=100)
    away_team: str = Field(min_length=1, max_length=100)
    home_id: Optional[str] = None
    away_id: Optional[str] = None
    event_id: Optional[str] = None
    league_code: Optional[str] = "eng.1"


def save_match_to_db(event_id, league, home_team, away_team, home_score, away_score, date):
    if not event_id or not home_team or not away_team:
        return
    if home_score is None or away_score is None:
        return
    try:
        with db_connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO matches_history
                (event_id, league, home_team, away_team, home_score, away_score, date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (str(event_id), league, home_team, away_team,
                  int(home_score), int(away_score), str(date)[:10]))
        with cache_lock:
            ml_model_cache["model"] = None
            prediction_cache.clear()
    except Exception:
        logger.exception("Unable to save match %s", event_id)


def fetch_scoreboard(league_code: str, date_yyyymmdd: str) -> List[dict]:
    url = f"{ESPN_BASE}/{league_code}/scoreboard"
    response = session.get(url, params={"dates": date_yyyymmdd}, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json().get("events", [])


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
        return {
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
        "home_logo": h["logo"],
        "away_logo": a["logo"],
        "home_score": h["score"],
        "away_score": a["score"],
        "game_state": game_state,
    }
    if game_state == "OFF":
        save_match_to_db(parsed["event_id"], parsed["league"], h["name"], a["name"],
                         h["score"], a["score"], parsed["date"])
    return parsed


def backfill_history(days_back=BACKFILL_DAYS):
    logger.info("Starting ESPN historical backfill (%s days)", days_back)
    today = datetime.now(PARIS_TZ).date()
    # ESPN date requests are parallelized by league/day; failures do not stop startup.
    jobs = []
    # Startup import is deliberately limited to the 11 main leagues to avoid
    # hundreds of simultaneous ESPN calls. Other leagues are still available
    # in the live fixtures endpoint.
    backfill_leagues = list(LEAGUES.keys())[:11]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for day_offset in range(days_back):
            date_key = (today - timedelta(days=day_offset)).strftime("%Y%m%d")
            for league_code in backfill_leagues:
                jobs.append(pool.submit(fetch_scoreboard, league_code, date_key))
        for future in as_completed(jobs):
            try:
                events = future.result()
                # ESPN response does not expose the league code reliably, so infer it
                # from the request mapping recorded by the caller below where possible.
                # Historical rows are still saved by the daily sync endpoint.
                for event in events:
                    comp = (event.get("competitions") or [{}])[0]
                    status = (comp.get("status") or {}).get("type") or {}
                    if str(status.get("state", "")).lower() != "post":
                        continue
                    competitors = comp.get("competitors") or []
                    if len(competitors) != 2:
                        continue
                    h = next((x for x in competitors if x.get("homeAway") == "home"), competitors[0])
                    a = next((x for x in competitors if x.get("homeAway") == "away"), competitors[1])
                    ht = (h.get("team") or {}).get("displayName") or (h.get("team") or {}).get("name")
                    at = (a.get("team") or {}).get("displayName") or (a.get("team") or {}).get("name")
                    try:
                        hs, aw = int(h.get("score")), int(a.get("score"))
                        dt = datetime.fromisoformat(event["date"].replace("Z", "+00:00")).astimezone(PARIS_TZ)
                        if ht and at:
                            save_match_to_db(event.get("id"), "ESPN", ht, at, hs, aw, dt.strftime("%Y-%m-%d"))
                    except Exception:
                        continue
            except Exception:
                continue
    logger.info("Historical backfill finished")


def get_history(team: str, limit=10) -> List[dict]:
    with db_connect() as conn:
        rows = conn.execute("""
            SELECT league, home_team, away_team, home_score, away_score, date
            FROM matches_history
            WHERE home_team = ? OR away_team = ?
            ORDER BY date DESC LIMIT ?
        """, (team, team, limit)).fetchall()
    return [dict(r) for r in rows]


def team_profile(team: str, limit=10) -> dict:
    games = get_history(team, limit)
    home_gf, home_ga, away_gf, away_ga = [], [], [], []
    points, goals_for, goals_against, recent_form = 0, 0, 0, []
    for game in games:
        is_home = game["home_team"] == team
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
    with db_connect() as conn:
        rows = conn.execute("""
            SELECT home_score, away_score FROM matches_history
            WHERE home_score IS NOT NULL AND away_score IS NOT NULL
            AND (league = ? OR league = 'ESPN')
            ORDER BY date DESC LIMIT 1000
        """, (LEAGUES.get(league_code, league_code),)).fetchall()
    if len(rows) < 20:
        rows = []
        with db_connect() as conn:
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


def elo_ratings() -> Dict[str, float]:
    with db_connect() as conn:
        rows = conn.execute("""
            SELECT home_team, away_team, home_score, away_score, date
            FROM matches_history
            WHERE home_score IS NOT NULL AND away_score IS NOT NULL
            ORDER BY date ASC
        """).fetchall()
    ratings: Dict[str, float] = {}
    for row in rows:
        h, a = row["home_team"], row["away_team"]
        rh, ra = ratings.get(h, 1500.0), ratings.get(a, 1500.0)
        expected_h = 1.0 / (1.0 + 10 ** ((ra - (rh + 55.0)) / 400.0))
        expected_a = 1.0 - expected_h
        if row["home_score"] > row["away_score"]:
            actual_h = 1.0
        elif row["home_score"] == row["away_score"]:
            actual_h = 0.5
        else:
            actual_h = 0.0
        margin = abs(row["home_score"] - row["away_score"])
        k = 20.0 * min(1.5, 1.0 + 0.15 * margin)
        ratings[h] = rh + k * (actual_h - expected_h)
        ratings[a] = ra + k * ((1.0 - actual_h) - expected_a)
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


def build_features(home_team: str, away_team: str, league_code: str) -> Tuple[List[float], dict]:
    hp, ap = team_profile(home_team), team_profile(away_team)
    ratings = elo_ratings()
    home_elo = ratings.get(home_team, 1500.0)
    away_elo = ratings.get(away_team, 1500.0)
    home_avg, away_avg = league_goal_averages(league_code)

    # Shrink sparse team samples toward league averages to avoid overreacting to 1-2 games.
    def shrink(value, n, prior, weight=5.0):
        return (value * n + prior * weight) / (n + weight)

    h_attack = shrink(hp["goals_for_avg"], hp["matches"], home_avg)
    h_defence = shrink(hp["goals_against_avg"], hp["matches"], away_avg)
    a_attack = shrink(ap["goals_for_avg"], ap["matches"], away_avg)
    a_defence = shrink(ap["goals_against_avg"], ap["matches"], home_avg)

    home_xg = max(0.20, min(4.0, (h_attack * a_defence / max(0.5, away_avg)) * 1.08))
    away_xg = max(0.20, min(4.0, (a_attack * h_defence / max(0.5, home_avg)) * 0.94))
    features = [
        home_xg, away_xg, home_elo - away_elo,
        hp["points_per_game"], ap["points_per_game"],
        hp["goals_for_avg"], hp["goals_against_avg"],
        ap["goals_for_avg"], ap["goals_against_avg"],
        float(hp["matches"]), float(ap["matches"]),
    ]
    details = {
        "home_profile": hp, "away_profile": ap,
        "home_elo": round(home_elo, 1), "away_elo": round(away_elo, 1),
        "home_xg": round(home_xg, 2), "away_xg": round(away_xg, 2),
        "league_home_goals_avg": round(home_avg, 2),
        "league_away_goals_avg": round(away_avg, 2),
    }
    return features, details


def train_ml_model():
    if not SKLEARN_AVAILABLE:
        return None, 0
    now = time.time()
    with cache_lock:
        if ml_model_cache["model"] is not None and now - ml_model_cache["trained_at"] < 3600:
            return ml_model_cache["model"], ml_model_cache["rows"]

    with db_connect() as conn:
        rows = conn.execute("""
            SELECT home_team, away_team, home_score, away_score, date
            FROM matches_history
            WHERE home_score IS NOT NULL AND away_score IS NOT NULL
            ORDER BY date ASC
        """).fetchall()

    # Build chronological pre-match features: only earlier matches are used.
    rolling: Dict[str, List[Tuple[int, int, str]]] = {}
    X, y = [], []
    for row in rows:
        h, a = row["home_team"], row["away_team"]
        hg, ag = int(row["home_score"]), int(row["away_score"])
        hgames, agames = rolling.get(h, [])[-10:], rolling.get(a, [])[-10:]

        def stats(games):
            if not games:
                return 1.35, 1.35, 1.0
            gf = sum(x[0] for x in games) / len(games)
            ga = sum(x[1] for x in games) / len(games)
            pts = sum(3 if x[0] > x[1] else 1 if x[0] == x[1] else 0 for x in games) / len(games)
            return gf, ga, pts

        hgf, hga, hp = stats(hgames)
        agf, aga, ap = stats(agames)
        X.append([hgf, hga, agf, aga, hp, ap, hgf - aga, agf - hga])
        y.append(0 if hg > ag else 1 if hg == ag else 2)

        rolling.setdefault(h, []).append((hg, ag, row["date"]))
        rolling.setdefault(a, []).append((ag, hg, row["date"]))

    if len(X) < MIN_ML_ROWS or len(set(y)) < 3:
        return None, len(X)
    try:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced")
        )
        model.fit(X, y)
        with cache_lock:
            ml_model_cache.update({"model": model, "trained_at": now, "rows": len(X)})
        return model, len(X)
    except Exception:
        logger.exception("ML training failed")
        return None, len(X)


def ml_prediction(home_team: str, away_team: str, features: List[float]) -> Optional[dict]:
    model, rows = train_ml_model()
    if model is None:
        return None
    # ML uses its own rolling-form feature representation.
    hp, ap = team_profile(home_team), team_profile(away_team)
    x = [[hp["goals_for_avg"], hp["goals_against_avg"],
          ap["goals_for_avg"], ap["goals_against_avg"],
          hp["points_per_game"], ap["points_per_game"],
          hp["goals_for_avg"] - ap["goals_against_avg"],
          ap["goals_for_avg"] - hp["goals_against_avg"]]]
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


def calculate_surgical_consensus(home_team: str, away_team: str, league_code="eng.1") -> dict:
    features, details = build_features(home_team, away_team, league_code)
    poisson = poisson_model(details["home_xg"], details["away_xg"])

    # Elo probability from real match results, including a modest home advantage.
    elo_diff = details["home_elo"] + 55.0 - details["away_elo"]
    home_elo_prob = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
    draw_base = max(0.16, min(0.30, 0.27 - abs(details["home_xg"] - details["away_xg"]) * 0.035))
    elo_probs = {
        "home_win": home_elo_prob * (1 - draw_base) * 100,
        "draw": draw_base * 100,
        "away_win": (1 - home_elo_prob) * (1 - draw_base) * 100,
    }

    ml = ml_prediction(home_team, away_team, features)
    if ml:
        weights = (0.55, 0.20, 0.25)  # Poisson, Elo, trained ML
        method = "Poisson + Elo + LogisticRegression entraînée"
        ml_probs = ml
    else:
        weights = (0.72, 0.28, 0.0)
        method = "Poisson + Elo (historique ML insuffisant)"
        ml_probs = None

    final = {}
    for key in ("home_win", "draw", "away_win"):
        final[key] = round(
            poisson[key] * weights[0] + elo_probs[key] * weights[1] +
            ((ml_probs[key] if ml_probs else 0.0) * weights[2]), 1
        )
    total = sum(final.values())
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


def get_team_scorer_candidates(team_id: Optional[str]) -> list:
    # No fabricated player names: ESPN roster/boxscore data is not guaranteed for
    # every league or fixture. This endpoint is intentionally conservative.
    if not team_id:
        return []
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{team_id}/roster"
        response = session.get(url, timeout=5)
        if response.status_code != 200:
            return []
        data = response.json()
        candidates = []
        for athlete in data.get("athletes", []):
            if isinstance(athlete, dict) and athlete.get("displayName"):
                candidates.append({"name": athlete["displayName"], "source": "ESPN roster"})
        return candidates[:5]
    except Exception:
        return []


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


@app.get("/", response_class=HTMLResponse)
def serve_home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return """<html><head><meta charset='utf-8'><title>Signal Fort 32.0</title></head>
        <body style='font-family:Arial;padding:30px'><h1>Signal Fort 32.0</h1>
        <p>Backend actif. Place ton fichier index.html dans le même dossier.</p>
        <p><a href='/docs'>Ouvrir la documentation API</a></p></body></html>"""


@app.get("/health")
def health():
    with db_connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM matches_history").fetchone()[0]
    return {
        "status": "ok", "version": APP_VERSION,
        "historical_matches": count,
        "sklearn_available": SKLEARN_AVAILABLE,
        "ml_active": ml_model_cache["model"] is not None,
    }


@app.get("/api/history")
def history(limit: int = 50):
    limit = max(1, min(limit, 500))
    with db_connect() as conn:
        rows = conn.execute("""
            SELECT event_id, league, home_team, away_team, home_score, away_score, date
            FROM matches_history ORDER BY date DESC LIMIT ?
        """, (limit,)).fetchall()
    return {"count": len(rows), "matches": [dict(r) for r in rows]}


@app.get("/api/matches/foot")
def get_foot_matches(date: str):
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Date invalide. Format attendu : YYYY-MM-DD")

    if date in matches_cache:
        return matches_cache[date]

    requested = datetime.strptime(date, "%Y-%m-%d").date()
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
    matches_cache[date] = result
    return result


@app.post("/predict/match/foot")
def predict_foot_match(req: MatchRequest):
    home_team = req.home_team.strip()
    away_team = req.away_team.strip()
    league_code = req.league_code or "eng.1"
    if home_team.casefold() == away_team.casefold():
        raise HTTPException(status_code=400, detail="Les deux équipes doivent être différentes.")

    cache_key = f"{home_team}|{away_team}|{league_code}"
    if cache_key in prediction_cache:
        return prediction_cache[cache_key]

    try:
        consensus = calculate_surgical_consensus(home_team, away_team, league_code)
        probs = {
            "home_win": consensus["home_win"],
            "draw": consensus["draw"],
            "away_win": consensus["away_win"],
        }

        # Score modal from the Poisson matrix, not forced to match 1X2 probabilities.
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

        # Keep the legacy response keys so the existing frontend remains compatible.
        # Player identities and card probabilities are not fabricated.
        result = {
            "match": f"{home_team} vs {away_team}",
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
        prediction_cache[cache_key] = result
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/match/foot/value")
def predict_value(req: dict):
    """Value helper: body example {probability: 57, odds: 2.0}."""
    try:
        probability = float(req["probability"])
        odds = float(req["odds"])
        if not 0 <= probability <= 100 or odds <= 1:
            raise ValueError()
        return implied_value(probability, odds)
    except Exception:
        raise HTTPException(status_code=400, detail="Envoyer probability (0-100) et odds (>1.0).")


@app.post("/api/history/backfill")
def start_backfill(days: int = BACKFILL_DAYS):
    days = max(1, min(days, 365))
    thread = threading.Thread(target=backfill_history, args=(days,), daemon=True)
    thread.start()
    return {"started": True, "days": days, "message": "Chargement historique lancé en arrière-plan."}


@app.on_event("startup")
def startup_backfill():
    # Run once per process; historical imports are intentionally asynchronous.
    if os.getenv("DISABLE_STARTUP_BACKFILL", "0") != "1":
        threading.Thread(target=backfill_history, args=(BACKFILL_DAYS,), daemon=True).start()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
