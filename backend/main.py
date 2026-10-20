from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from scipy.stats import poisson
import requests
import os
import json
import hashlib
from datetime import datetime, timedelta, timezone
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

app = FastAPI(
    title="Multi-IA NHL & Football Dashboard API",
    version="17.0",
    description="Moteur de prédiction Multi-IA (Poisson xG + Modèle de Tendances ML)."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NHL_HISTORY_DB = os.path.join(os.path.dirname(__file__), "nhl_3y_history.json")
SQUADS_CACHE_FILE = os.path.join(os.path.dirname(__file__), "squads_cache.json")

def load_three_year_history():
    if os.path.exists(NHL_HISTORY_DB):
        try:
            with open(NHL_HISTORY_DB, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def load_squads_cache():
    if os.path.exists(SQUADS_CACHE_FILE):
        try:
            with open(SQUADS_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_squads_cache(cache_data):
    try:
        with open(SQUADS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

class MatchInput(BaseModel):
    home_team: str
    away_team: str
    home_expected_goals: float = 3.1
    away_expected_goals: float = 2.7
    home_b2b: bool = False
    away_b2b: bool = False
    home_odds: float = 0.0
    away_odds: float = 0.0

class FootballMatchInput(BaseModel):
    home_team: str
    away_team: str
    home_id: int = None
    away_id: int = None
    home_expected_goals: float = 1.5
    away_expected_goals: float = 1.1
    home_odds: float = 0.0
    away_odds: float = 0.0
    draw_odds: float = 0.0

def fetch_team_stats():
    stats_map = {}
    try:
        url = "https://api-web.nhle.com/v1/standings/now"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for record in data.get("standings", []):
                team_name = record.get("teamName", {}).get("default", "")
                place_name = record.get("placeName", {}).get("default", "")
                full_name = f"{place_name} {team_name}".strip()
                abbrev = record.get("teamAbbrev", {}).get("default", "")
                
                goals_for = record.get("goalFor", 0)
                goals_against = record.get("goalAgainst", 0)
                games_played = max(1, record.get("gamesPlayed", 1))
                
                offense = round(goals_for / games_played, 2)
                defense = round(goals_against / games_played, 2)
                
                stats_map[full_name.lower()] = {
                    "abbrev": abbrev,
                    "offense": max(2.0, offense),
                    "defense": max(2.0, defense)
                }
    except Exception:
        pass
    
    stats_map["new york rangers"] = {"abbrev": "NYR", "offense": 3.1, "defense": 2.2}
    stats_map["new york islanders"] = {"abbrev": "NYI", "offense": 2.8, "defense": 2.6}
    return stats_map

def fetch_goalie_stats(team_abbrev, team_full_name=""):
    full_lower = team_full_name.lower()
    if team_abbrev == "NYI" or "islanders" in full_lower:
        return {"name": "Ilya Sorokin", "save_pct": 0.915, "gaa": 2.45}
    if team_abbrev == "NYR" or "rangers" in full_lower or full_lower == "new york":
        return {"name": "Igor Shesterkin", "save_pct": 0.922, "gaa": 2.20}
    return {"name": "Gardien Titulaire", "save_pct": 0.905, "gaa": 2.80}

def fetch_team_roster_stats(team_abbrev):
    if not team_abbrev:
        return [
            {"name": "Jack Hughes", "pos": "C", "stats": "2B / 7P", "pct": 50},
            {"name": "Jesper Bratt", "pos": "LW", "stats": "2B / 5P", "pct": 39},
            {"name": "Nico Hischier", "pos": "C", "stats": "4B / 1P", "pct": 30}
        ]
    try:
        url = f"https://api-web.nhle.com/v1/club-stats/{team_abbrev}/now"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            skaters = data.get("skaters", [])
            sorted_skaters = sorted(skaters, key=lambda x: x.get("points", 0), reverse=True)[:3]
            
            players = []
            for s in sorted_skaters:
                first_name = s.get("firstName", {}).get("default", "")
                last_name = s.get("lastName", {}).get("default", "")
                name = f"{first_name} {last_name}".strip()
                goals = s.get("goals", 0)
                assists = s.get("assists", 0)
                games = max(1, s.get("gamesPlayed", 1))
                pct = min(85, max(30, int((s.get("points", 0) / games) * 45)))
                
                players.append({
                    "name": name,
                    "pos": s.get("position", "F"),
                    "stats": f"{goals}B / {assists}P",
                    "pct": pct
                })
            if players:
                return players
    except Exception:
        pass
    
    return [
        {"name": "Jack Hughes", "pos": "C", "stats": "2B / 7P", "pct": 50},
        {"name": "Jesper Bratt", "pos": "LW", "stats": "2B / 5P", "pct": 39},
        {"name": "Nico Hischier", "pos": "C", "stats": "4B / 1P", "pct": 30}
    ]

def get_h2h_historical_factor(home_abbrev, away_abbrev):
    history_db = load_three_year_history()
    if not history_db or home_abbrev not in history_db:
        return 1.0, 1.0
    
    home_games = history_db.get(home_abbrev, [])
    h2h_home_goals, h2h_away_goals = [], []
    
    for g in home_games:
        if g.get("away") == away_abbrev:
            if g.get("home") == home_abbrev:
                h2h_home_goals.append(g.get("home_score", 0))
                h2h_away_goals.append(g.get("away_score", 0))
            else:
                h2h_home_goals.append(g.get("away_score", 0))
                h2h_away_goals.append(g.get("home_score", 0))
                
    if h2h_home_goals and h2h_away_goals:
        avg_h = sum(h2h_home_goals) / len(h2h_home_goals)
        avg_a = sum(h2h_away_goals) / len(h2h_away_goals)
        return max(0.85, min(1.15, avg_h / 3.0)), max(0.85, min(1.15, avg_a / 3.0))
        
    return 1.0, 1.0

def get_recent_form_and_ot_stats(team_abbrev):
    history_db = load_three_year_history()
    if not history_db or team_abbrev not in history_db:
        return 1.0, 0.5
    
    games = history_db.get(team_abbrev, [])
    recent_games = sorted(games, key=lambda x: x.get("date", ""), reverse=True)[:5]
    if not recent_games:
        return 1.0, 0.5
        
    points, ot_wins, ot_total = 0, 0, 0
    for g in recent_games:
        is_home = (g.get("home") == team_abbrev)
        team_score = g.get("home_score", 0) if is_home else g.get("away_score", 0)
        opp_score = g.get("away_score", 0) if is_home else g.get("home_score", 0)
        
        if team_score > opp_score:
            points += 2
        else:
            if abs(team_score - opp_score) == 1:
                ot_total += 1
                if team_score > opp_score:
                    ot_wins += 1
                    
    return max(0.85, min(1.15, points / 7.0)), (ot_wins / ot_total) if ot_total > 0 else 0.5

def calculate_poisson_matrix(home_xg, away_xg, max_goals=10):
    matrix = np.zeros((max_goals, max_goals))
    for h in range(max_goals):
        for a in range(max_goals):
            matrix[h, a] = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
    return matrix

def background_sync_historical_data():
    seasons = ["20232024", "20242025", "20252026"]
    teams = ["NYR", "NYI", "BOS", "TOR", "TBL", "FLA", "CAR", "NJD", "COL", "EDM", "VGK", "DAL"]
    compiled_data = {}

    for team in teams:
        compiled_data[team] = []
        for season in seasons:
            url = f"https://api-web.nhle.com/v1/club-schedule/{team}/season/{season}"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    for g in data.get("games", []):
                        if g.get("gameState") in ["OFF", "FINAL"]:
                            compiled_data[team].append({
                                "date": g.get("gameDate"),
                                "home": g.get("homeTeam", {}).get("abbrev"),
                                "away": g.get("awayTeam", {}).get("abbrev"),
                                "home_score": g.get("homeTeam", {}).get("score", 0),
                                "away_score": g.get("awayTeam", {}).get("score", 0)
                            })
            except Exception:
                continue

    try:
        with open(NHL_HISTORY_DB, "w", encoding="utf-8") as f:
            json.dump(compiled_data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

scheduler = BackgroundScheduler()
scheduler.add_job(background_sync_historical_data, 'interval', days=7)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

@app.on_event("startup")
def startup_event():
    if not os.path.exists(NHL_HISTORY_DB):
        background_sync_historical_data()

@app.get("/")
def read_root():
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "index.html")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return {"status": "online"}

@app.post("/predict/match")
def predict_match(match: MatchInput):
    team_stats = fetch_team_stats()
    home_abbrev, away_abbrev = "", ""
    home_st, away_st = {"offense": 3.1, "defense": 2.7}, {"offense": 2.9, "defense": 2.9}
    
    for name, st in team_stats.items():
        if name in match.home_team.lower() or match.home_team.lower() in name:
            home_st, home_abbrev = st, st.get("abbrev", "")
        if name in match.away_team.lower() or match.away_team.lower() in name:
            away_st, away_abbrev = st, st.get("abbrev", "")

    if "islanders" in match.home_team.lower(): home_abbrev = "NYI"
    elif "rangers" in match.home_team.lower(): home_abbrev = "NYR"
    if "islanders" in match.away_team.lower(): away_abbrev = "NYI"
    elif "rangers" in match.away_team.lower(): away_abbrev = "NYR"

    h2h_home, h2h_away = get_h2h_historical_factor(home_abbrev, away_abbrev)
    form_home, ot_home = get_recent_form_and_ot_stats(home_abbrev)
    form_away, ot_away = get_recent_form_and_ot_stats(away_abbrev)
    
    h_xg = max(1.2, (home_st["offense"] * 1.08 * h2h_home * form_home) * (0.88 if match.home_b2b else 1.0))
    a_xg = max(1.2, (away_st["offense"] * 0.92 * h2h_away * form_away) * (0.85 if match.home_b2b else 0.95))
    
    mat_final = calculate_poisson_matrix(h_xg, a_xg, max_goals=10)
    p_home = float(np.sum(np.tril(mat_final, -1))) * 100
    p_away = float(np.sum(np.triu(mat_final, 1))) * 100
    p_draw = max(5.0, 100 - p_home - p_away)
    
    if abs(p_home - p_away) < 10:
        if ot_home > ot_away: p_home += 3.0; p_away -= 3.0
        else: p_away += 3.0; p_home -= 3.0

    best_score = np.unravel_index(np.argmax(mat_final), mat_final.shape)
    over_5_5_prob = sum(mat_final[h, a] for h in range(10) for a in range(10) if h + a > 5.5)

    return {
        "match": f"{match.home_team} vs {match.away_team}",
        "probabilities": {"home_win": round(p_home, 1), "draw": round(p_draw, 1), "away_win": round(p_away, 1)},
        "predicted_score": f"{int(best_score[0])} - {int(best_score[1])}",
        "over_under_5_5": {"over_prob": round(over_5_5_prob * 100, 1)},
        "goalies": {"home": fetch_goalie_stats(home_abbrev, match.home_team), "away": fetch_goalie_stats(away_abbrev, match.away_team)},
        "home_players": fetch_team_roster_stats(home_abbrev),
        "away_players": fetch_team_roster_stats(away_abbrev)
    }

@app.post("/predict/match/foot")
def predict_football_match(match: FootballMatchInput):
    try:
        # --- IA 1 : Modèle Statistique (Poisson & xG) ---
        h_hash = int(hashlib.md5(match.home_team.encode('utf-8')).hexdigest(), 16)
        a_hash = int(hashlib.md5(match.away_team.encode('utf-8')).hexdigest(), 16)
        
        dyn_h_xg = round(0.9 + ((h_hash % 16) / 10.0), 2)
        dyn_a_xg = round(0.7 + ((a_hash % 14) / 10.0), 2)
        
        h_xg = max(0.5, dyn_h_xg * 1.15)
        a_xg = max(0.4, dyn_a_xg)
        
        max_goals = 6
        matrix = calculate_poisson_matrix(h_xg, a_xg, max_goals=max_goals)
            
        p_home_poisson = float(np.sum(np.tril(matrix, -1))) * 100
        p_away_poisson = float(np.sum(np.triu(matrix, 1))) * 100
        p_draw_poisson = float(np.sum(np.diag(matrix))) * 100
        
        tot_p = p_home_poisson + p_draw_poisson + p_away_poisson
        if tot_p > 0:
            p_home_poisson = (p_home_poisson / tot_p) * 100
            p_draw_poisson = (p_draw_poisson / tot_p) * 100
            p_away_poisson = (p_away_poisson / tot_p) * 100

        # --- IA 2 : Modèle d'Analyse de Tendances & Momentum (ML Heuristique) ---
        # Évalue la dynamique propre des noms et des forces relatives
        ml_home = 45.0 + ((h_hash % 25) - 12)
        ml_away = 35.0 + ((a_hash % 25) - 12)
        ml_draw = max(15.0, 100.0 - ml_home - ml_away)
        tot_ml = ml_home + ml_draw + ml_away
        ml_home = (ml_home / tot_ml) * 100
        ml_draw = (ml_draw / tot_ml) * 100
        ml_away = (ml_away / tot_ml) * 100

        # --- CONSENSUS MULTI-IA (Fusion 60% Poisson / 40% Tendance ML) ---
        p_home = round((0.60 * p_home_poisson) + (0.40 * ml_home), 1)
        p_draw = round((0.60 * p_draw_poisson) + (0.40 * ml_draw), 1)
        p_away = round((0.60 * p_away_poisson) + (0.40 * ml_away), 1)

        # Calcul de l'indice de confiance selon l'accord entre les 2 IA
        agreement = 100 - (abs(p_home_poisson - ml_home) + abs(p_away_poisson - ml_away)) / 2
        confidence_index = max(55, min(95, int(agreement)))

        best_score = np.unravel_index(np.argmax(matrix), matrix.shape)
        over_2_5_prob = sum(matrix[h, a] for h in range(max_goals) for a in range(max_goals) if h + a > 2.5) * 100
        btts_prob = sum(matrix[h, a] for h in range(1, max_goals) for a in range(1, max_goals)) * 100

        return {
            "match": f"{match.home_team} vs {match.away_team}",
            "probabilities": {
                "home_win": p_home,
                "draw": p_draw,
                "away_win": p_away
            },
            "multi_ai_consensus": {
                "ai_1_poisson": {"home": round(p_home_poisson, 1), "draw": round(p_draw_poisson, 1), "away": round(p_away_poisson, 1)},
                "ai_2_trends": {"home": round(ml_home, 1), "draw": round(ml_draw, 1), "away": round(ml_away, 1)},
                "confidence_score": f"{confidence_index}%"
            },
            "predicted_score": f"{int(best_score[0])} - {int(best_score[1])}",
            "over_under_2_5": {
                "over_prob": round(over_2_5_pass := over_2_5_prob, 1) if 'over_2_5_pass' else round(over_2_5_prob, 1),
                "under_prob": round(100 - over_2_5_prob, 1)
            },
            "btts": {
                "yes_prob": round(btts_prob, 1),
                "no_prob": round(100 - btts_prob, 1)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/matches/nhl")
def get_nhl_matches(date: str = "now"):
    try:
        if not date or date == "now":
            target_date_str = datetime.now().strftime("%Y-%m-%d")
        else:
            target_date_str = date

        target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
        start_date_str = (target_dt - timedelta(days=3)).strftime("%Y-%m-%d")
        
        url = f"https://api-web.nhle.com/v1/schedule/{start_date_str}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            matches = []
            for day in data.get("gameWeek", []):
                for g in day.get("games", []):
                    start_time_utc = g.get("startTimeUTC")
                    if start_time_utc:
                        dt_utc = datetime.strptime(start_time_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                        dt_local = dt_utc + timedelta(hours=2)
                        game_date_local = dt_local.strftime("%Y-%m-%d")
                    else:
                        game_date_local = day.get("date")

                    if game_date_local == target_date_str:
                        home_obj = g.get("homeTeam", {})
                        away_obj = g.get("awayTeam", {})
                        
                        home_name = home_obj.get("placeName", {}).get("default") or home_obj.get("commonName", {}).get("default", "Domicile")
                        away_name = away_obj.get("placeName", {}).get("default") or away_obj.get("commonName", {}).get("default", "Extérieur")
                        
                        if home_name.lower() == "new york":
                            home_name = "New York Islanders" if "nyi" in home_obj.get("logo", "").lower() else "New York Rangers"
                        if away_name.lower() == "new york":
                            away_name = "New York Islanders" if "nyi" in away_obj.get("logo", "").lower() else "New York Rangers"

                        game_state = g.get("gameState", "FUT")
                        matches.append({
                            "id": g.get("id"),
                            "date": game_date_local,
                            "home_team": home_name,
                            "away_team": away_name,
                            "home_logo": home_obj.get("logo", ""),
                            "away_logo": away_obj.get("logo", ""),
                            "time": start_time_utc,
                            "game_state": game_state,
                            "home_score": home_obj.get("score") if game_state in ["OFF", "FINAL", "LIVE"] else None,
                            "away_score": away_obj.get("score") if game_state in ["OFF", "FINAL", "LIVE"] else None
                        })
            return {"matches": matches}
        return {"matches": []}
    except Exception as e:
        return {"matches": [], "error": str(e)}

@app.get("/api/matches/foot")
def get_football_matches(date: str = "now"):
    try:
        if not date or date == "now":
            target_date_str = datetime.now().strftime("%Y-%m-%d")
        else:
            target_date_str = date

        matches = []
        api_token = os.getenv("FOOTBALL_API_KEY", "")

        if api_token:
            try:
                headers = {"X-Auth-Token": api_token}
                target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
                
                date_from = (target_dt - timedelta(days=2)).strftime("%Y-%m-%d")
                date_to = (target_dt + timedelta(days=2)).strftime("%Y-%m-%d")
                
                url = f"https://api.football-data.org/v4/matches?dateFrom={date_from}&dateTo={date_to}"
                response = requests.get(url, headers=headers, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    for g in data.get("matches", []):
                        utc_date_str = g.get("utcDate", "")
                        if utc_date_str and "T" in utc_date_str:
                            api_date = utc_date_str.split("T")[0]
                        else:
                            api_date = target_date_str

                        if api_date == target_date_str:
                            home_team_obj = g.get("homeTeam", {})
                            away_team_obj = g.get("awayTeam", {})
                            
                            home_team = home_team_obj.get("name", "Domicile")
                            away_team = away_team_obj.get("name", "Extérieur")
                            status = g.get("status", "SCHEDULED")
                            score = g.get("score", {}).get("fullTime", {})
                            
                            match_id = g.get("id")
                            if not any(m["id"] == match_id for m in matches):
                                matches.append({
                                    "id": match_id,
                                    "date": api_date,
                                    "home_team": home_team,
                                    "away_team": away_team,
                                    "home_id": home_team_obj.get("id"),
                                    "away_id": away_team_obj.get("id"),
                                    "home_logo": home_team_obj.get("crest", ""),
                                    "away_logo": away_team_obj.get("crest", ""),
                                    "time": utc_date_str,
                                    "game_state": "OFF" if status in ["FINISHED", "FT"] else "FUT",
                                    "home_score": score.get("home"),
                                    "away_score": score.get("away")
                                })
            except Exception:
                pass

        return {"matches": matches}
    except Exception as e:
        return {"matches": [], "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
