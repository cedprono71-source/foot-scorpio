from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from scipy.stats import poisson
import requests
import os
import json
from datetime import datetime, timedelta

app = FastAPI(
    title="Multi-IA NHL & Football Dashboard API",
    version="4.0",
    description="API avec calcul prédictif basé sur l'historique des 3 dernières années et l'interrogation dynamique."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.json")
NHL_HISTORY_DB = os.path.join(os.path.dirname(__file__), "nhl_3y_history.json")

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def load_three_year_history():
    if os.path.exists(NHL_HISTORY_DB):
        try:
            with open(NHL_HISTORY_DB, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

class MatchInput(BaseModel):
    home_team: str
    away_team: str
    home_expected_goals: float = 3.1
    away_expected_goals: float = 2.7
    home_b2b: bool = False
    away_b2b: bool = False

class BetRecord(BaseModel):
    match: str
    prediction: str
    odds: float = 0.0
    status: str = "En attente"  # En attente, Gagné, Perdu

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
                
                pp_pct = round(min(0.30, max(0.12, (offense / 3.0) * 0.20)), 3)
                pk_pct = round(min(0.88, max(0.70, 0.80 + (2.8 - defense) * 0.05)), 3)
                
                stats_map[full_name.lower()] = {
                    "abbrev": abbrev,
                    "offense": max(2.0, offense),
                    "defense": max(2.0, defense),
                    "pp_pct": pp_pct,
                    "pk_pct": pk_pct
                }
    except Exception:
        pass
    
    stats_map["new york rangers"] = {"abbrev": "NYR", "offense": 3.1, "defense": 2.2, "pp_pct": 0.20, "pk_pct": 0.935}
    stats_map["new york islanders"] = {"abbrev": "NYI", "offense": 2.8, "defense": 2.6, "pp_pct": 0.19, "pk_pct": 0.810}
    return stats_map

def fetch_goalie_stats(team_abbrev, team_full_name=""):
    full_lower = team_full_name.lower()
    if team_abbrev == "NYI" or "islanders" in full_lower:
        return {"name": "Ilya Sorokin", "save_pct": 0.915, "gaa": 2.45}
    if team_abbrev == "NYR" or "rangers" in full_lower or full_lower == "new york":
        return {"name": "Igor Shesterkin", "save_pct": 0.922, "gaa": 2.20}
        
    if not team_abbrev:
        return {"name": "Gardien Titulaire", "save_pct": 0.905, "gaa": 2.80}
    try:
        url = f"https://api-web.nhle.com/v1/club-stats/{team_abbrev}/now"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            goalies = data.get("goalies", [])
            if goalies:
                top_goalie = max(goalies, key=lambda x: x.get("gamesPlayed", 0))
                name = f"{top_goalie.get('firstName', {}).get('default', '')} {top_goalie.get('lastName', {}).get('default', '')}"
                sv_pct = top_goalie.get("savePercentage", 0.905)
                gaa = top_goalie.get("goalsAgainstAverage", 2.80)
                return {
                    "name": name,
                    "save_pct": round(sv_pct, 3) if sv_pct else 0.905,
                    "gaa": round(gaa, 2) if gaa else 2.80
                }
    except Exception:
        pass
    return {"name": "Gardien Titulaire", "save_pct": 0.905, "gaa": 2.80}

def fetch_all_players_projections(team_abbrev, expected_team_goals, team_full_name=""):
    players = []
    top_scorer, top_passer = None, None
    if not team_abbrev:
        return {"all": players, "top_scorer": top_scorer, "top_passer": top_passer}
        
    try:
        url = f"https://api-web.nhle.com/v1/club-stats/{team_abbrev}/now"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            skaters = data.get("skaters", [])
            for p in skaters:
                name = f"{p.get('firstName', {}).get('default', '')} {p.get('lastName', {}).get('default', '')}"
                gp = max(1, p.get("gamesPlayed", 1))
                goals = p.get("goals", 0)
                assists = p.get("assists", 0)
                points = goals + assists
                
                g_pg = goals / gp
                a_pg = assists / gp
                pts_pg = points / gp
                
                goal_prob = min(90, round((1 - np.exp(-g_pg * (expected_team_goals / 3.0))) * 100, 1))
                assist_prob = min(90, round((1 - np.exp(-a_pg * (expected_team_goals / 3.0))) * 100, 1))
                point_prob = min(95, round((1 - np.exp(-pts_pg * (expected_team_goals / 3.0))) * 100, 1))
                
                position = p.get("positionCode", "F")
                players.append({
                    "name": name, "pos": position, "goals": goals, "assists": assists,
                    "points": points, "goal_prob": goal_prob, "assist_prob": assist_prob, "point_prob": point_prob
                })
            players = sorted(players, key=lambda x: x["point_prob"], reverse=True)
            if players:
                top_scorer = max(players, key=lambda x: x["goal_prob"])
                top_passer = max(players, key=lambda x: x["assist_prob"])
    except Exception:
        pass
    return {"all": players, "top_scorer": top_scorer, "top_passer": top_passer}

def get_h2h_historical_factor(home_abbrev, away_abbrev):
    """
    Analyse l'historique des 3 dernières années pour calculer un facteur correctif 
    basé sur les confrontations directes et la moyenne de buts historiques.
    """
    history_db = load_three_year_history()
    if not history_db or home_abbrev not in history_db:
        return 1.0, 1.0
    
    home_games = history_db.get(home_abbrev, [])
    h2h_home_goals = []
    h2h_away_goals = []
    
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
        # Ajustement modéré par rapport à une moyenne standard de 3.0 buts
        factor_home = max(0.85, min(1.15, avg_h / 3.0))
        factor_away = max(0.85, min(1.15, avg_a / 3.0))
        return factor_home, factor_away
        
    return 1.0, 1.0

def calculate_poisson_matrix(home_xg, away_xg):
    max_goals = 10
    matrix = np.zeros((max_goals, max_goals))
    for h in range(max_goals):
        for a in range(max_goals):
            matrix[h, a] = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
    return matrix

@app.get("/")
def read_root():
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "index.html")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return {"status": "online", "message": "API opérationnelle."}

@app.post("/predict/match")
def predict_match(match: MatchInput):
    team_stats = fetch_team_stats()
    
    home_abbrev, away_abbrev = "", ""
    home_st = {"offense": 3.1, "defense": 2.7, "pp_pct": 0.20, "pk_pct": 0.80}
    away_st = {"offense": 2.9, "defense": 2.9, "pp_pct": 0.18, "pk_pct": 0.78}
    
    for name, st in team_stats.items():
        if name in match.home_team.lower() or match.home_team.lower() in name:
            home_st = st
            home_abbrev = st.get("abbrev", "")
        if name in match.away_team.lower() or match.away_team.lower() in name:
            away_st = st
            away_abbrev = st.get("abbrev", "")

    if "islanders" in match.home_team.lower():
        home_abbrev = "NYI"
    elif "rangers" in match.home_team.lower() or match.home_team.lower() == "new york":
        home_abbrev = "NYR"

    if "islanders" in match.away_team.lower():
        away_abbrev = "NYI"
    elif "rangers" in match.away_team.lower() or match.away_team.lower() == "new york":
        away_abbrev = "NYR"

    home_goalie = fetch_goalie_stats(home_abbrev, match.home_team)
    away_goalie = fetch_goalie_stats(away_abbrev, match.away_team)
    
    # Récupération du facteur d'historique Head-to-Head sur 3 ans
    h2h_factor_home, h2h_factor_away = get_h2h_historical_factor(home_abbrev, away_abbrev)
    
    h_xg2 = (home_st["offense"] * 1.08) * h2h_factor_home
    a_xg2 = (away_st["offense"] * 0.92) * h2h_factor_away
    
    h_xg3 = max(1.5, (home_st["offense"] - (away_st["defense"] * 0.1) + 0.2) * h2h_factor_home)
    a_xg3 = max(1.5, (away_st["offense"] - (home_st["defense"] * 0.1)) * h2h_factor_away)
    
    h_xg4 = max(1.2, home_st["offense"] * (1 - (away_goalie["save_pct"] - 0.900) * 2))
    a_xg4 = max(1.2, away_st["offense"] * (1 - (home_goalie["save_pct"] - 0.900) * 2))
    
    pp_advantage_home = home_st["pp_pct"] * (1 - away_st["pk_pct"]) * 10.0
    pp_advantage_away = away_st["pp_pct"] * (1 - home_st["pk_pct"]) * 10.0
    h_xg5 = max(1.2, h_xg2 + pp_advantage_home)
    a_xg5 = max(1.2, a_xg2 + pp_advantage_away)
    
    home_fatigue = 0.88 if match.home_b2b else 1.0
    away_fatigue = 0.88 if match.away_b2b else 0.95
    h_xg6 = max(1.2, h_xg2 * home_fatigue)
    a_xg6 = max(1.2, a_xg2 * away_fatigue)
    
    mat1 = calculate_poisson_matrix(3.1, 2.7)
    p1_home = float(np.sum(np.tril(mat1, -1))) * 100
    mat2 = calculate_poisson_matrix(h_xg2, a_xg2)
    p2_home = float(np.sum(np.tril(mat2, -1))) * 100
    mat3 = calculate_poisson_matrix(h_xg3, a_xg3)
    p3_home = float(np.sum(np.tril(mat3, -1))) * 100
    mat4 = calculate_poisson_matrix(h_xg4, a_xg4)
    p4_home = float(np.sum(np.tril(mat4, -1))) * 100
    mat5 = calculate_poisson_matrix(h_xg5, a_xg5)
    p5_home = float(np.sum(np.tril(mat5, -1))) * 100
    mat6 = calculate_poisson_matrix(h_xg6, a_xg6)
    p6_home = float(np.sum(np.tril(mat6, -1))) * 100
    
    final_home_win = round((p1_home + p2_home + p3_home + p4_home + p5_home + p6_home) / 6, 1)
    final_away_win = round(max(5.0, 100 - final_home_win - 15.0), 1)
    final_draw = round(15.0, 1)
    
    final_h_xg = (h_xg2 + h_xg4 + h_xg5 + h_xg6) / 4
    final_a_xg = (a_xg2 + a_xg4 + a_xg5 + a_xg6) / 4
    mat_final = calculate_poisson_matrix(final_h_xg, final_a_xg)
    
    best_score = np.unravel_index(np.argmax(mat_final), mat_final.shape)
    
    over_5_5_prob = 0.0
    for h in range(10):
        for a in range(10):
            if h + a > 5.5:
                over_5_5_prob += mat_final[h, a]
                
    home_players_data = fetch_all_players_projections(home_abbrev, final_h_xg, match.home_team)
    away_players_data = fetch_all_players_projections(away_abbrev, final_a_xg, match.away_team)
    
    return {
        "match": f"{match.home_team} vs {match.away_team}",
        "probabilities": {"home_win": final_home_win, "draw": final_draw, "away_win": final_away_win},
        "predicted_score": f"{int(best_score[0])} - {int(best_score[1])}",
        "over_under_5_5": {"over_prob": round(over_5_5_prob * 100, 1), "under_prob": round((1 - over_5_5_prob) * 100, 1)},
        "goalies": {"home": home_goalie, "away": away_goalie},
        "special_teams": {"home_pp": round(home_st["pp_pct"] * 100, 1), "away_pp": round(away_st["pp_pct"] * 100, 1)},
        "historical_factors": {"home_h2h_boost": h2h_factor_home, "away_h2h_boost": h2h_factor_away},
        "players": {"home": home_players_data, "away": away_players_data}
    }

@app.post("/api/history/add")
def add_bet_to_history(bet: BetRecord):
    history = load_history()
    entry_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    history[entry_id] = {
        "match": bet.match,
        "prediction": bet.prediction,
        "odds": bet.odds,
        "status": bet.status,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    save_history(history)
    return {"status": "success", "message": "Pari enregistré avec succès dans l'historique."}

@app.get("/api/history/list")
def get_history():
    history = load_history()
    total = len(history)
    won = sum(1 for v in history.values() if v.get("status") == "Gagné")
    lost = sum(1 for v in history.values() if v.get("status") == "Perdu")
    win_rate = round((won / max(1, (won + lost))) * 100, 1)
    
    return {
        "stats": {"total": total, "won": won, "lost": lost, "win_rate": win_rate},
        "history": history
    }

@app.get("/api/history/sync-3y")
def sync_historical_data():
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
                    games = data.get("games", [])
                    for g in games:
                        if g.get("gameState") in ["OFF", "FINAL"]:
                            home_team = g.get("homeTeam", {}).get("abbrev")
                            away_team = g.get("awayTeam", {}).get("abbrev")
                            home_score = g.get("homeTeam", {}).get("score", 0)
                            away_score = g.get("awayTeam", {}).get("score", 0)
                            
                            compiled_data[team].append({
                                "date": g.get("gameDate"),
                                "home": home_team,
                                "away": away_team,
                                "home_score": home_score,
                                "away_score": away_score
                            })
            except Exception:
                continue

    try:
        with open(NHL_HISTORY_DB, "w", encoding="utf-8") as f:
            json.dump(compiled_data, f, ensure_ascii=False, indent=2)
        return {"status": "success", "message": "Historique des 3 dernières années synchronisé avec succès !"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/matches/nhl")
def get_nhl_matches(date: str = "now"):
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d") if date != "now" else datetime.now()
        start_date = (target_date - timedelta(days=1)).strftime("%Y-%m-%d")
        
        url = f"https://api-web.nhle.com/v1/schedule/{start_date}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            matches = []
            for day in data.get("gameWeek", []):
                game_date = day.get("date")
                for g in day.get("games", []):
                    home_obj = g.get("homeTeam", {})
                    away_obj = g.get("awayTeam", {})
                    
                    home_name = home_obj.get("placeName", {}).get("default") or home_obj.get("commonName", {}).get("default", "Domicile")
                    away_name = away_obj.get("placeName", {}).get("default") or away_obj.get("commonName", {}).get("default", "Extérieur")
                    
                    if home_name.lower() == "new york":
                        logo_url = home_obj.get("logo", "").lower()
                        if "nyi" in logo_url or "islanders" in logo_url:
                            home_name = "New York Islanders"
                        else:
                            home_name = "New York Rangers"
                            
                    if away_name.lower() == "new york":
                        logo_url = away_obj.get("logo", "").lower()
                        if "nyi" in logo_url or "islanders" in logo_url:
                            away_name = "New York Islanders"
                        else:
                            away_name = "New York Rangers"

                    matches.append({
                        "id": g.get("id"), "date": game_date, "home_team": home_name, "away_team": away_name,
                        "home_logo": home_obj.get("logo", ""), "away_logo": away_obj.get("logo", ""), "time": g.get("startTimeUTC")
                    })
            return {"matches": matches}
        return {"matches": []}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/football/matches")
def get_football_matches():
    return {"sport": "Football", "matches": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
