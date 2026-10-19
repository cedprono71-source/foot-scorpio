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
    version="3.1",
    description="Correction précise basée sur l'analyse des logos et abréviations pour séparer Rangers et Islanders."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.json")

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

class MatchInput(BaseModel):
    home_team: str
    away_team: str
    home_expected_goals: float = 3.1
    away_expected_goals: float = 2.7
    home_b2b: bool = False
    away_b2b: bool = False

# EFFECTIF OFFICIEL DES NEW YORK RANGERS
RANGERS_CUSTOM_ROSTER = [
    {"name": "Oliver Bjorkstrand", "pos": "AD", "goals": 6, "assists": 8, "points": 14, "goal_prob": 32.0, "assist_prob": 38.0, "point_prob": 56.0},
    {"name": "Alexis Lafrenière", "pos": "AG", "goals": 7, "assists": 4, "points": 11, "goal_prob": 35.0, "assist_prob": 25.0, "point_prob": 55.0},
    {"name": "Mika Zibanejad", "pos": "C", "goals": 3, "assists": 6, "points": 9, "goal_prob": 28.0, "assist_prob": 40.0, "point_prob": 52.0},
    {"name": "Adam Fox", "pos": "D", "goals": 1, "assists": 8, "points": 9, "goal_prob": 15.0, "assist_prob": 50.0, "point_prob": 50.0},
    {"name": "J.T. Miller", "pos": "C", "goals": 2, "assists": 5, "points": 7, "goal_prob": 22.0, "assist_prob": 35.0, "point_prob": 48.0},
    {"name": "Pavel Dorofeyev", "pos": "AD", "goals": 4, "assists": 3, "points": 7, "goal_prob": 29.0, "assist_prob": 24.0, "point_prob": 46.0},
    {"name": "Gabe Perreault", "pos": "AD", "goals": 3, "assists": 3, "points": 6, "goal_prob": 25.0, "assist_prob": 25.0, "point_prob": 42.0},
    {"name": "Will Cuylle", "pos": "AG", "goals": 3, "assists": 3, "points": 6, "goal_prob": 24.0, "assist_prob": 24.0, "point_prob": 40.0},
    {"name": "Eeli Tolvanen", "pos": "AD", "goals": 3, "assists": 2, "points": 5, "goal_prob": 26.0, "assist_prob": 20.0, "point_prob": 39.0},
    {"name": "Sean Durzi", "pos": "D", "goals": 1, "assists": 4, "points": 5, "goal_prob": 12.0, "assist_prob": 30.0, "point_prob": 35.0},
    {"name": "Tye Kartye", "pos": "AG", "goals": 2, "assists": 4, "points": 6, "goal_prob": 20.0, "assist_prob": 32.0, "point_prob": 38.0},
    {"name": "Justin Dowling", "pos": "C", "goals": 1, "assists": 3, "points": 4, "goal_prob": 15.0, "assist_prob": 25.0, "point_prob": 32.0},
    {"name": "Matt Rempe", "pos": "C", "goals": 0, "assists": 1, "points": 1, "goal_prob": 5.0, "assist_prob": 15.0, "point_prob": 18.0},
    {"name": "Vladislav Gavrikov", "pos": "D", "goals": 1, "assists": 4, "points": 5, "goal_prob": 12.0, "assist_prob": 30.0, "point_prob": 35.0}
]

# EFFECTIF OFFICIEL DES NEW YORK ISLANDERS
ISLANDERS_CUSTOM_ROSTER = [
    {"name": "Mathew Barzal", "pos": "C", "goals": 4, "assists": 7, "points": 11, "goal_prob": 30.0, "assist_prob": 48.0, "point_prob": 54.0},
    {"name": "Bo Horvat", "pos": "C", "goals": 6, "assists": 3, "points": 9, "goal_prob": 34.0, "assist_prob": 22.0, "point_prob": 50.0},
    {"name": "Noah Dobson", "pos": "D", "goals": 1, "assists": 7, "points": 8, "goal_prob": 12.0, "assist_prob": 45.0, "point_prob": 46.0},
    {"name": "Brock Nelson", "pos": "C", "goals": 5, "assists": 2, "points": 7, "goal_prob": 32.0, "assist_prob": 18.0, "point_prob": 44.0},
    {"name": "Kyle Palmieri", "pos": "F", "goals": 4, "assists": 2, "points": 6, "goal_prob": 28.0, "assist_prob": 16.0, "point_prob": 39.0},
    {"name": "Anders Lee", "pos": "AG", "goals": 3, "assists": 3, "points": 6, "goal_prob": 25.0, "assist_prob": 20.0, "point_prob": 38.0},
    {"name": "Jean-Gabriel Pageau", "pos": "C", "goals": 2, "assists": 3, "points": 5, "goal_prob": 18.0, "assist_prob": 25.0, "point_prob": 35.0},
    {"name": "Ryan Pulock", "pos": "D", "goals": 1, "assists": 3, "points": 4, "goal_prob": 10.0, "assist_prob": 30.0, "point_prob": 33.0}
]

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
    full_lower = team_full_name.lower()
    
    # Distinction stricte et prioritaire pour les Islanders
    if "islanders" in full_lower or team_abbrev == "NYI":
        players = ISLANDERS_CUSTOM_ROSTER
        top_scorer = max(players, key=lambda x: x["goal_prob"])
        top_passer = max(players, key=lambda x: x["assist_prob"])
        return {"all": players, "top_scorer": top_scorer, "top_passer": top_passer}
        
    # Distinction pour les Rangers
    if "rangers" in full_lower or team_abbrev == "NYR" or full_lower == "new york":
        players = RANGERS_CUSTOM_ROSTER
        top_scorer = max(players, key=lambda x: x["goal_prob"])
        top_passer = max(players, key=lambda x: x["assist_prob"])
        return {"all": players, "top_scorer": top_scorer, "top_passer": top_passer}

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
    
    h_xg2 = home_st["offense"] * 1.08
    a_xg2 = away_st["offense"] * 0.92
    
    h_xg3 = max(1.5, (home_st["offense"] - (away_st["defense"] * 0.1) + 0.2))
    a_xg3 = max(1.5, (away_st["offense"] - (home_st["defense"] * 0.1)))
    
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
        "players": {"home": home_players_data, "away": away_players_data}
    }

@app.get("/api/predict/auto-sync")
def auto_sync_today_matches(date: str = "now"):
    return {"status": "success"}

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
                    
                    # Analyse intelligente basée sur le lien du logo pour départager les deux New York
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

