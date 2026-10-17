from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
from scipy.stats import poisson
import requests
import os
from datetime import datetime, timedelta

app = FastAPI(title="Multi-IA NHL Pronostics")

class MatchInput(BaseModel):
    home_team: str
    away_team: str
    home_expected_goals: float = 3.1
    away_expected_goals: float = 2.7
    home_b2b: bool = False
    away_b2b: bool = False

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
                
                goals_for = record.get("goalFor", 0)
                goals_against = record.get("goalAgainst", 0)
                games_played = record.get("gamesPlayed", 1)
                if games_played < 1:
                    games_played = 1
                
                offense = round(goals_for / games_played, 2)
                defense = round(goals_against / games_played, 2)
                stats_map[full_name.lower()] = {"offense": max(2.0, offense), "defense": max(2.0, defense)}
    except Exception:
        pass
    return stats_map

def calculate_poisson_matrix(home_xg, away_xg):
    max_goals = 10
    matrix = np.zeros((max_goals, max_goals))
    for h in range(max_goals):
        for a in range(max_goals):
            matrix[h, a] = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
    return matrix

@app.post("/predict/match")
def predict_match(match: MatchInput):
    team_stats = fetch_team_stats()
    
    home_st = {"offense": 3.1, "defense": 2.7}
    away_st = {"offense": 2.9, "defense": 2.9}
    
    for name, st in team_stats.items():
        if name in match.home_team.lower() or match.home_team.lower() in name:
            home_st = st
        if name in match.away_team.lower() or match.away_team.lower() in name:
            away_st = st

    # --- IA 1 : Modèle Poisson pur (Global) ---
    mat1 = calculate_poisson_matrix(3.1, 2.7)
    p1_home = float(np.sum(np.tril(mat1, -1))) * 100

    # --- IA 2 : Modèle Dynamique (Basé sur le classement actuel) ---
    h_xg2 = home_st["offense"] * 1.08
    a_xg2 = away_st["offense"] * 0.92
    mat2 = calculate_poisson_matrix(h_xg2, a_xg2)
    p2_home = float(np.sum(np.tril(mat2, -1))) * 100

    # --- IA 3 : Modèle Tactique / Momentum (Pondéré par la défense) ---
    h_xg3 = max(1.5, home_st["offense"] - (away_st["defense"] * 0.1) + 0.2)
    a_xg3 = max(1.5, away_st["offense"] - (home_st["defense"] * 0.1))
    mat3 = calculate_poisson_matrix(h_xg3, a_xg3)
    p3_home = float(np.sum(np.tril(mat3, -1))) * 100

    # Consensus global des 3 IA
    final_home_win = round((p1_home + p2_home + p3_home) / 3, 1)
    final_away_win = round(100 - final_home_win - 15.0, 1) # Marge de tirage
    final_draw = round(15.0, 1)

    # Score le plus probable de l'IA principale (Modèle 2)
    best_score = np.unravel_index(np.argmax(mat2), mat2.shape)

    # Over / Under 5.5
    over_5_5_prob = 0.0
    for h in range(10):
        for a in range(10):
            if h + a > 5.5:
                over_5_5_prob += mat2[h, a]

    return {
        "match": f"{match.home_team} vs {match.away_team}",
        "probabilities": {
            "home_win": final_home_win,
            "draw": final_draw,
            "away_win": final_away_win
        },
        "ai_consensus": {
            "ia_standard": round(p1_home, 1),
            "ia_stats_saison": round(p2_home, 1),
            "ia_tactique": round(p3_home, 1)
        },
        "predicted_score": f"{int(best_score[0])} - {int(best_score[1])}",
        "over_under_5_5": {
            "over_prob": round(over_5_5_prob * 100, 1),
            "under_prob": round((1 - over_5_5_prob) * 100, 1)
        }
    }

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
                    
                    home_name = home_obj.get("placeName", {}).default if hasattr(home_obj.get("placeName", {}), "default") else home_obj.get("placeName", {}).get("default", "Domicile")
                    away_name = away_obj.get("placeName", {}).get("default") or away_obj.get("commonName", {}).get("default", "Extérieur")
                    
                    matches.append({
                        "id": g.get("id"),
                        "date": game_date,
                        "home_team": home_name,
                        "away_team": away_name,
                        "home_logo": home_obj.get("logo", ""),
                        "away_logo": away_obj.get("logo", ""),
                        "time": g.get("startTimeUTC")
                    })
            return {"matches": matches}
        return {"matches": []}
    except Exception as e:
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
def lire_index():
    base_dir = os.path.dirname(__file__)
    for path in [os.path.join(base_dir, "..", "index.html"), os.path.join(base_dir, "index.html"), "index.html"]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read(), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return "<h1>index.html introuvable</h1>"
