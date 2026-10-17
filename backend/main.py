from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
from scipy.stats import poisson
import requests
import os
from datetime import datetime, timedelta

app = FastAPI(title="NHL IA Pronostics")

class MatchInput(BaseModel):
    home_team: str
    away_team: str
    home_expected_goals: float = 3.1
    away_expected_goals: float = 2.7
    home_b2b: bool = False
    away_b2b: bool = False

@app.post("/predict/match")
def predict_match(match: MatchInput):
    # Ajustement des buts attendus selon le Back-to-Back (fatigue)
    home_xg = match.home_expected_goals * (0.90 if match.home_b2b else 1.0)
    away_xg = match.away_expected_goals * (0.90 if match.away_b2b else 1.0)
    
    max_goals = 10
    matrix = np.zeros((max_goals, max_goals))
    
    # Matrice des probabilités de score exact
    for h in range(max_goals):
        for a in range(max_goals):
            matrix[h, a] = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
            
    prob_home = float(np.sum(np.tril(matrix, -1)))
    prob_draw = float(np.sum(np.diag(matrix)))
    prob_away = float(np.sum(np.triu(matrix, 1)))
    
    # Score le plus probable
    best_score = np.unravel_index(np.argmax(matrix), matrix.shape)
    predicted_home_goals = int(best_score[0])
    predicted_away_goals = int(best_score[1])
    
    # Over / Under 5.5 buts
    over_5_5_prob = 0.0
    for h in range(max_goals):
        for a in range(max_goals):
            if h + a > 5.5:
                over_5_5_prob += matrix[h, a]
                
    return {
        "match": f"{match.home_team} vs {match.away_team}",
        "probabilities": {
            "home_win": round(prob_home * 100, 1),
            "draw": round(prob_draw * 100, 1),
            "away_win": round(prob_away * 100, 1)
        },
        "predicted_score": f"{predicted_home_goals} - {predicted_away_goals}",
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
                    
                    home_name = home_obj.get("placeName", {}).get("default") or home_obj.get("commonName", {}).get("default", "Domicile")
                    away_name = away_obj.get("placeName", {}).get("default") or away_obj.get("commonName", {}).get("default", "Extérieur")
                    
                    home_logo = home_obj.get("logo", "")
                    away_logo = away_obj.get("logo", "")
                    
                    matches.append({
                        "id": g.get("id"),
                        "date": game_date,
                        "home_team": home_name,
                        "away_team": away_name,
                        "home_logo": home_logo,
                        "away_logo": away_logo,
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
