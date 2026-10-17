from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
from scipy.stats import poisson
import requests
import os

app = FastAPI(title="NHL IA Pronostics")

class MatchInput(BaseModel):
    home_team: str
    away_team: str
    home_expected_goals: float = 3.0
    away_expected_goals: float = 2.7

@app.post("/predict/match")
def predict_match(match: MatchInput):
    max_goals = 10
    matrix = np.zeros((max_goals, max_goals))
    for h in range(max_goals):
        for a in range(max_goals):
            matrix[h, a] = poisson.pmf(h, match.home_expected_goals) * poisson.pmf(a, match.away_expected_goals)
            
    prob_home = float(np.sum(np.tril(matrix, -1)))
    prob_draw = float(np.sum(np.diag(matrix)))
    prob_away = float(np.sum(np.triu(matrix, 1)))
    
    return {
        "match": f"{match.home_team} vs {match.away_team}",
        "probabilities": {
            "home_win": round(prob_home * 100, 1),
            "draw": round(prob_draw * 100, 1),
            "away_win": round(prob_away * 100, 1)
        }
    }

@app.get("/api/matches/nhl")
def get_nhl_matches(date: str = "now"):
    url = f"https://api-web.nhle.com/v1/schedule/{date}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            matches = []
            for day in data.get("gameWeek", []):
                game_date = day.get("date")
                for g in day.get("games", []):
                    home = g.get("homeTeam", {}).get("placeName", {}).get("default") or g.get("homeTeam", {}).get("commonName", {}).get("default", "Domicile")
                    away = g.get("awayTeam", {}).get("placeName", {}).get("default") or g.get("awayTeam", {}).get("commonName", {}).get("default", "Extérieur")
                    matches.append({"id": g.get("id"), "date": game_date, "home_team": home, "away_team": away, "time": g.get("startTimeUTC")})
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
