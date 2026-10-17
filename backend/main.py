from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
from scipy.stats import poisson
import requests
import os

app = FastAPI(
    title="API Pronostics IA - Foot & Hockey",
    description="Backend de calcul des probabilités et récupération des matchs",
    version="1.0"
)

class Player(BaseModel):
    name: str
    position: str
    goal_share: float
    is_penalty_taker: bool = False
    powerplay_line1: bool = False

class MatchInput(BaseModel):
    home_team: str
    away_team: str
    home_expected_goals: float
    away_expected_goals: float
    home_players: list[Player] = []
    away_players: list[Player] = []

@app.post("/predict/match")
def predict_match(match: MatchInput):
    max_goals = 10
    matrix = np.zeros((max_goals, max_goals))
    
    for h in range(max_goals):
        for a in range(max_goals):
            prob_h = poisson.pmf(h, match.home_expected_goals)
            prob_a = poisson.pmf(a, match.away_expected_goals)
            matrix[h, a] = prob_h * prob_a
            
    prob_home_win = float(np.sum(np.tril(matrix, -1)))
    prob_draw = float(np.sum(np.diag(matrix)))
    prob_away_win = float(np.sum(np.triu(matrix, 1)))
    
    return {
        "match": f"{match.home_team} vs {match.away_team}",
        "probabilities": {
            "home_win": round(prob_home_win * 100, 2),
            "draw": round(prob_draw * 100, 2),
            "away_win": round(prob_away_win * 100, 2)
        }
    }

@app.get("/api/matches/nhl")
def get_todays_nhl_matches():
    # Endpoint du calendrier officiel de la NHL
    url = "https://api-web.nhle.com/v1/schedule/now"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            game_week = data.get("gameWeek", [])
            if game_week:
                # Récupère les matchs de la première journée disponible dans la semaine
                games = game_week[0].get("games", [])
                matches = []
                for g in games:
                    matches.append({
                        "id": g.get("id"),
                        "home_team": g.get("homeTeam", {}).get("placeName", {}).get("default", "Home"),
                        "away_team": g.get("awayTeam", {}).get("placeName", {}).get("default", "Away"),
                        "time": g.get("startTimeUTC")
                    })
                return {"matches": matches}
        return {"matches": []}
    except Exception as e:
        return {"error": str(e)}

# Route unique pour la page d'accueil qui sert le fichier index.html
@app.get("/", response_class=HTMLResponse)
def lire_index():
    chemins_possibles = [
        os.path.join(os.path.dirname(__file__), "..", "index.html"),
        os.path.join(os.path.dirname(__file__), "index.html"),
        "index.html",
        "../index.html"
    ]
    for chemin in chemins_possibles:
        if os.path.exists(chemin):
            with open(chemin, "r", encoding="utf-8") as f:
                return f.read()
    return "<h1>Fichier index.html introuvable</h1>"
