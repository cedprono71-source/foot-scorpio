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
def get_nhl_matches(date: str = "now"):
    """
    Récupère le calendrier de la NHL. 
    Laissez vide pour la semaine courante ou passez une date au format AAAA-MM-JJ (ex: /api/matches/nhl?date=2026-11-12)
    """
    url = f"https://api-web.nhle.com/v1/schedule/{date}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            game_weeks = data.get("gameWeek", [])
            matches = []
            
            for day in game_weeks:
                game_date = day.get("date")
                games = day.get("games", [])
                for g in games:
                    home = g.get("homeTeam", {}).get("placeName", {}).get("default") or g.get("homeTeam", {}).get("commonName", {}).get("default", "Domicile")
                    away = g.get("awayTeam", {}).get("placeName", {}).get("default") or g.get("awayTeam", {}).get("commonName", {}).get("default", "Extérieur")
                    
                    matches.append({
                        "id": g.get("id"),
                        "date": game_date,
                        "home_team": home,
                        "away_team": away,
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
