from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from scipy.stats import poisson
import requestsfrom fastapi.responses import HTMLResponse
import os

app = FastAPI(
    title="API Pronostics IA - Foot & Hockey",
    description="Backend de calcul des probabilités inspiré de Flashscore",
    version="1.0"
)

class Player(BaseModel):
    name: str
    position: str
    goal_share: float
    is_penalty_taker: bool = False
    powerplay_line1: bool = False

class MatchRequest(BaseModel):
    sport: str
    home_team: str
    away_team: str
    home_xg: float
    away_xg: float
    home_players: list[Player]
    away_players: list[Player]

def calculate_match_probabilities(home_xg, away_xg, max_goals=6):
    home_probs = [poisson.pmf(i, home_xg) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(i, away_xg) for i in range(max_goals + 1)]
    score_matrix = np.outer(home_probs, away_probs)
    
    prob_home_win = np.sum(np.tril(score_matrix, -1)) * 100
    prob_draw = np.sum(np.diag(score_matrix)) * 100
    prob_away_win = np.sum(np.triu(score_matrix, 1)) * 100
    
    total_goals_prob = sum(score_matrix[h, a] for h in range(max_goals + 1) for a in range(max_goals + 1) if h + a > 2.5)
    prob_btts = (1 - score_matrix[0, :].sum() - score_matrix[:, 0].sum() + score_matrix[0, 0]) * 100

    return {
        "1X2": {
            "domicile": round(prob_home_win, 1),
            "nul": round(prob_draw, 1),
            "exterieur": round(prob_away_win, 1)
        },
        "over_2_5_buts": round(total_goals_prob * 100, 1),
        "btts_oui": round(prob_btts, 1),
        "esperance_buts_totale": round(home_xg + away_xg, 2)
    }

def calculate_goalscorers(team_xg, players):
    if not players:
        return []
    total_share = sum(p.goal_share for p in players)
    scorers = []
    
    for player in players:
        player_share = player.goal_share / total_share
        player_xg = team_xg * player_share
        
        if player.is_penalty_taker or player.powerplay_line1:
            player_xg *= 1.15
            
        prob_to_score = (1 - np.exp(-player_xg)) * 100
        scorers.append({
            "joueur": player.name,
            "poste": player.position,
            "probabilite_but": round(prob_to_score, 1)
        })
        
    scorers.sort(key=lambda x: x["probabilite_but"], reverse=True)
    return scorers

@app.post("/api/predict")
def predict_match(data: MatchRequest):
    match_preds = calculate_match_probabilities(data.home_xg, data.away_xg)
    home_scorers = calculate_goalscorers(data.home_xg, data.home_players)
    away_scorers = calculate_goalscorers(data.away_xg, data.away_players)
    
    return {
        "sport": data.sport,
        "match": f"{data.home_team} vs {data.away_team}",
        "predictions_match": match_preds,
        "buteurs_probables": {
            data.home_team: home_scorers,
            data.away_team: away_scorers
        }
    }
# Route pour récupérer automatiquement les matchs NHL du jour
@app.get("/api/matches/nhl")
def get_todays_nhl_matches():
    url = "https://api-web.nhle.com/v1/schedule/now"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            game_week = data.get("gameWeek", [])
            if game_week:
                games = game_week[0].get("games", [])
                matches = []
                for g in games:
                    matches.append({
                        "id": g.get("id"),
                        "home_team": g.get("homeTeam", {}).get("placeName", {}).get("default"),
                        "away_team": g.get("awayTeam", {}).get("placeName", {}).get("default"),
                        "time": g.get("startTimeUTC")
                    })
                return {"matches": matches}
        return {"matches": []}
    except Exception as e:
        return {"error": str(e)}
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
from fastapi.responses import HTMLResponse
import os

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
