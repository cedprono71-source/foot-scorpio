from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uvicorn
import random

app = FastAPI(title="NHL & Football Prediction Dashboard API", version="2.0")

# Autoriser les requêtes (CORS) pour que votre interface communique sans souci
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# MOTEUR DE DÉTECTION DES "VALEURS CACHÉES / COTE 3+"
# ---------------------------------------------------------
def find_value_goalscorers(players_list, opponent_def_rating):
    """
    Analyse la liste des joueurs d'une équipe pour dénicher 
    les 'pépites' (cotes estimées à 3+ ou à forte valeur) en se basant 
    sur le volume de tirs, le temps de glace et l'unité de PP.
    """
    value_picks = []
    
    for player in players_list:
        shots_per_game = player.get("shots_per_game", 1.5)
        is_on_pp2 = player.get("pp_unit", 2) == 2  # 2ème unité = cotes plus hautes
        ice_time = player.get("toi_minutes", 15.0)
        
        # Calcul du score de valeur cachée
        score_value = (shots_per_game * 1.5) + (ice_time * 0.1)
        if is_on_pp2:
            score_value *= 1.2  # Bonus pour la 2ème unité (cotes souvent > 3.00)
            
        # Si le score dépasse le seuil et que l'adversaire est friable en défense
        if score_value > 4.0 and opponent_def_rating > 3.0:
            value_picks.append({
                "player_name": player.get("name", "Joueur Inconnu"),
                "estimated_odds": "~3.40",
                "reason": f"Gros volume de tirs ({shots_per_game}/match) et aligné en PP2 face à une défense perméable."
            })
            
    # On retourne la meilleure pépite trouvée (ou un message par défaut si vide)
    if value_picks:
        return value_picks[0]
    return {
        "player_name": "Aucune value évidente",
        "estimated_odds": "N/A",
        "reason": "Match trop serré ou profils équilibrés."
    }

# ---------------------------------------------------------
# ROUTES API DE L'APPLICATION
# ---------------------------------------------------------

@app.get("/")
def read_root():
    return {"status": "online", "message": "API NHL & Football opérationnelle avec moteur Value Bets !"}

@app.get("/api/nhl/matches")
async def get_nhl_matches():
    """
    Récupère ou simule les matchs NHL du jour avec intégration 
    du calcul de pépite (cote 3+).
    """
    matches = [
        {
            "id": 101,
            "home_team": "Edmonton Oilers",
            "away_team": "Toronto Maple Leafs",
            "time": "01:00",
            "prediction": "Edmonton Oilers (58%)",
            "over_under": "Over 6.5 buts",
            "opponent_def_rating": 3.8,
            "home_players": [
                {"name": "Evander Kane", "shots_per_game": 3.2, "pp_unit": 2, "toi_minutes": 17.5},
                {"name": "Ryan Nugent-Hopkins", "shots_per_game": 2.1, "pp_unit": 1, "toi_minutes": 20.0}
            ]
        },
        {
            "id": 102,
            "home_team": "Boston Bruins",
            "away_team": "Tampa Bay Lightning",
            "time": "01:30",
            "prediction": "Tampa Bay Lightning (52%)",
            "over_under": "Under 5.5 buts",
            "opponent_def_rating": 2
