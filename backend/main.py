import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Multi-IA Pronostics API", version="20.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Votre clé d'API Football-Data
API_KEY = os.getenv("API_KEY", "VOTRE_CLE_FOOTBALL_DATA")

class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    home_id: int = None
    away_id: int = None


# --- Moteur Consensus 5 IA ---
def ai_poisson_model(home_team, away_team): return {"home": 45.0, "draw": 25.0, "away": 30.0}
def ai_machine_learning_model(home_team, away_team): return {"home": 48.0, "draw": 22.0, "away": 30.0}
def ai_streak_model(home_team, away_team): return {"home": 42.0, "draw": 28.0, "away": 30.0}
def ai_h2h_model(home_team, away_team): return {"home": 44.0, "draw": 24.0, "away": 32.0}
def ai_fatigue_model(home_team, away_team): return {"home": 46.0, "draw": 22.0, "away": 32.0}

def calculate_5_ai_consensus(home_team, away_team):
    p1 = ai_poisson_model(home_team, away_team)
    p2 = ai_machine_learning_model(home_team, away_team)
    p3 = ai_streak_model(home_team, away_team)
    p4 = ai_h2h_model(home_team, away_team)
    p5 = ai_fatigue_model(home_team, away_team)
    
    final_home = p1['home'] * 0.35 + p2['home'] * 0.25 + p3['home'] * 0.15 + p4['home'] * 0.125 + p5['home'] * 0.125
    final_away = p1['away'] * 0.35 + p2['away'] * 0.25 + p3['away'] * 0.15 + p4['away'] * 0.125 + p5['away'] * 0.125
    final_draw = max(5.0, 100.0 - (final_home + final_away))

    return {
        "home_win": round(final_home, 1),
        "draw": round(final_draw, 1),
        "away_win": round(final_away, 1)
    }


# --- ROUTE PRINCIPALE ---
@app.get("/", response_class=HTMLResponse)
def serve_home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h3>Erreur : Le fichier index.html est introuvable sur le serveur.</h3>"


# --- Route Football ---
@app.get("/api/matches/foot")
def get_foot_matches(date: str):
    url = f"https://api.football-data.org/v4/matches?date={date}"
    headers = {"X-Auth-Token": API_KEY}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return {"date": date, "matches": [], "error": f"Erreur API ({response.status_code})" }
        
        data = response.json()
        formatted_matches = []
        
        for m in data.get("matches", []):
            utc_date_str = m.get("utcDate", "")
            game_time = utc_date_str[11:16] if len(utc_date_str) >= 16 else ""
            status = m.get("status")
            game_state = "OFF" if status == "FINISHED" else game_time

            formatted_matches.append({
                "home_team": m.get("homeTeam", {}).get("name", "Domicile"),
                "away_team": m.get("awayTeam", {}).get("name", "Extérieur"),
                "home_id": m.get("homeTeam", {}).get("id"),
                "away_id": m.get("awayTeam", {}).get("id"),
                "home_logo": m.get("homeTeam", {}).get("crest", ""),
                "away_logo": m.get("awayTeam", {}).get("crest", ""),
                "home_score": m.get("score", {}).get("fullTime", {}).get("home"),
                "away_score": m.get("score", {}).get("fullTime", {}).get("away"),
                "game_state": game_state
            })
            
        return {"date": date, "matches": formatted_matches}
    except Exception as e:
        return {"date": date, "matches": [], "error": str(e)}

@app.post("/predict/match/foot")
def predict_foot_match(req: MatchRequest):
    probs = calculate_5_ai_consensus(req.home_team, req.away_team)
    return {
        "match": f"{req.home_team} vs {req.away_team}",
        "probabilities": probs,
        "predicted_score": "2 - 1",
        "btts": {"yes_prob": 62.5},
        "over_under_2_5": {"over_prob": 58.0}
    }


# --- Route NHL (Corrigée pour récupérer les vrais noms d'équipes) ---
@app.get("/api/matches/nhl")
def get_nhl_matches(date: str):
    url = f"https://api-web.nhle.com/v1/score/{date}"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return {"date": date, "matches": []}
            
        data = response.json()
        formatted_matches = []
        
        for game in data.get("games", []):
            home_obj = game.get("homeTeam", {})
            away_obj = game.get("awayTeam", {})
            
            # Récupération sécurisée du nom ou de l'abréviation de l'équipe
            home_team = home_obj.get("placeName", {}).get("default") or home_obj.get("abbrev") or "Home"
            away_team = away_obj.get("placeName", {}).get("default") or away_obj.get("abbrev") or "Away"
            
            # Si on a un nom de ville, on ajoute le nom usuel si possible
            home_common = home_obj.get("commonName", {}).get("default", "")
            away_common = away_obj.get("commonName", {}).get("default", "")
            if home_common:
                home_team = f"{home_team} {home_common}"
            if away_common:
                away_team = f"{away_team} {away_common}"

            home_logo = home_obj.get("logo", "")
            away_logo = away_obj.get("logo", "")
            
            home_score = home_obj.get("score")
            away_score = away_obj.get("score")
            
            state = game.get("gameState")
            game_state = "OFF" if state in ["OFF", "FINAL"] else game.get("startTimeUTC", "")[11:16]

            formatted_matches.append({
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "home_logo": home_logo,
                "away_logo": away_logo,
                "game_state": game_state
            })
            
        return {"date": date, "matches": formatted_matches}
    except Exception as e:
        return {"date": date, "matches": [], "error": str(e)}

@app.post("/predict/match")
def predict_nhl_match(req: MatchRequest):
    probs = calculate_5_ai_consensus(req.home_team, req.away_team)
    return {
        "match": f"{req.home_team} vs {req.away_team}",
        "probabilities": probs,
        "predicted_score": "3 - 2",
        "over_under_5_5": {"over_prob": 56.6},
        "home_players": [
            {"name": "Meilleur Buteur Domicile", "pos": "F", "stats": "En forme", "pct": 65}
        ],
        "away_players": [
            {"name": "Meilleur Buteur Extérieur", "pos": "F", "stats": "En forme", "pct": 60}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
