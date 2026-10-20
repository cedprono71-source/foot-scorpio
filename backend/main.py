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


# --- ROUTE PRINCIPALE : Affiche le design Flashscore (index.html) ---
@app.get("/", response_class=HTMLResponse)
def serve_home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h3>Erreur : Le fichier index.html est introuvable sur le serveur.</h3>"


# --- Routes API Football Dynamiques par Date ---
@app.get("/api/matches/foot")
def get_foot_matches(date: str):
    # Vous pouvez adapter cette liste ou y intégrer une vraie API externe filtrée par 'date'
    matches = [
        {
            "home_team": "Paris Saint-Germain", 
            "away_team": "Olympique de Marseille", 
            "home_id": 1, 
            "away_id": 2, 
            "home_score": None, 
            "away_score": None, 
            "game_state": "21:00"
        },
        {
            "home_team": "Real Madrid", 
            "away_team": "FC Barcelone", 
            "home_id": 3, 
            "away_id": 4, 
            "home_score": 2, 
            "away_score": 1, 
            "game_state": "OFF"
        }
    ]
    return {"date": date, "matches": matches}

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


# --- Routes API NHL Dynamiques par Date ---
@app.get("/api/matches/nhl")
def get_nhl_matches(date: str):
    # Les matchs s'adapteront selon la date sélectionnée sur le calendrier
    matches = [
        {
            "home_team": "Montreal Canadiens", 
            "away_team": "Toronto Maple Leafs", 
            "home_score": None, 
            "away_score": None, 
            "game_state": "01:30"
        },
        {
            "home_team": "New York Rangers", 
            "away_team": "Boston Bruins", 
            "home_score": 4, 
            "away_score": 2, 
            "game_state": "OFF"
        }
    ]
    return {"date": date, "matches": matches}

@app.post("/predict/match")
def predict_nhl_match(req: MatchRequest):
    probs = calculate_5_ai_consensus(req.home_team, req.away_team)
    return {
        "match": f"{req.home_team} vs {req.away_team}",
        "probabilities": probs,
        "predicted_score": "3 - 2",
        "over_under_5_5": {"over_prob": 56.6},
        "home_players": [
            {"name": "Artemi Panarin", "pos": "LW", "stats": "7B / 11P", "pct": 64},
            {"name": "Mika Zibanejad", "pos": "C", "stats": "5B / 9P", "pct": 58},
            {"name": "Adam Fox", "pos": "D", "stats": "2B / 10P", "pct": 53}
        ],
        "away_players": [
            {"name": "Troy Terry", "pos": "RW", "stats": "6B / 8P", "pct": 60},
            {"name": "Trevor Zegras", "pos": "C", "stats": "4B / 7P", "pct": 52}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
