from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
from scipy.stats import poisson
import requests
import os

app = FastAPI(title="NHL IA Pronostics")

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
