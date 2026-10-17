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
                abbrev = record.get("teamAbbrev", {}).get("default", "")
                
                goals_for = record.get("goalFor", 0)
                goals_against = record.get("goalAgainst", 0)
                games_played = max(1, record.get("gamesPlayed", 1))
                
                offense = round(goals_for / games_played, 2)
                defense = round(goals_against / games_played, 2)
                
                # Taux approximatifs PP / PK basés sur les buts (pondérés par défaut si non fournis)
                pp_pct = round(min(0.30, max(0.12, (offense / 3.0) * 0.20)), 3)
                pk_pct = round(min(0.88, max(0.70, 0.80 + (2.8 - defense) * 0.05)), 3)

                stats_map[full_name.lower()] = {
                    "abbrev": abbrev,
                    "offense": max(2.0, offense), 
                    "defense": max(2.0, defense),
                    "pp_pct": pp_pct,
                    "pk_pct": pk_pct
                }
    except Exception:
        pass
    return stats_map

def fetch_goalie_stats(team_abbrev):
    if not team_abbrev:
        return {"name": "Gardien Titulaire", "save_pct": 0.905, "gaa": 2.80}
    try:
        url = f"https://api-web.nhle.com/v1/club-stats/{team_abbrev}/now"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            goalies = data.get("goalies", [])
            if goalies:
                top_goalie = max(goalies, key=lambda x: x.get("gamesPlayed", 0))
                name = f"{top_goalie.get('firstName', {}).get('default', '')} {top_goalie.get('lastName', {}).get('default', '')}"
                sv_pct = top_goalie.get("savePercentage", 0.905)
                gaa = top_goalie.get("goalsAgainstAverage", 2.80)
                return {
                    "name": name,
                    "save_pct": round(sv_pct, 3) if sv_pct else 0.905,
                    "gaa": round(gaa, 2) if gaa else 2.80
                }
    except Exception:
        pass
    return {"name": "Gardien Titulaire", "save_pct": 0.905, "gaa": 2.80}

def fetch_all_players_projections(team_abbrev, expected_team_goals):
    players = []
    top_scorer, top_passer = None, None
    if not team_abbrev:
        return {"all": players, "top_scorer": top_scorer, "top_passer": top_passer}
        
    try:
        url = f"https://api-web.nhle.com/v1/club-stats/{team_abbrev}/now"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            skaters = data.get("skaters",
