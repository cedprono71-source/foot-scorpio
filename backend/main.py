from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from scipy.stats import poisson
import requests
import os
import json
from datetime import datetime, timedelta

app = FastAPI(
    title="Multi-IA NHL & Football Dashboard API",
    version="2.0",
    description="API complète combinant prédictions Poisson, données NHL en direct, historique et module IA Pépites (Cote 3+)."
)

# Configuration CORS pour le front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.json")

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

class MatchInput(BaseModel):
    home_team: str
    away_team: str
    home_expected_goals: float = 3.1
    away_expected_goals: float = 2.7
    home_b2b: bool = False
    away_b2b: bool = False

# ---------------------------------------------------------
# MOTEUR AVANCÉ : IA PÉPITES & VALEUR CACHÉE (COTE 3+)
# ---------------------------------------------------------
def find_value_goalscorers(home_players, away_players, home_def_rating, away_def_rating):
    """
    Analyse les deux équipes pour dénicher les 'pépites' 
    (joueurs de profondeur, volume de tirs) offrant des cotes dynamiques à 3+.
    """
    all_candidates = []

    # Analyse des joueurs de l'équipe domicile face à la défense adverse
    for player in home_players:
        shots = player.get("shots_per_game", 1.8)
        is_pp2 = player.get("pp_unit", 2) == 2
        toi = player.get("toi_minutes", 16.0)
        
        score = (shots * 1.6) + (toi * 0.1)
        if is_pp2:
            score *= 1.25  # Bonus unité spéciale secondaire (cotes > 3.00)
            
        if score > 3.8 and away_def_rating > 2.8:
            # Calcul dynamique de la cote en fonction du score (entre 3.10 et 4.80)
            dynamic_odds = round(max(3.10, min(4.80, 5.2 - (score * 0.20))), 2)
            
            all_candidates.append({
                "player_name": player.get("name", "Inconnu"),
                "team": "Domicile",
                "estimated_odds": f"~{dynamic_odds:.2f}",
                "confidence": "Élevée" if score > 4.5 else "Moyenne",
                "reason": f"Gros volume de tirs ({shots} tirs/m) et PP2 face à une défense extérieure friable."
            })

    # Analyse des joueurs de l'équipe extérieure face à la défense locale
    for player in away_players:
        shots = player.get("shots_per_game", 1.8)
        is_pp2 = player.get("pp_unit", 2) == 2
        toi = player.get("toi_minutes", 16.0)
        
        score = (shots * 1.6) + (toi * 0.1)
        if is_pp2:
            score *= 1.25
            
        if score > 3.8 and home_def_rating > 2.8:
            # Calcul dynamique de la cote pour l'équipe extérieure
            dynamic_odds = round(max(3.10, min(4.80, 5.3 - (score * 0.20))), 2)
            
            all_candidates.append({
                "player_name": player.get("name", "Inconnu"),
                "team": "Extérieur",
                "estimated_odds": f"~{dynamic_odds:.2f}",
                "confidence": "Élevée" if score > 4.5 else "Moyenne",
                "reason": f"Impact offensif et tirs répétés sur la 2ème unité face à la défense locale."
            })

    if all_candidates:
        return all_candidates[0]
    
    return {
        "player_name": "Aucun value bet majeur",
        "team": "N/A",
        "estimated_odds": "N/A",
        "confidence": "Faible",
        "reason": "Rencontre trop cadenassée ou statistiques équilibrées."
    }

# ---------------------------------------------------------
# FONCTIONS DE RÉCUPÉRATION DES STATS NHL (OFFICIEL)
# ---------------------------------------------------------
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

def fetch_team_roster_and_injuries(team_abbrev):
    injuries = []
    injury_malus = 1.0
    if not team_abbrev:
        return {"injuries": injuries, "malus": injury_malus}
    
    try:
        url = f"https://api-web.nhle.com/v1/roster/{team_abbrev}/current"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for group in ["forwards", "defensemen"]:
                for player in data.get(group, []):
                    if player.get("status") in ["IR", "LTIR", "OUT"]:
                        p_name = f"{player.get('firstName', {}).get('default', '')} {player.get('lastName', {}).get('default', '')}"
                        injuries.append(p_name)
                        injury_malus -= 0.05
    except Exception:
        pass
        
    return {"injuries": injuries, "malus": max(0.75, round(injury_malus, 2))}

def fetch_all_players_projections(team_abbrev, expected_team_goals):
    players = []
    top_scorer, top_passer = None, None
    if not team_abbrev:
        return {"all": players, "top_scorer": top_scorer, "top_passer": top_passer}
        
    try:
        active_player_ids = set()
        roster_url = f"https://api-web.nhle.com/v1/roster/{team_abbrev}/current"
        roster_res = requests.get(roster_url)
        if roster_res.status_code == 200:
            roster_data = roster_res.json()
            for group in ["forwards", "defensemen"]:
                for p in roster_data.get(group, []):
                    if p.get("status") in ["IR", "LTIR", "OUT"]:
                        continue
                    active_player_ids.add(p.get("id"))

        url = f"https://api-web.nhle.com/v1/club-stats/{team_abbrev}/now"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            skaters = data.get("skaters", [])
            
            for p in skaters:
                name = f"{p.get('firstName', {}).get('default', '')} {p.get('lastName', {}).get('default', '')}"
                p_id = p.get("playerId")
                if active_player_ids and p_id not in active_player_ids:
                    continue
                
                gp = max(1, p.get("gamesPlayed", 1))
                goals = p.get("goals", 0)
                assists = p.get("assists", 0)
                points = goals + assists
                
                g_pg = goals / gp
                a_pg = assists / gp
                pts_pg = points / gp
                
                goal_prob = min(90, round((1 - np.exp(-g_pg * (expected_team_goals / 3.0))) * 100, 1))
                assist_prob = min(90, round((1 - np.exp(-a_pg * (expected_team_goals / 3.0))) * 100, 1))
                point_prob = min(95, round((1 - np.exp(-pts_pg * (expected_team_goals / 3.0))) * 100, 1))
                
                position = p.get("positionCode", "F")
                
                # Simulation de métriques additionnelles pour l'IA Pépites
                players.append({
                    "name": name,
                    "pos": position,
                    "goals": goals,
                    "assists": assists,
                    "points": points,
                    "goal_prob": goal_prob,
                    "assist_prob": assist_prob,
                    "point_prob": point_prob,
                    "shots_per_game": round(g_pg * 2.5 + 1.2, 1),
                    "pp_unit": 2 if goals < 5 else 1,
                    "toi_minutes": 16.5
                })
                
            players = sorted(players, key=lambda x: x["point_prob"], reverse=True)
            if players:
                top_scorer = max(players, key=lambda x: x["goal_prob"])
                top_passer = max(players, key=lambda x: x["assist_prob"])
    except Exception:
        pass
        
    return {"all": players, "top_scorer": top_scorer, "top_passer": top_passer}

def calculate_poisson_matrix(home_xg, away_xg):
    max_goals = 10
    matrix = np.zeros((max_goals, max_goals))
    for h in range(max_goals):
        for a in range(max_goals):
            matrix[h, a] = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
    return matrix

# ---------------------------------------------------------
# ROUTES API DE L'APPLICATION
# ---------------------------------------------------------

@app.get("/")
def read_root(response_class=HTMLResponse):
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "index.html")
    
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
            
    return {"status":
