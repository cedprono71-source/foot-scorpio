from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
from scipy.stats import poisson
import requests
import os
import json
from datetime import datetime, timedelta

app = FastAPI(title="Multi-IA NHL Pronostics")

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
        
    MANUAL_INJURED_PLAYERS = ["travis konecny"] 
    
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
                
                if name.lower() in MANUAL_INJURED_PLAYERS:
                    continue
                
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
                
                players.append({
                    "name": name,
                    "pos": position,
                    "goals": goals,
                    "assists": assists,
                    "points": points,
                    "goal_prob": goal_prob,
                    "assist_prob": assist_prob,
                    "point_prob": point_prob
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

@app.post("/predict/match")
def predict_match(match: MatchInput):
    team_stats = fetch_team_stats()
    
    home_abbrev, away_abbrev = "", ""
    home_st = {"offense": 3.1, "defense": 2.7, "pp_pct": 0.20, "pk_pct": 0.80}
    away_st = {"offense": 2.9, "defense": 2.9, "pp_pct": 0.18, "pk_pct": 0.78}
    
    for name, st in team_stats.items():
        if name in match.home_team.lower() or match.home_team.lower() in name:
            home_st = st
            home_abbrev = st.get("abbrev", "")
        if name in match.away_team.lower() or match.away_team.lower() in name:
            away_st = st
            away_abbrev = st.get("abbrev", "")

    home_goalie = fetch_goalie_stats(home_abbrev)
    away_goalie = fetch_goalie_stats(away_abbrev)

    home_injuries_data = fetch_team_roster_and_injuries(home_abbrev)
    away_injuries_data = fetch_team_roster_and_injuries(away_abbrev)

    mat1 = calculate_poisson_matrix(3.1, 2.7)
    p1_home = float(np.sum(np.tril(mat1, -1))) * 100

    h_xg2 = home_st["offense"] * 1.08 * home_injuries_data["malus"]
    a_xg2 = away_st["offense"] * 0.92 * away_injuries_data["malus"]
    mat2 = calculate_poisson_matrix(h_xg2, a_xg2)
    p2_home = float(np.sum(np.tril(mat2, -1))) * 100

    h_xg3 = max(1.5, (home_st["offense"] - (away_st["defense"] * 0.1) + 0.2) * home_injuries_data["malus"])
    a_xg3 = max(1.5, (away_st["offense"] - (home_st["defense"] * 0.1)) * away_injuries_data["malus"])
    mat3 = calculate_poisson_matrix(h_xg3, a_xg3)
    p3_home = float(np.sum(np.tril(mat3, -1))) * 100

    h_xg4 = max(1.2, home_st["offense"] * (1 - (away_goalie["save_pct"] - 0.900) * 2))
    a_xg4 = max(1.2, away_st["offense"] * (1 - (home_goalie["save_pct"] - 0.900) * 2))
    mat4 = calculate_poisson_matrix(h_xg4, a_xg4)
    p4_home = float(np.sum(np.tril(mat4, -1))) * 100

    pp_advantage_home = home_st["pp_pct"] * (1 - away_st["pk_pct"]) * 10.0
    pp_advantage_away = away_st["pp_pct"] * (1 - home_st["pk_pct"]) * 10.0
    h_xg5 = max(1.2, h_xg2 + pp_advantage_home)
    a_xg5 = max(1.2, a_xg2 + pp_advantage_away)
    mat5 = calculate_poisson_matrix(h_xg5, a_xg5)
    p5_home = float(np.sum(np.tril(mat5, -1))) * 100

    home_fatigue = 0.88 if match.home_b2b else 1.0
    away_fatigue = 0.88 if match.away_b2b else 0.95
    h_xg6 = max(1.2, h_xg2 * home_fatigue)
    a_xg6 = max(1.2, a_xg2 * away_fatigue)
    mat6 = calculate_poisson_matrix(h_xg6, a_xg6)
    p6_home = float(np.sum(np.tril(mat6, -1))) * 100

    final_home_win = round((p1_home + p2_home + p3_home + p4_home + p5_home + p6_home) / 6, 1)
    final_away_win = round(max(5.0, 100 - final_home_win - 15.0), 1)
    final_draw = round(15.0, 1)

    final_h_xg = (h_xg2 + h_xg4 + h_xg5 + h_xg6) / 4
    final_a_xg = (a_xg2 + a_xg4 + a_xg5 + a_xg6) / 4
    mat_final = calculate_poisson_matrix(final_h_xg, final_a_xg)

    best_score = np.unravel_index(np.argmax(mat_final), mat_final.shape)

    over_5_5_prob = 0.0
    for h in range(10):
        for a in range(10):
            if h + a > 5.5:
                over_5_5_prob += mat_final[h, a]

    home_players_data = fetch_all_players_projections(home_abbrev, final_h_xg)
    away_players_data = fetch_all_players_projections(away_abbrev, final_a_xg)

    predicted_winner = match.home_team if final_home_win >= final_away_win else match.away_team

    history = load_history()
    match_key = f"{match.home_team}_vs_{match.away_team}_{datetime.now().strftime('%Y-%m-%d')}"
    if match_key not in history:
        history[match_key] = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "match": f"{match.home_team} vs {match.away_team}",
            "home_team": match.home_team,
            "away_team": match.away_team,
            "predicted_winner": predicted_winner,
            "predicted_score": f"{int(best_score[0])} - {int(best_score[1])}",
            "predicted_over": round(over_5_5_prob * 100, 1) > 50.0,
            "actual_score": None,
            "winner_correct": None,
            "over_correct": None,
            "status": "PENDING"
        }
        save_history(history)

    return {
        "match": f"{match.home_team} vs {match.away_team}",
        "probabilities": {
            "home_win": final_home_win,
            "draw": final_draw,
            "away_win": final_away_win
        },
        "predicted_score": f"{int(best_score[0])} - {int(best_score[1])}",
        "over_under_5_5": {
            "over_prob": round(over_5_5_prob * 100, 1),
            "under_prob": round((1 - over_5_5_prob) * 100, 1)
        },
        "goalies": {
            "home": home_goalie,
            "away": away_goalie
        },
        "special_teams": {
            "home_pp": round(home_st["pp_pct"] * 100, 1),
            "away_pp": round(away_st["pp_pct"] * 100, 1)
        },
        "injuries": {
            "home": home_injuries_data["injuries"],
            "away": away_injuries_data["injuries"]
        },
        "players": {
            "home": home_players_data,
            "away": away_players_data
        }
    }

@app.get("/api/history")
def get_prediction_history():
    history = load_history()
    updated = False
    for key, item in history.items():
        if item.get("status") == "PENDING":
            try:
                m_date = item.get("date")
                url = f"https://api-web.nhle.com/v1/schedule/{m_date}"
                res = requests.get(url)
                if res.status_code == 200:
                    data = res.json()
                    for day in data.get("gameWeek", []):
                        if day.get("date") == m_date:
                            for g in day.get("games", []):
                                h_name = g.get("homeTeam", {}).get("placeName", {}).get("default") or g.get("homeTeam", {}).get("commonName", {}).get("default", "")
                                a_name = g.get("awayTeam", {}).get("placeName", {}).get("default") or g.get("awayTeam", {}).get("commonName", {}).get("default", "")
                                
                                if (h_name.lower() in item["home_team"].lower() or item["home_team"].lower() in h_name.lower()) and \
                                   (a_name.lower() in item["away_team"].lower() or item["away_team"].lower() in a_name.lower()):
                                    
                                    gameState = g.get("gameState")
                                    if gameState in ["OFF", "FINAL"]:
                                        h_score = g.get("homeTeam", {}).get("score", 0)
                                        a_score = g.get("awayTeam", {}).get("score", 0)
                                        actual_winner = item["home_team"] if h_score > a_score else item["away_team"]
                                        actual_over = (h_score + a_score) > 5.5
                                        
                                        item["actual_score"] = f"{h_score} - {a_score}"
                                        item["winner_correct"] = (actual_winner == item["predicted_winner"])
                                        item["over_correct"] = (actual_over == item["predicted_over"])
                                        item["status"] = "FINISHED"
                                        updated = True
            except Exception:
                pass

    if updated:
        save_history(history)

    total = 0
    winner_wins = 0
    over_wins = 0
    
    items_list = list(history.values())
    for item in items_list:
        if item.get("status") == "FINISHED":
            total += 1
            if item.get("winner_correct"):
                winner_wins += 1
            if item.get("over_correct"):
                over_wins += 1

    winrate_winner = round((winner_wins / total * 100), 1) if total > 0 else 0.0
    winrate_over = round((over_wins / total * 100), 1) if total > 0 else 0.0

    return {
        "stats": {
            "total_matches": total,
            "winrate_winner": winrate_winner,
            "winrate_over": winrate_over
        },
        "history": sorted(items_list, key=lambda x: x["date"], reverse=True)
    }

@app.get("/api/matches/nhl")
def get_nhl_matches(date: str = "now"):
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d") if date != "now" else datetime.now()
        start_date = (target_date - timedelta(days=1)).strftime("%Y-%m-%d")
        
        url = f"https://api-web.nhle.com/v1/schedule/{start_date}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            matches = []
            for day in data.get("gameWeek", []):
                game_date = day.get("date")
                for g in day.get("games", []):
                    home_obj = g.get("homeTeam", {})
                    away_obj = g.get("awayTeam", {})
                    
                    home_name = home_obj.get("placeName", {}).get("default") or home_obj.get("commonName", {}).get("default", "Domicile")
                    away_name = away_obj.get("placeName", {}).get("default") or away_obj.get("commonName", {}).get("default", "Extérieur")
                    
                    matches.append({
                        "id": g.get("id"),
                        "date": game_date,
                        "home_team": home_name,
                        "away_team": away_name,
                        "home_logo": home_obj.get("logo", ""),
                        "away_logo": away_obj.get("logo", ""),
                        "time": g.get("startTimeUTC")
                    })
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
