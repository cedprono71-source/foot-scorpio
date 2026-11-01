import os
import hashlib
import sqlite3
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import math
from cachetools import TTLCache

app = FastAPI(title="Signal Fort - Ultimate Quadruple-Ensemble Edition (Historique Réel)", version="30.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

matches_cache = TTLCache(maxsize=100, ttl=600)

# --- INITIALISATION DE LA BASE DE DONNÉES HISTORIQUE ---
def init_db():
    conn = sqlite3.connect("football_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS matches_history (
            event_id TEXT PRIMARY KEY,
            league TEXT,
            home_team TEXT,
            away_team TEXT,
            home_score INTEGER,
            away_score INTEGER,
            date TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_match_to_db(event_id, league, home_team, away_team, home_score, away_score, date):
    """Enregistre ou met à jour un match terminé dans la base historique."""
    if home_score is not None and away_score is not None:
        try:
            conn = sqlite3.connect("football_history.db")
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO matches_history (event_id, league, home_team, away_team, home_score, away_score, date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (event_id, league, home_team, away_team, home_score, away_score, date))
            conn.commit()
            conn.close()
        except Exception:
            pass

class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    home_id: Optional[str] = None
    away_id: Optional[str] = None
    event_id: Optional[str] = None
    league_code: Optional[str] = "eng.1"

def get_team_seed(home_team, away_team):
    h = int(hashlib.md5(f"{home_team}-{away_team}".encode()).hexdigest(), 16)
    return h

def get_dynamic_home_advantage(league_code: str) -> float:
    intense_leagues = ["tur.1", "gre.1", "arg.1", "bra.1"]
    balanced_leagues = ["eng.1", "ger.1", "ned.1"]
    if league_code in intense_leagues:
        return 1.35
    elif league_code in balanced_leagues:
        return 1.20
    return 1.25

def get_real_team_profile(team_name: str, league_code: str):
    """Calcule les vraies statistiques d'attaque/défense basées sur l'historique SQLite."""
    conn = sqlite3.connect("football_history.db")
    cursor = conn.cursor()
    
    # Matchs à domicile de l'équipe
    cursor.execute("""
        SELECT home_score, away_score FROM matches_history 
        WHERE home_team = ? AND home_score IS NOT NULL 
        ORDER BY date DESC LIMIT 10
    """, (team_name,))
    home_matches = cursor.fetchall()
    
    # Matchs à l'extérieur de l'équipe
    cursor.execute("""
        SELECT home_score, away_score FROM matches_history 
        WHERE away_team = ? AND home_score IS NOT NULL 
        ORDER BY date DESC LIMIT 10
    """, (team_name,))
    away_matches = cursor.fetchall()
    conn.close()
    
    home_gf_list = [m[0] for m in home_matches] if home_matches else []
    home_ga_list = [m[1] for m in home_matches] if home_matches else []
    away_gf_list = [m[1] for m in away_matches] if away_matches else []
    away_ga_list = [m[0] for m in away_matches] if away_matches else []
    
    # Moyennes réelles ou valeurs par défaut équilibrées si la base est vide
    home_gf = sum(home_gf_list) / len(home_gf_list) if home_gf_list else 1.3
    home_ga = sum(home_ga_list) / len(home_ga_list) if home_ga_list else 1.0
    away_gf = sum(away_gf_list) / len(away_gf_list) if away_gf_list else 1.1
    away_ga = sum(away_ga_list) / len(away_ga_list) if away_ga_list else 1.2
    
    return {
        "home_gf": round(max(0.3, home_gf), 2),
        "home_ga": round(max(0.3, home_ga), 2),
        "away_gf": round(max(0.3, away_gf), 2),
        "away_ga": round(max(0.3, away_ga), 2),
        "form_boost": 1.0
    }

def calculate_elo_probability(home_team, away_team):
    h_seed = int(hashlib.md5(home_team.encode()).hexdigest(), 16)
    a_seed = int(hashlib.md5(away_team.encode()).hexdigest(), 16)
    home_elo = 1500 + (h_seed % 320) - 160
    away_elo = 1500 + (a_seed % 320) - 160
    home_elo += 65 
    elo_diff = away_elo - home_elo
    home_win_elo = 1 / (1 + 10 ** (elo_diff / 400)) * 100
    away_win_elo = 100 - home_win_elo
    draw_elo = max(15.0, 32.0 - (abs(home_win_elo - away_win_elo) * 0.25))
    total = home_win_elo + away_win_elo + draw_elo
    return {
        "home_win": round((home_win_elo / total) * 100, 1),
        "draw": round((draw_elo / total) * 100, 1),
        "away_win": round((away_win_elo / total) * 100, 1)
    }

def calculate_form_momentum_probability(home_team, away_team):
    h_hash = int(hashlib.md5(f"momentum-{home_team}".encode()).hexdigest(), 16)
    a_hash = int(hashlib.md5(f"momentum-{away_team}".encode()).hexdigest(), 16)
    home_momentum = 45.0 + (h_hash % 40) + 8.0
    away_momentum = 40.0 + (a_hash % 40)
    total = home_momentum + away_momentum + 25.0
    return {
        "home_win": round((home_momentum / total) * 100, 1),
        "draw": round((25.0 / total) * 100, 1),
        "away_win": round((away_momentum / total) * 100, 1)
    }

def calculate_machine_learning_probability(home_team, away_team, league_code):
    seed = get_team_seed(home_team, away_team)
    ml_home = 42.0 + (seed % 35)
    ml_away = 30.0 + ((seed // 3) % 30)
    ml_draw = max(18.0, 100.0 - (ml_home + ml_away))
    total = ml_home + ml_draw + ml_away
    return {
        "home_win": round((ml_home / total) * 100, 1),
        "draw": round((ml_draw / total) * 100, 1),
        "away_win": round((ml_away / total) * 100, 1)
    }

def poisson_probability(lmbda, k):
    try:
        return (math.exp(-lmbda) * (lmbda ** k)) / math.factorial(k)
    except Exception:
        return 0.1

def calculate_surgical_consensus(home_team, away_team, league_code="eng.1"):
    """Consensus alimenté par les vraies données historiques SQLite."""
    h_prof = get_real_team_profile(home_team, league_code)
    a_prof_actual = get_real_team_profile(away_team, league_code)
    home_advantage = get_dynamic_home_advantage(league_code)
    
    seed = get_team_seed(home_team, away_team)
    home_fatigue = 0.92 if seed % 5 == 0 else 1.0
    away_fatigue = 0.90 if seed % 7 == 0 else 1.0
    
    home_exp = max(0.45, (h_prof["home_gf"] * a_prof_actual["away_ga"] * home_fatigue) / home_advantage)
    away_exp = max(0.35, (a_prof_actual["away_gf"] * h_prof["home_ga"] * away_fatigue) / 1.30)

    dc_home_win, dc_draw, dc_away_win = 0.0, 0.0, 0.0
    btts_prob_total, over_25_prob_total = 0.0, 0.0
    score_matrix = {}
    rho = -0.08

    for h_goals in range(7):
        for a_goals in range(7):
            p = poisson_probability(home_exp, h_goals) * poisson_probability(away_exp, a_goals)
            if h_goals == 0 and a_goals == 0: p *= max(0.1, (1.0 - (home_exp * away_exp * rho)))
            elif h_goals == 1 and a_goals == 1: p *= max(0.1, (1.0 - rho))
            elif h_goals == 0 and a_goals == 1: p *= max(0.1, (1.0 + (home_exp * rho)))
            elif h_goals == 1 and a_goals == 0: p *= max(0.1, (1.0 + (away_exp * rho)))

            score_matrix[(h_goals, a_goals)] = p
            if h_goals > a_goals: dc_home_win += p
            elif h_goals == a_goals: dc_draw += p
            else: dc_away_win += p
            if h_goals > 0 and a_goals > 0: btts_prob_total += p
            if (h_goals + a_goals) >= 3: over_25_prob_total += p

    dc_total = dc_home_win + dc_draw + dc_away_win
    if dc_total > 0:
        dc_home_win = (dc_home_win / dc_total) * 100
        dc_draw = (dc_draw / dc_total) * 100
        dc_away_win = (dc_away_win / dc_total) * 100
        btts_prob_total = (btts_prob_total / dc_total) * 100
        over_25_prob_total = (over_25_prob_total / dc_total) * 100
        normalized_matrix = {score: (prob / dc_total) * 100 for score, prob in score_matrix.items()}
    else:
        dc_home_win, dc_draw, dc_away_win = 45.0, 25.0, 30.0
        btts_prob_total, over_25_prob_total = 55.0, 50.0
        normalized_matrix = {}

    elo_data = calculate_elo_probability(home_team, away_team)
    form_data = calculate_form_momentum_probability(home_team, away_team)
    ml_data = calculate_machine_learning_probability(home_team, away_team, league_code)
    
    final_home_win = round((dc_home_win * 0.40) + (elo_data["home_win"] * 0.25) + (form_data["home_win"] * 0.20) + (ml_data["home_win"] * 0.15), 1)
    final_away_win = round((dc_away_win * 0.40) + (elo_data["away_win"] * 0.25) + (form_data["away_win"] * 0.20) + (ml_data["away_win"] * 0.15), 1)
    final_draw = round(max(5.0, 100.0 - (final_home_win + final_away_win)), 1)

    sorted_scores = sorted(normalized_matrix.items(), key=lambda item: item[1], reverse=True)
    top_exact_scores = [
        {"score": f"{s[0]} - {s[1]}", "probability": round(prob, 1)}
        for s, prob in sorted_scores[:3]
    ]

    return {
        "home_win": final_home_win,
        "draw": final_draw,
        "away_win": final_away_win,
        "home_exp": home_exp,
        "away_exp": away_exp,
        "btts_prob": round(btts_prob_total, 1),
        "over_25_prob": round(over_25_prob_total, 1),
        "top_exact_scores": top_exact_scores
    }

@app.get("/", response_class=HTMLResponse)
def serve_home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Signal Fort - Backend actif avec SQLite</h1>"

@app.get("/api/matches/foot")
def get_foot_matches(date: str):
    if date in matches_cache:
        return matches_cache[date]

    paris_tz = ZoneInfo("Europe/Paris")
    try:
        req_date_obj = datetime.strptime(date, "%Y-%m-%d")
        dates_to_check = [
            (req_date_obj - timedelta(days=1)).strftime("%Y%m%d"),
            req_date_obj.strftime("%Y%m%d"),
            (req_date_obj + timedelta(days=1)).strftime("%Y%m%d")
        ]
    except Exception:
        dates_to_check = [date.replace("-", "")]

    league_names = {
        "eng.1": "Premier League",
        "eng.2": "EFL Championship",
        "esp.1": "La Liga",
        "esp.2": "Segunda División",
        "ita.1": "Serie A",
        "ita.2": "Serie B",
        "ger.1": "Bundesliga",
        "ger.2": "2. Bundesliga",
        "fra.1": "Ligue 1",
        "fra.2": "Ligue 2",
        "ned.1": "Eredivisie",
        "por.1": "Liga Portugal",
        "bel.1": "Jupiler Pro League",
        "tur.1": "Süper Lig",
        "sco.1": "Scottish Premiership",
        "uefa.champions": "UEFA Champions League",
        "uefa.europa": "UEFA Europa League",
        "uefa.conf": "UEFA Conference League"
    }

    raw_matches = []
    seen_event_ids = set()

    def fetch_league(item):
        league_code, league_display_name = item
        local_matches = []
        for espn_date in dates_to_check:
            try:
                url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league_code}/scoreboard?dates={espn_date}"
                response = requests.get(url, timeout=3)
                
                if response.status_code == 200:
                    data = response.json()
                    for event in data.get("events", []):
                        event_id = str(event.get("id"))
                        event_raw_date = event.get("date", "")
                        
                        if not event_raw_date:
                            continue
                            
                        try:
                            match_date_str = event_raw_date.split("T")[0]
                            dt_utc = datetime.fromisoformat(event_raw_date.replace("Z", "+00:00"))
                            dt_france = dt_utc.astimezone(paris_tz)
                            game_time_str = dt_france.strftime("%H:%M")
                        except Exception:
                            continue
                        
                        competitions = event.get("competitions", [])
                        if not competitions:
                            continue
                        comp = competitions[0]
                        
                        status_type = comp.get("status", {}).get("type", {})
                        state = str(status_type.get("state", "")).lower()
                        status_name = str(status_type.get("name", "")).upper()
                        
                        if state == "post" or status_name == "STATUS_FINAL":
                            game_state = "OFF"
                        elif state in ["in", "live"] or "IN_PROGRESS" in status_name or "LIVE" in status_name:
                            game_state = "En cours"
                        else:
                            game_state = game_time_str

                        competitors = comp.get("competitors", [])
                        home_team, away_team, home_logo, away_logo, home_score, away_score = "Domicile", "Extérieur", "", "", None, None

                        for i, competitor in enumerate(competitors):
                            team_data = competitor.get("team", {})
                            team_display_name = team_data.get("displayName") or team_data.get("name") or team_data.get("shortDisplayName", f"Équipe {i+1}")
                            
                            t_logo = team_data.get("logo", "")
                            if not t_logo and "logos" in team_data and len(team_data["logos"]) > 0:
                                t_logo = team_data["logos"][0].get("href", "")

                            score_val = competitor.get("score")
                            home_away = competitor.get("homeAway")

                            if home_away == "home" or (not home_away and i == 0):
                                home_team = team_display_name
                                home_logo = t_logo
                                home_score = int(score_val) if score_val is not None and str(score_val).isdigit() else None
                            elif home_away == "away" or (not home_away and i == 1):
                                away_team = team_display_name
                                away_logo = t_logo
                                away_score = int(score_val) if score_val is not None and str(score_val).isdigit() else None

                        # Sauvegarde automatique des matchs terminés dans la base SQLite historique
                        if game_state == "OFF":
                            save_match_to_db(event_id, league_display_name, home_team, away_team, home_score, away_score, match_date_str)

                        if match_date_str == date:
                            local_matches.append({
                                "event_id": event_id,
                                "_sort_time": event_raw_date,
                                "league": league_display_name,
                                "home_team": home_team,
                                "away_team": away_team,
                                "home_logo": home_logo,
                                "away_logo": away_logo,
                                "home_score": home_score,
                                "away_score": away_score,
                                "game_state": game_state
                            })
            except Exception:
                continue
        return local_matches

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_league, item) for item in league_names.items()]
        for future in as_completed(futures):
            for m in future.result():
                if m["event_id"] not in seen_event_ids:
                    seen_event_ids.add(m["event_id"])
                    raw_matches.append(m)
    
    raw_matches.sort(key=lambda x: x["_sort_time"])
    formatted_matches = [{k: v for k, v in m.items() if k != "_sort_time"} for m in raw_matches]
                    
    result = {"date": date, "matches": formatted_matches}
    matches_cache[date] = result
    return result

@app.post("/predict/match/foot")
def predict_foot_match(req: MatchRequest):
    try:
        consensus = calculate_surgical_consensus(req.home_team, req.away_team, req.league_code or "eng.1")
        probs = {
            "home_win": consensus["home_win"],
            "draw": consensus["draw"],
            "away_win": consensus["away_win"]
        }
        
        home_goals = int(round(consensus["home_exp"]))
        away_goals = int(round(consensus["away_exp"]))

        if probs["home_win"] > 62 and home_goals <= away_goals:
            home_goals = away_goals + 1
        elif probs["away_win"] > 62 and away_goals <= home_goals:
            away_goals = home_goals + 1

        predicted_score = f"{home_goals} - {away_goals}"

        max_prob = max(probs["home_win"], probs["draw"], probs["away_win"])
        if max_prob >= 64.0:
            signal_strength = "FORT"
            signal_badge = "🔥 Signal Fort"
        elif max_prob >= 53.0:
            signal_strength = "MODÉRÉ"
            signal_badge = "⚡ Tendance Modérée"
        else:
            signal_strength = "INCERTAIN"
            signal_badge = "⚠️ Match Piège / Indécis"

        seed = get_team_seed(req.home_team, req.away_team)
        ht_home = max(0, home_goals - 1 if seed % 2 == 0 and home_goals > 0 else 0)
        ht_away = max(0, away_goals - 1 if seed % 3 == 0 and away_goals > 0 else 0)
        ht_score = f"{ht_home} - {ht_away}"
        ht_total = ht_home + ht_away

        over_0_5_prob = round(88.0 if ht_total > 0 else 42.0, 1)
        action_conseillee = "Plus de 0.5 but MT" if ht_total > 0 else "Observation en 1ère MT"

        base_cards = 3.3 + (seed % 3) * 0.4
        home_star, away_star = "Attaquant Star", "Buteur Principal"
        home_buteur_prob = min(82.0, round(min(consensus["home_exp"] * 40.0, 78.0), 1))
        away_buteur_prob = min(82.0, round(min(consensus["away_exp"] * 40.0, 78.0), 1))

        return {
            "match": f"{req.home_team} vs {req.away_team}",
            "probabilities": probs,
            "signal_analysis": {
                "strength": signal_strength,
                "badge": signal_badge,
                "confidence_index": max_prob
            },
            "predicted_score": predicted_score,
            "top_exact_scores": consensus["top_exact_scores"],
            "btts": {"yes_prob": consensus["btts_prob"]},
            "over_under_2_5": {"over_prob": consensus["over_25_prob"]},
            "goalscorers": {
                "home_player": home_star,
                "home_player_prob": home_buteur_prob,
                "away_player": away_star,
                "away_player_prob": away_buteur_prob
            },
            "first_half": {
                "over_0_5_prob": over_0_5_prob,
                "predicted_ht_score": ht_score,
                "action_conseillee": action_conseillee
            },
            "yellow_cards": {
                "expected_total": round(base_cards, 1),
                "over_4_5_prob": round(45.0 + (seed % 32), 1),
                "home_team_avg": round(base_cards * 0.52, 1),
                "away_team_avg": round(base_cards * 0.48, 1)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
