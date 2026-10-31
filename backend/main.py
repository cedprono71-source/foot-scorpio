import os
import hashlib
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

app = FastAPI(title="Multi-Sport Pronostics API - Europe Max", version="22.54")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

matches_cache = TTLCache(maxsize=100, ttl=600)
real_team_stats = {}

class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    home_id: Optional[str] = None
    away_id: Optional[str] = None

def get_team_seed(home_team, away_team):
    h = int(hashlib.md5(f"{home_team}-{away_team}".encode()).hexdigest(), 16)
    return h

def poisson_probability(lmbda, k):
    """Calcule la probabilité exacte d'un nombre de buts selon la loi de Poisson."""
    try:
        return (math.exp(-lmbda) * (lmbda ** k)) / math.factorial(k)
    except Exception:
        return 0.1

def calculate_real_poisson_consensus(home_team, away_team):
    """Calcule les probabilités de Poisson optimisées pour les standards européens."""
    h_stats = real_team_stats.get(home_team, {"gf": 1.4, "ga": 1.1})
    a_stats = real_team_stats.get(away_team, {"gf": 1.1, "ga": 1.4})
    
    # Standard européen : léger avantage domicile tactique
    home_exp = max(0.4, ((h_stats["gf"] + a_stats["ga"]) / 2.0) * 1.15)
    away_exp = max(0.3, ((a_stats["gf"] + h_stats["ga"]) / 2.0) * 0.90)

    home_win_prob = 0.0
    draw_prob = 0.0
    away_win_prob = 0.0

    # Matrice de Poisson (0 à 6 buts)
    for h_goals in range(7):
        for a_goals in range(7):
            p = poisson_probability(home_exp, h_goals) * poisson_probability(away_exp, a_goals)
            if h_goals > a_goals:
                home_win_prob += p
            elif h_goals == a_goals:
                draw_prob += p
            else:
                away_win_prob += p

    total = home_win_prob + draw_prob + away_win_prob
    if total > 0:
        home_win_prob = (home_win_prob / total) * 100
        draw_prob = (draw_prob / total) * 100
        away_win_prob = (away_win_prob / total) * 100
    else:
        home_win_prob, draw_prob, away_win_prob = 45.0, 25.0, 30.0

    return {
        "home_win": round(home_win_prob, 1),
        "draw": round(draw_prob, 1),
        "away_win": round(away_win_prob, 1),
        "home_exp": home_exp,
        "away_exp": away_exp
    }

@app.get("/", response_class=HTMLResponse)
def serve_home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h3>Erreur : Le fichier index.html est introuvable sur le serveur.</h3>"

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

    # Catalogue maximum des championnats et coupes en Europe
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
                        
                        if match_date_str == date:
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

                                gf, ga = 1.3, 1.2
                                records = competitor.get("records", [])
                                for rec in records:
                                    stats_summary = rec.get("stats", [])
                                    for s in stats_summary:
                                        if s.get("name") == "pointsFor" or s.get("name") == "goalsFor":
                                            gf = float(s.get("value", gf))
                                        elif s.get("name") == "pointsAgainst" or s.get("name") == "goalsAgainst":
                                            ga = float(s.get("value", ga))

                                real_team_stats[team_display_name] = {
                                    "gf": max(0.5, gf),
                                    "ga": max(0.5, ga)
                                }

                                if home_away == "home" or (not home_away and i == 0):
                                    home_team = team_display_name
                                    home_logo = t_logo
                                    home_score = score_val
                                elif home_away == "away" or (not home_away and i == 1):
                                    away_team = team_display_name
                                    away_logo = t_logo
                                    away_score = score_val

                            local_matches.append({
                                "event_id": event_id,
                                "_sort_time": event_raw_date,
                                "league": league_display_name,
                                "home_team": home_team,
                                "away_team": away_team,
                                "home_logo": home_logo,
                                "away_logo": away_logo,
                                "home_score": int(home_score) if home_score is not None and str(home_score).isdigit() else None,
                                "away_score": int(away_score) if away_score is not None and str(away_score).isdigit() else None,
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
                    raw_matches.append({k: v for k, v in m.items() if k != "event_id"})
    
    raw_matches.sort(key=lambda x: x["_sort_time"])
    formatted_matches = [{k: v for k, v in m.items() if k != "_sort_time"} for m in raw_matches]
                    
    result = {"date": date, "matches": formatted_matches}
    matches_cache[date] = result
    return result

@app.post("/predict/match/foot")
def predict_foot_match(req: MatchRequest):
    try:
        consensus = calculate_real_poisson_consensus(req.home_team, req.away_team)
        probs = {
            "home_win": consensus["home_win"],
            "draw": consensus["draw"],
            "away_win": consensus["away_win"]
        }
        
        home_goals = int(round(consensus["home_exp"]))
        away_goals = int(round(consensus["away_exp"]))

        if probs["home_win"] > 60 and home_goals <= away_goals:
            home_goals = away_goals + 1
        elif probs["away_win"] > 60 and away_goals <= home_goals:
            away_goals = home_goals + 1

        predicted_score = f"{home_goals} - {away_goals}"
        total_goals = home_goals + away_goals

        seed = get_team_seed(req.home_team, req.away_team)
        ht_home = max(0, home_goals - 1 if seed % 2 == 0 and home_goals > 0 else 0)
        ht_away = max(0, away_goals - 1 if seed % 3 == 0 and away_goals > 0 else 0)
        ht_score = f"{ht_home} - {ht_away}"
        ht_total = ht_home + ht_away

        over_0_5_prob = round(86.0 if ht_total > 0 else 40.0, 1)
        action_conseillee = "Plus de 0.5 but MT" if ht_total > 0 else "Observation en 1ère MT"

        btts_yes = (home_goals > 0 and away_goals > 0)
        btts_prob = round(74.0 if btts_yes else 28.0, 1)

        over_25 = (total_goals >= 3)
        over_25_prob = round(76.0 if over_25 else 24.0, 1)

        base_cards = 3.2 + (seed % 3) * 0.4

        return {
            "match": f"{req.home_team} vs {req.away_team}",
            "probabilities": probs,
            "predicted_score": predicted_score,
            "btts": {"yes_prob": btts_prob},
            "over_under_2_5": {"over_prob": over_25_prob},
            "first_half": {
                "over_0_5_prob": over_0_5_prob,
                "predicted_ht_score": ht_score,
                "action_conseillee": action_conseillee
            },
            "yellow_cards": {
                "expected_total": round(base_cards, 1),
                "over_4_5_prob": round(46.0 + (seed % 30), 1),
                "home_team_avg": round(base_cards * 0.52, 1),
                "away_team_avg": round(base_cards * 0.48, 1)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
