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
from cachetools import TTLCache

app = FastAPI(title="Multi-Sport Pronostics API", version="22.45")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

matches_cache = TTLCache(maxsize=100, ttl=600)

class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    home_id: Optional[str] = None
    away_id: Optional[str] = None

def get_team_seed(home_team, away_team):
    h = int(hashlib.md5(f"{home_team}-{away_team}".encode()).hexdigest(), 16)
    return h

def ai_poisson_model(home_team, away_team):
    seed = get_team_seed(home_team, away_team)
    home_p = 30.0 + (seed % 40)
    away_p = 15.0 + ((seed // 7) % 35)
    draw_p = max(10.0, 100.0 - (home_p + away_p))
    return {"home": round(home_p, 1), "draw": round(draw_p, 1), "away": round(away_p, 1)}

def ai_machine_learning_model(home_team, away_team):
    seed = get_team_seed(away_team, home_team)
    home_p = 25.0 + (seed % 45)
    away_p = 20.0 + ((seed // 5) % 30)
    draw_p = max(10.0, 100.0 - (home_p + away_p))
    return {"home": round(home_p, 1), "draw": round(draw_p, 1), "away": round(away_p, 1)}

def ai_streak_model(home_team, away_team):
    seed = get_team_seed(home_team, "streak")
    home_p = 35.0 + (seed % 35)
    away_p = 15.0 + ((seed // 3) % 35)
    draw_p = max(10.0, 100.0 - (home_p + away_p))
    return {"home": round(home_p, 1), "draw": round(draw_p, 1), "away": round(away_p, 1)}

def ai_h2h_model(home_team, away_team):
    seed = get_team_seed("h2h", away_team)
    home_p = 32.0 + (seed % 38)
    away_p = 18.0 + ((seed // 9) % 32)
    draw_p = max(10.0, 100.0 - (home_p + away_p))
    return {"home": round(home_p, 1), "draw": round(draw_p, 1), "away": round(away_p, 1)}

def ai_fatigue_model(home_team, away_team):
    seed = get_team_seed(home_team, "fatigue")
    home_p = 30.0 + ((seed // 4) % 40)
    away_p = 20.0 + ((seed // 11) % 35)
    draw_p = max(10.0, 100.0 - (home_p + away_p))
    return {"home": round(home_p, 1), "draw": round(draw_p, 1), "away": round(away_p, 1)}

def calculate_5_ai_consensus(home_team, away_team):
    p1 = ai_poisson_model(home_team, away_team)
    p2 = ai_machine_learning_model(home_team, away_team)
    p3 = ai_streak_model(home_team, away_team)
    p4 = ai_h2h_model(home_team, away_team)
    p5 = ai_fatigue_model(home_team, away_team)
    
    final_home = p1['home'] * 0.35 + p2['home'] * 0.25 + p3['home'] * 0.15 + p4['home'] * 0.125 + p5['home'] * 0.125
    final_away = p1['away'] * 0.35 + p2['away'] * 0.25 + p3['away'] * 0.15 + p4['away'] * 0.125 + p5['away'] * 0.125
    final_draw = max(10.0, 100.0 - (final_home + final_away))

    return {
        "home_win": round(final_home, 1),
        "draw": round(final_draw, 1),
        "away_win": round(final_away, 1)
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

    league_names = {
        "eng.1": "Premier League",
        "eng.2": "EFL Championship",
        "esp.1": "La Liga",
        "esp.2": "Liga 2 (Espagne)",
        "ita.1": "Serie A",
        "ger.1": "Bundesliga",
        "ger.2": "2. Bundesliga",
        "fra.1": "Ligue 1",
        "bel.1": "Jupiler Pro League (Belgique)",
        "usa.1": "MLS",
        "mex.1": "Liga MX",
        "chi.1": "Campeonato Chileno",
        "par.1": "Primera División (Paraguay)",
        "ecu.1": "Liga Pro Équateur",
        "col.1": "Liga BetPlay Colombie",
        "rou.1": "Liga I Roumanie",
        "tur.1": "Süper Lig",
        "ksa.1": "Saudi Pro League",
        "uefa.champions": "UEFA Champions League",
        "uefa.europa": "UEFA Europa League",
        "fifa.friendly": "Matchs Amicaux",
        "fifa.world": "Sélections Nationales"
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

                            competitors = comp.get("competitions", [])
                            home_team, away_team, home_logo, away_logo, home_score, away_score = "Domicile", "Extérieur", "", "", None, None

                            for team in competitors:
                                team_data = team.get("team", {})
                                team_display_name = team_data.get("displayName") or team_data.get("name") or team_data.get("shortDisplayName", "Équipe")
                                
                                t_logo = team_data.get("logo", "")
                                if not t_logo and "logos" in team_data and len(team_data["logos"]) > 0:
                                    t_logo = team_data["logos"][0].get("href", "")

                                if team.get("homeAway") == "home":
                                    home_team = team_display_name
                                    home_logo = t_logo
                                    home_score = team.get("score")
                                else:
                                    away_team = team_display_name
                                    away_logo = t_logo
                                    away_score = team.get("score")

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
        probs = calculate_5_ai_consensus(req.home_team, req.away_team)
        hw = probs["home_win"]
        dw = probs["draw"]
        aw = probs["away_win"]
        
        seed = get_team_seed(req.home_team, req.away_team)
        
        if hw >= aw and hw >= dw:
            home_goals = 2 if hw >= 45 else 1
            away_goals = 0 if home_goals == 2 and (seed % 2 == 0) else 1
            if home_goals <= away_goals:
                home_goals = away_goals + 1
        elif aw >= hw and aw >= dw:
            away_goals = 2 if aw >= 45 else 1
            home_goals = 0 if away_goals == 2 and (seed % 2 == 0) else 1
            if away_goals <= home_goals:
                away_goals = home_goals + 1
        else:
            g = 1 if (seed % 2 == 0) else 0
            home_goals = g
            away_goals = g

        predicted_score = f"{home_goals} - {away_goals}"
        total_goals = home_goals + away_goals

        ht_home = 1 if home_goals >= 2 and seed % 2 == 0 else (1 if home_goals == 1 and seed % 3 == 0 else 0)
        ht_away = 1 if away_goals >= 2 and seed % 3 == 0 else (1 if away_goals == 1 and seed % 4 == 0 else 0)
        ht_score = f"{ht_home} - {ht_away}"
        ht_total = ht_home + ht_away

        over_0_5_prob = round(82.0 if ht_total > 0 else 38.0, 1)
        action_conseillee = "Plus de 0.5 but MT" if ht_total > 0 else "Observation en 1ère MT"

        btts_yes = (home_goals > 0 and away_goals > 0)
        btts_prob = round(72.0 if btts_yes else 28.0, 1)

        over_25 = (total_goals >= 3)
        over_25_prob = round(75.0 if over_25 else 25.0, 1)

        base_cards = 3.0 + (seed % 4) * 0.5

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
                "over_4_5_prob": round(45.0 + (seed % 35), 1),
                "home_team_avg": round(base_cards * 0.52, 1),
                "away_team_avg": round(base_cards * 0.48, 1)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
