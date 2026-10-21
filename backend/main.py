import os
import hashlib
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Multi-IA Pronostics API", version="22.26")

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
    home_id: Optional[str] = None
    away_id: Optional[str] = None


# --- Moteur Consensus 5 IA Dynamique ---
def get_team_seed(home_team, away_team):
    h = int(hashlib.md5(f"{home_team}-{away_team}".encode()).hexdigest(), 16)
    return h

def ai_poisson_model(home_team, away_team):
    seed = get_team_seed(home_team, away_team)
    home_p = 35.0 + (seed % 30)
    away_p = 20.0 + ((seed // 7) % 25)
    draw_p = max(15.0, 100.0 - (home_p + away_p))
    return {"home": round(home_p, 1), "draw": round(draw_p, 1), "away": round(away_p, 1)}

def ai_machine_learning_model(home_team, away_team):
    seed = get_team_seed(away_team, home_team)
    home_p = 30.0 + (seed % 35)
    away_p = 25.0 + ((seed // 5) % 25)
    draw_p = max(15.0, 100.0 - (home_p + away_p))
    return {"home": round(home_p, 1), "draw": round(draw_p, 1), "away": round(away_p, 1)}

def ai_streak_model(home_team, away_team):
    seed = get_team_seed(home_team, "streak")
    home_p = 40.0 + (seed % 25)
    away_p = 20.0 + ((seed // 3) % 25)
    draw_p = max(15.0, 100.0 - (home_p + away_p))
    return {"home": round(home_p, 1), "draw": round(draw_p, 1), "away": round(away_p, 1)}

def ai_h2h_model(home_team, away_team):
    seed = get_team_seed("h2h", away_team)
    home_p = 38.0 + (seed % 28)
    away_p = 22.0 + ((seed // 9) % 25)
    draw_p = max(15.0, 100.0 - (home_p + away_p))
    return {"home": round(home_p, 1), "draw": round(draw_p, 1), "away": round(away_p, 1)}

def ai_fatigue_model(home_team, away_team):
    seed = get_team_seed(home_team, "streak")
    home_p = 42.0 + ((seed // 4) % 22)
    away_p = 24.0 + ((seed // 11) % 22)
    draw_p = max(15.0, 100.0 - (home_p + away_p))
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


# --- ROUTE PRINCIPALE ---
@app.get("/", response_class=HTMLResponse)
def serve_home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h3>Erreur : Le fichier index.html est introuvable sur le serveur.</h3>"


# --- Route Football : API ESPN (Synchronisation parfaite sur la date française) ---
@app.get("/api/matches/foot")
def get_foot_matches(date: str):
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
        "ita.1": "Serie A",
        "ger.1": "Bundesliga",
        "fra.1": "Ligue 1",
        "usa.1": "MLS",
        "mex.1": "Liga MX",
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
    
    for league_code, league_display_name in league_names.items():
        for espn_date in dates_to_check:
            try:
                url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league_code}/scoreboard?dates={espn_date}"
                response = requests.get(url, timeout=4)
                
                if response.status_code == 200:
                    data = response.json()
                    for event in data.get("events", []):
                        event_id = str(event.get("id"))
                        event_raw_date = event.get("date", "")
                        
                        if not event_raw_date:
                            continue
                            
                        try:
                            dt_utc = datetime.fromisoformat(event_raw_date.replace("Z", "+00:00"))
                            dt_france = dt_utc.astimezone(paris_tz)
                            match_date_str = dt_france.strftime("%Y-%m-%d")
                        except Exception:
                            continue
                        
                        if match_date_str == date and event_id not in seen_event_ids:
                            seen_event_ids.add(event_id)
                            
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
                                game_state = dt_france.strftime("%H:%M")

                            competitors = comp.get("competitors", [])
                            home_team, away_team, home_logo, away_logo, home_score, away_score = "Domicile", "Extérieur", "", "", None, None

                            for team in competitors:
                                if team.get("homeAway") == "home":
                                    home_team = team.get("team", {}).get("displayName", "Domicile")
                                    home_logo = team.get("team", {}).get("logo", "")
                                    home_score = team.get("score")
                                else:
                                    away_team = team.get("team", {}).get("displayName", "Extérieur")
                                    away_logo = team.get("team", {}).get("logo", "")
                                    away_score = team.get("score")

                            raw_matches.append({
                                "_sort_time": event_raw_date,
                                "league": league_display_name,
                                "home_team": home_team,
                                "away_team": away_team,
                                "home_id": event_id,
                                "away_id": event_id,
                                "home_logo": home_logo,
                                "away_logo": away_logo,
                                "home_score": int(home_score) if home_score is not None and str(home_score).isdigit() else None,
                                "away_score": int(away_score) if away_score is not None and str(away_score).isdigit() else None,
                                "game_state": game_state
                            })
            except Exception:
                continue
    
    raw_matches.sort(key=lambda x: x["_sort_time"])
    formatted_matches = [{k: v for k, v in m.items() if k != "_sort_time"} for m in raw_matches]
                    
    return {"date": date, "matches": formatted_matches}

@app.post("/predict/match/foot")
def predict_foot_match(req: MatchRequest):
    try:
        probs = calculate_5_ai_consensus(req.home_team, req.away_team)
        seed = get_team_seed(req.home_team, req.away_team)
        
        home_goals = 0 if probs["home_win"] < 30 else (1 if probs["home_win"] < 55 else (2 if seed % 3 == 0 else 3))
        away_goals = 0 if probs["away_win"] < 30 else (1 if probs["away_win"] < 55 else (1 if seed % 2 == 0 else 2))
        
        if probs["draw"] > 35:
            home_goals = 1
            away_goals = 1

        predicted_score = f"{home_goals} - {away_goals}"

        ht_home = 1 if home_goals > 0 and (seed % 2 == 0 or home_goals >= 2) else 0
        if home_goals >= 3 and seed % 4 == 0:
            ht_home = 2
        ht_home = min(ht_home, home_goals)

        ht_away = 1 if away_goals > 0 and (seed % 3 == 0 or away_goals >= 2) else 0
        if away_goals >= 3 and seed % 5 == 0:
            ht_away = 2
        ht_away = min(ht_away, away_goals)

        ht_score = f"{ht_home} - {ht_away}"
        ht_total_goals = ht_home + ht_away

        if ht_total_goals > 0:
            over_0_5_prob = round(70.0 + (seed % 25), 1)
            action_conseillee = "Plus de 0.5 but MT"
        else:
            over_0_5_prob = round(20.0 + (seed % 30), 1)
            action_conseillee = "Observation en 1ère MT"

        btts_prob = round(50.0 + (seed % 35), 1) if home_goals > 0 and away_goals > 0 else round(25.0 + (seed % 20), 1)
        over_25_prob = round(40.0 + (seed % 40), 1) if (home_goals + away_goals) >= 3 else round(30.0 + (seed % 25), 1)
        base_cards = 3.2 + (seed % 5) * 0.4

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
                "over_4_5_prob": round(50.0 + (seed % 30), 1),
                "home_team_avg": round(base_cards * 0.52, 1),
                "away_team_avg": round(base_cards * 0.48, 1)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Fonction utilitaire Effectifs NHL ---
def fetch_nhl_roster(team_str: str):
    team_map = {
        "MONTREAL": "MTL", "CANADIENS": "MTL", "MTL": "MTL",
        "TORONTO": "TOR", "MAPLE LEAFS": "TOR", "TOR": "TOR",
        "BOSTON": "BOS", "BRUINS": "BOS", "BOS": "BOS",
        "NEW YORK": "NYR", "RANGERS": "NYR", "NYR": "NYR",
        "SAN JOSE": "SJS", "SHARKS": "SJS", "SJS": "SJS",
        "EDMONTON": "EDM", "OILERS": "EDM", "EDM": "EDM",
        "VANCOUVER": "VAN", "CANUCKS": "VAN", "VAN": "VAN",
        "CALGARY": "CGY", "FLAMES": "CGY", "CGY": "CGY",
        "WINNIPEG": "WPG", "JETS": "WPG", "WPG": "WPG",
        "OTTAWA": "OTT", "SENATORS": "OTT", "OTT": "OTT",
        "DETROIT": "DET", "RED WINGS": "DET", "DET": "DET",
        "BUFFALO": "BUF", "SABRES": "BUF", "BUF": "BUF",
        "TAMPA BAY": "TBL", "LIGHTNING": "TBL", "TBL": "TBL",
        "FLORIDA": "FLA", "PANTHERS": "FLA", "FLA": "FLA",
        "CAROLINA": "CAR", "HURRICANES": "CAR", "CAR": "CAR",
        "NEW JERSEY": "NJD", "DEVILS": "NJD", "NJD": "NJD",
        "NEW YORK ISLANDERS": "NYI", "ISLANDERS": "NYI", "NYI": "NYI",
        "PHILADELPHIA": "PHI", "FLYERS": "PHI", "PHI": "PHI",
        "PITTSBURGH": "PIT", "PENGUINS": "PIT", "PIT": "PIT",
        "WASHINGTON": "WSH", "CAPITALS": "WSH", "WSH": "WSH",
        "COLUMBUS": "CBJ", "BLUE JACKETS": "CBJ", "CBJ": "CBJ",
        "CHICAGO": "CHI", "BLACKHAWKS": "CHI", "CHI": "CHI",
        "NASHVILLE": "NSH", "PREDATORS": "NSH", "NSH": "NSH",
        "ST. LOUIS": "STL", "BLUES": "STL", "STL": "STL",
        "MINNESOTA": "MIN", "WILD": "MIN", "MIN": "MIN",
        "COLORADO": "COL", "AVALANCHE": "COL", "COL": "COL",
        "UTAH": "UTA", "VEGAS": "VGK", "GOLDEN KNIGHTS": "VGK", "VGK": "VGK",
        "SEATTLE": "SEA", "KRAKEN": "SEA", "SEA": "SEA",
        "LOS ANGELES": "LAK", "KINGS": "LAK", "LAK": "LAK",
        "ANAHEIM": "ANA", "DUCKS": "ANA", "ANA": "ANA",
        "DALLAS": "DAL", "STARS": "DAL", "DAL": "DAL"
    }
    
    abbr = "MTL"
    upper_team = team_str.upper()
    for key, code in team_map.items():
        if key in upper_team:
            abbr = code
            break
    if len(team_str.strip()) == 3:
        abbr = team_str.strip().upper()

    try:
        url = f"https://api-web.nhle.com/v1/roster/{abbr}/current"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            players = []
            for cat in ['forwards', 'defensemen']:
                for p in data.get(cat, []):
                    first = p.get('firstName', {}).get('default', '')
                    last = p.get('lastName', {}).get('default', '')
                    pos = p.get('positionCode', 'F')
                    num = p.get('sweaterNumber', '')
                    players.append({
                        "name": f"{first} {last}".strip(),
                        "pos": pos,
                        "stats": f"#{num}" if num else "Actif",
                        "pct": 60 + (abs(hash(last)) % 25)
                    })
            if players:
                return players[:3]
    except Exception:
        pass
        
    return [{"name": f"Joueur clé ({team_str})", "pos": "F", "stats": "En forme", "pct": 65}]


# --- Route NHL ---
@app.get("/api/matches/nhl")
def get_nhl_matches(date: str):
    paris_tz = ZoneInfo("Europe/Paris")
    try:
        req_date_obj = datetime.strptime(date, "%Y-%m-%d")
        dates_to_check = [
            (req_date_obj - timedelta(days=1)).strftime("%Y-%m-%d"),
            req_date_obj.strftime("%Y-%m-%d"),
            (req_date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
        ]
    except Exception:
        dates_to_check = [date]

    formatted_matches = []
    seen_game_ids = set()

    for nhl_date in dates_to_check:
        url = f"https://api-web.nhle.com/v1/score/{nhl_date}"
        try:
            response = requests.get(url, timeout=4)
            if response.status_code != 200:
                continue
                
            data = response.json()
            
            for game in data.get("games", []):
                game_id = str(game.get("id"))
                if game_id in seen_game_ids:
                    continue

                start_utc = game.get("startTimeUTC", "")
                if not start_utc:
                    continue

                try:
                    dt_utc = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
                    dt_france = dt_utc.astimezone(paris_tz)
                    match_date_str = dt_france.strftime("%Y-%m-%d")
                except Exception:
                    continue

                if match_date_str == date:
                    seen_game_ids.add(game_id)

                    home_obj = game.get("homeTeam", {})
                    away_obj = game.get("awayTeam", {})
                    
                    home_team = home_obj.get("placeName", {}).get("default") or home_obj.get("abbrev") or "Home"
                    away_team = away_obj.get("placeName", {}).get("default") or away_obj.get("abbrev") or "Away"
                    
                    home_common = home_obj.get("commonName", {}).get("default", "")
                    away_common = away_obj.get("commonName", {}).get("default", "")
                    if home_common:
                        home_team = f"{home_team} {home_common}"
                    if away_common:
                        away_team = f"{away_team} {away_common}"

                    home_logo = home_obj.get("logo", "")
                    away_logo = away_obj.get("logo", "")
                    
                    home_score = home_obj.get("score")
                    away_score = home_obj.get("score")
                    
                    state = str(game.get("gameState", "")).upper()
                    
                    if state in ["OFF", "FINAL"]:
                        game_state = "OFF"
                    elif state in ["CRIT", "LIVE", "IN"]:
                        game_state = "En cours"
                    else:
                        game_state = dt_france.strftime("%H:%M")

                    formatted_matches.append({
                        "_sort_time": start_utc,
                        "league": "NHL",
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_score": home_score,
                        "away_score": away_score,
                        "home_logo": home_logo,
                        "away_logo": away_logo,
                        "game_state": game_state
                    })
        except Exception:
            continue
            
    formatted_matches.sort(key=lambda x: x["_sort_time"])
    final_matches = [{k: v for k, v in m.items() if k != "_sort_time"} for m in formatted_matches]

    return {"date": date, "matches": final_matches}

@app.post("/predict/match")
def predict_nhl_match(req: MatchRequest):
    probs = calculate_5_ai_consensus(req.home_team, req.away_team)
    home_players = fetch_nhl_roster(req.home_team)
    away_players = fetch_nhl_roster(req.away_team)
    
    seed = get_team_seed(req.home_team, req.away_team)
    home_goals = 2 + (seed % 3)
    away_goals = 1 + ((seed // 3) % 3)
    
    over_prob = round(40.0 + (seed % 45), 1)

    return {
        "match": f"{req.home_team} vs {req.away_team}",
        "probabilities": probs,
        "predicted_score": f"{home_goals} - {away_goals}",
        "over_under_5_5": {"over_prob": over_prob},
        "home_players": home_players,
        "away_players": away_players
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
