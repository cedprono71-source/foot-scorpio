from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from scipy.stats import poisson
import requests
import os
import json
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from functools import lru_cache

app = FastAPI(
    title="Multi-IA NHL & Football Dashboard API",
    version="3.0",
    description="Statistiques NHL dynamiques et projections de matchs."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NHL_API = "https://api-web.nhle.com/v1"
NHL_TIMEOUT = 10
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.json")


class MatchInput(BaseModel):
    home_team: str = Field(min_length=1)
    away_team: str = Field(min_length=1)
    home_expected_goals: float = Field(default=3.1, ge=0.1, le=10)
    away_expected_goals: float = Field(default=2.7, ge=0.1, le=10)
    home_b2b: bool = False
    away_b2b: bool = False


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
                return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def save_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as file:
            json.dump(history, file, ensure_ascii=False, indent=2)
    except OSError:
        pass


def normalize_name(value):
    value = str(value or "").lower().strip()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]", "", value)


def nhl_get(endpoint):
    """Récupère une ressource JSON NHL ou lève une erreur explicite."""
    response = requests.get(
        f"{NHL_API}/{endpoint.lstrip('/')}",
        timeout=NHL_TIMEOUT,
        headers={"User-Agent": "NHL-Football-Dashboard/3.0"},
    )
    response.raise_for_status()
    return response.json()


def _default_team_stats(abbrev="", name="Équipe non identifiée"):
    return {
        "abbrev": abbrev,
        "name": name,
        "offense": 3.0,
        "defense": 3.0,
        "pp_pct": 0.20,
        "pk_pct": 0.80,
        "games_played": 0,
        "goals_for": 0,
        "goals_against": 0,
        "stats_available": False,
    }


@lru_cache(maxsize=8)
def _fetch_team_stats_cached(cache_slot):
    data = nhl_get("standings/now")
    stats_map = {}

    for record in data.get("standings", []):
        abbrev = record.get("teamAbbrev", {})
        place = record.get("placeName", {})
        team = record.get("teamName", {})

        abbrev = abbrev.get("default", "") if isinstance(abbrev, dict) else str(abbrev)
        place = place.get("default", "") if isinstance(place, dict) else str(place)
        team = team.get("default", "") if isinstance(team, dict) else str(team)
        full_name = f"{place} {team}".strip()

        if not abbrev or not full_name:
            continue

        games = int(record.get("gamesPlayed") or 0)
        goals_for = int(record.get("goalFor") or 0)
        goals_against = int(record.get("goalAgainst") or 0)
        offense = goals_for / games if games else 3.0
        defense = goals_against / games if games else 3.0

        # Estimations de repli uniquement : ce ne sont pas les taux officiels PP/PK.
        pp_estimate = min(0.30, max(0.12, (offense / 3.0) * 0.20))
        pk_estimate = min(0.88, max(0.70, 0.80 + (2.8 - defense) * 0.05))

        stats_map[abbrev.upper()] = {
            "abbrev": abbrev.upper(),
            "name": full_name,
            "offense": offense,
            "defense": defense,
            "pp_pct": pp_estimate,
            "pk_pct": pk_estimate,
            "games_played": games,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "stats_available": games > 0,
        }

    return stats_map


def fetch_team_stats():
    cache_slot = int(datetime.now(timezone.utc).timestamp() // 300)
    try:
        return _fetch_team_stats_cached(cache_slot)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Statistiques NHL indisponibles temporairement : {type(exc).__name__}",
        )


def resolve_team(team_input, team_stats):
    query = normalize_name(team_input)
    aliases = {
        "NYR": ["NYR", "Rangers", "New York Rangers"],
        "NYI": ["NYI", "Islanders", "New York Islanders"],
        "ANA": ["ANA", "Ducks", "Anaheim Ducks"],
        "BOS": ["BOS", "Bruins", "Boston Bruins"],
        "BUF": ["BUF", "Sabres", "Buffalo Sabres"],
        "CGY": ["CGY", "Flames", "Calgary Flames"],
        "CAR": ["CAR", "Hurricanes", "Carolina Hurricanes"],
        "CHI": ["CHI", "Blackhawks", "Chicago Blackhawks"],
        "COL": ["COL", "Avalanche", "Colorado Avalanche"],
        "CBJ": ["CBJ", "Blue Jackets", "Columbus Blue Jackets"],
        "DAL": ["DAL", "Stars", "Dallas Stars"],
        "DET": ["DET", "Red Wings", "Detroit Red Wings"],
        "EDM": ["EDM", "Oilers", "Edmonton Oilers"],
        "FLA": ["FLA", "Panthers", "Florida Panthers"],
        "LAK": ["LAK", "Kings", "Los Angeles Kings"],
        "MIN": ["MIN", "Wild", "Minnesota Wild"],
        "MTL": ["MTL", "Canadiens", "Montreal Canadiens"],
        "NSH": ["NSH", "Predators", "Nashville Predators"],
        "NJD": ["NJD", "Devils", "New Jersey Devils"],
        "NYI": ["NYI", "Islanders", "New York Islanders"],
        "OTT": ["OTT", "Senators", "Ottawa Senators"],
        "PHI": ["PHI", "Flyers", "Philadelphia Flyers"],
        "PIT": ["PIT", "Penguins", "Pittsburgh Penguins"],
        "SEA": ["SEA", "Kraken", "Seattle Kraken"],
        "SJS": ["SJS", "Sharks", "San Jose Sharks"],
        "STL": ["STL", "Blues", "St. Louis Blues"],
        "TBL": ["TBL", "Lightning", "Tampa Bay Lightning"],
        "TOR": ["TOR", "Maple Leafs", "Toronto Maple Leafs"],
        "UTA": ["UTA", "Mammoth", "Utah Mammoth", "Utah Hockey Club"],
        "VAN": ["VAN", "Canucks", "Vancouver Canucks"],
        "VGK": ["VGK", "Golden Knights", "Vegas Golden Knights", "Vegas"],
        "WSH": ["WSH", "Capitals", "Washington Capitals"],
        "WPG": ["WPG", "Jets", "Winnipeg Jets"],
    }

    for abbrev, names in aliases.items():
        if query in {normalize_name(name) for name in names}:
            if abbrev in team_stats:
                return team_stats[abbrev]

    exact = [
        stats for abbrev, stats in team_stats.items()
        if query in (normalize_name(abbrev), normalize_name(stats["name"]))
    ]
    if len(exact) == 1:
        return exact[0]

    partial = [
        stats for stats in team_stats.values()
        if query and (
            query in normalize_name(stats["name"])
            or normalize_name(stats["name"]) in query
        )
    ]
    if len(partial) == 1:
        return partial[0]

    raise HTTPException(
        status_code=400,
        detail=f"Équipe NHL non reconnue ou ambiguë : {team_input}. "
               "Utilise le nom complet ou l'abréviation NHL.",
    )


@lru_cache(maxsize=64)
def _fetch_goalie_stats_cached(team_abbrev, cache_slot):
    data = nhl_get(f"club-stats/{team_abbrev}/now")
    goalies = [
        goalie for goalie in data.get("goalies", [])
        if int(goalie.get("gamesPlayed") or 0) > 0
    ]

    if not goalies:
        return {
            "name": "Gardien à confirmer",
            "save_pct": 0.905,
            "gaa": 2.80,
            "source": "valeurs de repli; gardien non confirmé",
            "stats_available": False,
        }

    goalie = max(goalies, key=lambda item: int(item.get("gamesPlayed") or 0))
    first = goalie.get("firstName", {})
    last = goalie.get("lastName", {})
    first = first.get("default", "") if isinstance(first, dict) else str(first)
    last = last.get("default", "") if isinstance(last, dict) else str(last)

    return {
        "name": f"{first} {last}".strip() or "Gardien à confirmer",
        "save_pct": float(goalie.get("savePercentage") or 0.905),
        "gaa": float(goalie.get("goalsAgainstAverage") or 2.80),
        "games_played": int(goalie.get("gamesPlayed") or 0),
        "source": "statistiques de saison NHL; titularisation non confirmée",
        "stats_available": True,
    }


def fetch_goalie_stats(team_abbrev, team_full_name=""):
    if not team_abbrev:
        return {
            "name": "Gardien à confirmer",
            "save_pct": 0.905,
            "gaa": 2.80,
            "source": "équipe non identifiée",
            "stats_available": False,
        }

    slot = int(datetime.now(timezone.utc).timestamp() // 300)
    try:
        return _fetch_goalie_stats_cached(team_abbrev.upper(), slot)
    except Exception:
        return {
            "name": "Gardien à confirmer",
            "save_pct": 0.905,
            "gaa": 2.80,
            "source": "API indisponible; valeurs de repli",
            "stats_available": False,
        }


@lru_cache(maxsize=64)
def _fetch_skaters_cached(team_abbrev, cache_slot):
    data = nhl_get(f"club-stats/{team_abbrev}/now")
    result = {}

    for player in data.get("skaters", []):
        first = player.get("firstName", {})
        last = player.get("lastName", {})
        first = first.get("default", "") if isinstance(first, dict) else str(first)
        last = last.get("default", "") if isinstance(last, dict) else str(last)
        name = f"{first} {last}".strip()
        if not name:
            continue

        result[normalize_name(name)] = {
            "name": name,
            "pos": player.get("positionCode", "F"),
            "games": int(player.get("gamesPlayed") or 0),
            "goals": int(player.get("goals") or 0),
            "assists": int(player.get("assists") or 0),
            "points": int(player.get("goals") or 0) + int(player.get("assists") or 0),
            "shots": int(player.get("shots") or 0),
            "power_play_goals": int(player.get("powerPlayGoals") or 0),
            "power_play_points": int(player.get("powerPlayPoints") or 0),
        }

    return result


@lru_cache(maxsize=64)
def _fetch_all_players_cached(team_abbrev, expected_team_goals, cache_slot):
    roster_data = nhl_get(f"roster/{team_abbrev}/current")
    skaters = _fetch_skaters_cached(team_abbrev, cache_slot)
    players = []

    # Les gardiens ne sont pas inclus dans les projections de patineurs.
    for group in ("forwards", "defensemen"):
        for person in roster_data.get(group, []):
            first = person.get("firstName", {})
            last = person.get("lastName", {})
            first = first.get("default", "") if isinstance(first, dict) else str(first)
            last = last.get("default", "") if isinstance(last, dict) else str(last)
            name = f"{first} {last}".strip()
            if not name:
                continue

            season = skaters.get(normalize_name(name), {})
            games = int(season.get("games", 0))
            goals = int(season.get("goals", 0))
            assists = int(season.get("assists", 0))
            points = goals + assists

            if games > 0:
                scale = max(0.1, float(expected_team_goals)) / 3.0
                goal_prob = float(np.clip(
                    (1 - np.exp(-(goals / games) * scale)) * 100, 0, 90
                ))
                assist_prob = float(np.clip(
                    (1 - np.exp(-(assists / games) * scale)) * 100, 0, 90
                ))
                point_prob = float(np.clip(
                    (1 - np.exp(-(points / games) * scale)) * 100, 0, 95
                ))
                goal_prob = round(goal_prob, 1)
                assist_prob = round(assist_prob, 1)
                point_prob = round(point_prob, 1)
            else:
                goal_prob = assist_prob = point_prob = None

            players.append({
                "player_id": person.get("id"),
                "name": name,
                "pos": person.get("positionCode", season.get("pos", "F")),
                "games": games,
                "goals": goals,
                "assists": assists,
                "points": points,
                "shots": season.get("shots", 0),
                "power_play_goals": season.get("power_play_goals", 0),
                "power_play_points": season.get("power_play_points", 0),
                "goal_prob": goal_prob,
                "assist_prob": assist_prob,
                "point_prob": point_prob,
                "stats_available": games > 0,
            })

    players.sort(
        key=lambda player: (
            player["point_prob"] is not None,
            player["point_prob"] or 0
        ),
        reverse=True,
    )
    scorers = [p for p in players if p["goal_prob"] is not None]
    passers = [p for p in players if p["assist_prob"] is not None]

    return {
        "all": players,
        "top_scorer": max(scorers, key=lambda p: p["goal_prob"], default=None),
        "top_passer": max(passers, key=lambda p: p["assist_prob"], default=None),
        "source": "API NHL : effectif actuel et statistiques de saison",
    }


def fetch_all_players_projections(team_abbrev, expected_team_goals, team_full_name=""):
    if not team_abbrev:
        return {
            "all": [],
            "top_scorer": None,
            "top_passer": None,
            "source": "équipe non identifiée",
        }

    slot = int(datetime.now(timezone.utc).timestamp() // 300)
    try:
        return _fetch_all_players_cached(
            team_abbrev.upper(), round(float(expected_team_goals), 2), slot
        )
    except Exception as exc:
        return {
            "all": [],
            "top_scorer": None,
            "top_passer": None,
            "source": f"API NHL indisponible : {type(exc).__name__}",
        }


def calculate_poisson_matrix(home_xg, away_xg):
    # La matrice inclut les scores de 0 à 10 buts par équipe.
    max_goals = 11
    matrix = np.zeros((max_goals, max_goals))
    for home_goals in range(max_goals):
        for away_goals in range(max_goals):
            matrix[home_goals, away_goals] = (
                poisson.pmf(home_goals, home_xg)
                * poisson.pmf(away_goals, away_xg)
            )
    return matrix


@app.get("/")
def read_root():
    file_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return HTMLResponse(
                content=file.read(),
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )
    return {"status": "online", "message": "API NHL & Football opérationnelle."}


@app.get("/health")
def health():
    return {"status": "online", "version": app.version}


@app.get("/api/nhl/teams")
def get_nhl_teams():
    return {"teams": list(fetch_team_stats().values())}


@app.get("/api/nhl/roster/{team_abbrev}")
def get_current_nhl_roster(team_abbrev: str):
    team_abbrev = team_abbrev.upper().strip()
    if not re.fullmatch(r"[A-Z]{3}", team_abbrev):
        raise HTTPException(
            status_code=400,
            detail="Utilise une abréviation NHL de 3 lettres, par exemple NYR.",
        )

    teams = fetch_team_stats()
    if team_abbrev not in teams:
        raise HTTPException(status_code=404, detail="Équipe NHL inconnue.")

    try:
        roster = nhl_get(f"roster/{team_abbrev}/current")
        projections = fetch_all_players_projections(team_abbrev, 3.0)
        return {
            "team": teams[team_abbrev],
            "roster": roster,
            "projections": projections,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Impossible de récupérer l'effectif NHL : {type(exc).__name__}",
        )


@app.post("/predict/match")
def predict_match(match: MatchInput):
    team_stats = fetch_team_stats()
    home_st = resolve_team(match.home_team, team_stats)
    away_st = resolve_team(match.away_team, team_stats)

    home_abbrev = home_st["abbrev"]
    away_abbrev = away_st["abbrev"]

    home_goalie = fetch_goalie_stats(home_abbrev, match.home_team)
    away_goalie = fetch_goalie_stats(away_abbrev, match.away_team)

    # Les buts attendus peuvent être fournis par le tableau de bord.
    # Sinon, on utilise l'attaque de saison avec un avantage domicile modéré.
    home_base = match.home_expected_goals
    away_base = match.away_expected_goals

    home_fatigue = 0.88 if match.home_b2b else 1.0
    away_fatigue = 0.88 if match.away_b2b else 1.0

    # Ajustement léger selon la défense adverse et la forme du gardien.
    home_defence_factor = np.clip(3.0 / max(1.0, away_st["defense"]), 0.85, 1.15)
    away_defence_factor = np.clip(3.0 / max(1.0, home_st["defense"]), 0.85, 1.15)

    home_goalie_factor = np.clip(
        1.0 - (away_goalie["save_pct"] - 0.905) * 1.5, 0.90, 1.10
    )
    away_goalie_factor = np.clip(
        1.0 - (home_goalie["save_pct"] - 0.905) * 1.5, 0.90, 1.10
    )

    final_h_xg = float(np.clip(
        home_base * home_defence_factor * home_goalie_factor * home_fatigue,
        0.2, 9.0
    ))
    final_a_xg = float(np.clip(
        away_base * away_defence_factor * away_goalie_factor * away_fatigue,
        0.2, 9.0
    ))

    matrix = calculate_poisson_matrix(final_h_xg, final_a_xg)
    total_mass = float(matrix.sum())
    if total_mass > 0:
        matrix = matrix / total_mass

    # En hockey, un match à égalité après 60 minutes part en prolongation.
    regulation_home = float(np.tril(matrix, -1).sum())
    regulation_away = float(np.triu(matrix, 1).sum())
    regulation_draw = float(np.trace(matrix))

    # Estimation simplifiée du vainqueur en prolongation/tirs au but :
    # partage 50/50 des probabilités de match à égalité après 60 minutes.
    home_win = (regulation_home + 0.5 * regulation_draw) * 100
    away_win = (regulation_away + 0.5 * regulation_draw) * 100

    best_score = np.unravel_index(np.argmax(matrix), matrix.shape)
    over_5_5 = float(sum(
        matrix[h, a]
        for h in range(matrix.shape[0])
        for a in range(matrix.shape[1])
        if h + a >= 6
    ))

    home_players = fetch_all_players_projections(
        home_abbrev, final_h_xg, match.home_team
    )
    away_players = fetch_all_players_projections(
        away_abbrev, final_a_xg, match.away_team
    )

    return {
        "match": f"{home_st['name']} vs {away_st['name']}",
        "teams": {
            "home": home_st,
            "away": away_st,
        },
        "probabilities": {
            "home_win": round(home_win, 1),
            "draw_after_60": round(regulation_draw * 100, 1),
            "away_win": round(away_win, 1),
        },
        "expected_goals": {
            "home": round(final_h_xg, 2),
            "away": round(final_a_xg, 2),
        },
        "predicted_score": f"{int(best_score[0])} - {int(best_score[1])}",
        "over_under_5_5": {
            "over_prob": round(over_5_5 * 100, 1),
            "under_prob": round((1 - over_5_5) * 100, 1),
        },
        "goalies": {
            "home": home_goalie,
            "away": away_goalie,
        },
        "special_teams": {
            "home_pp_estimate": round(home_st["pp_pct"] * 100, 1),
            "away_pp_estimate": round(away_st["pp_pct"] * 100, 1),
            "note": "Estimations dérivées des buts par match, pas des taux officiels PP/PK.",
        },
        "players": {
            "home": home_players,
            "away": away_players,
        },
        "model_note": (
            "Estimations statistiques, non garanties. Le score prédit est le score "
            "le plus probable après 60 minutes selon la matrice de Poisson; "
            "la probabilité de victoire partage à parts égales les égalités après 60 minutes."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/predict/auto-sync")
def auto_sync_today_matches(date: str = "now"):
    # Route conservée pour compatibilité avec le tableau de bord existant.
    return {"status": "success", "date": date, "message": "Route disponible."}


@app.get("/api/matches/nhl")
def get_nhl_matches(date: str = "now"):
    try:
        target_date = (
            datetime.strptime(date, "%Y-%m-%d").date()
            if date != "now"
            else datetime.now(timezone.utc).date()
        )
        # L'API schedule utilise des dates NHL; on récupère la veille
        # pour couvrir les rencontres autour du décalage horaire.
        start_date = (target_date - timedelta(days=1)).strftime("%Y-%m-%d")
        data = nhl_get(f"schedule/{start_date}")
        matches = []

        for day in data.get("gameWeek", []):
            game_date = day.get("date")
            for game in day.get("games", []):
                home = game.get("homeTeam", {})
                away = game.get("awayTeam", {})

                home_place = home.get("placeName", {})
                home_common = home.get("commonName", {})
                away_place = away.get("placeName", {})
                away_common = away.get("commonName", {})

                home_name = (
                    home_place.get("default", "") if isinstance(home_place, dict) else ""
                ) or (
                    home_common.get("default", "Domicile")
                    if isinstance(home_common, dict) else "Domicile"
                )
                away_name = (
                    away_place.get("default", "") if isinstance(away_place, dict) else ""
                ) or (
                    away_common.get("default", "Extérieur")
                    if isinstance(away_common, dict) else "Extérieur"
                )

                matches.append({
                    "id": game.get("id"),
                    "date": game_date,
                    "home_team": home_name,
                    "away_team": away_name,
                    "home_abbrev": home.get("abbrev", ""),
                    "away_abbrev": away.get("abbrev", ""),
                    "home_logo": home.get("logo", ""),
                    "away_logo": away.get("logo", ""),
                    "time": game.get("startTimeUTC"),
                    "game_state": game.get("gameState"),
                })

        # Ne garder que la date demandée lorsque l'API la fournit.
        matches = [item for item in matches if item.get("date") == target_date.isoformat()]
        return {"matches": matches, "date": target_date.isoformat()}
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Format de date attendu : YYYY-MM-DD ou now."
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Calendrier NHL indisponible : {type(exc).__name__}",
        )


@app.get("/api/football/matches")
def get_football_matches():
    # À connecter à un fournisseur de données football.
    return {
        "sport": "Football",
        "matches": [],
        "message": "Aucune source de données football n'est configurée.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
