import os
import sys
import json
import math
import numpy as np
import pandas as pd

COMPETITION_MATRIX = {
    "england": {"tier_1": "Premier League", "baseline_goals": 2.84, "tier_2": "Championship", "tier_2_baseline_goals": 2.68, "cups": ["FA Cup", "EFL Cup"]},
    "spain": {"tier_1": "La Liga", "baseline_goals": 2.55, "tier_2": "Segunda División", "tier_2_baseline_goals": 2.42, "cups": ["Copa del Rey", "Supercopa de España"]},
    "france": {"tier_1": "Ligue 1", "baseline_goals": 2.60, "tier_2": "Ligue 2", "tier_2_baseline_goals": 2.38, "cups": ["Coupe de France", "Trophée des Champions"]},
    "germany": {"tier_1": "Bundesliga", "baseline_goals": 3.18, "tier_2": "2. Bundesliga", "tier_2_baseline_goals": 3.02, "cups": ["DFB-Pokal", "DFL-Supercup"]},
    "italy": {"tier_1": "Serie A", "baseline_goals": 2.58, "tier_2": "Serie B", "tier_2_baseline_goals": 2.46, "cups": ["Coppa Italia", "Supercoppa Italiana"]},
    "australia": {"tier_1": "A-League Men", "baseline_goals": 3.04, "cups": ["Australia Cup"]},
    "austria": {"tier_1": "Austrian Football Bundesliga", "baseline_goals": 2.90, "cups": ["Austrian Cup"]},
    "belgium": {"tier_1": "Belgian Pro League", "baseline_goals": 2.85, "cups": ["Belgian Cup", "Belgian Super Cup"]},
    "brazil": {"tier_1": "Campeonato Brasileiro Série A", "baseline_goals": 2.40, "cups": ["Copa do Brasil"]},
    "china": {"tier_1": "Chinese Super League", "baseline_goals": 2.65, "cups": ["Chinese FA Cup", "Chinese FA Super Cup"]},
    "croatia": {"tier_1": "Croatian Football League", "baseline_goals": 2.60, "cups": ["Croatian Football Cup"]},
    "czech": {"tier_1": "Czech First League", "baseline_goals": 2.75, "cups": ["Czech Cup"]},
    "denmark": {"tier_1": "Danish Superliga", "baseline_goals": 2.75, "cups": ["Danish Cup"]},
    "estonia": {"tier_1": "Meistriliiga", "baseline_goals": 3.00, "cups": ["Estonian Cup"]},
    "finland": {"tier_1": "Veikkausliiga", "baseline_goals": 2.68, "cups": ["Suomen Cup", "Finnish League Cup"]},
    "iceland": {"tier_1": "Best deild karla", "baseline_goals": 3.20, "cups": ["Icelandic Cup", "Icelandic League Cup"]},
    "japan": {"tier_1": "J1 League", "baseline_goals": 2.58, "cups": ["Emperor's Cup", "J.League Cup"]},
    "netherlands": {"tier_1": "Eredivisie", "baseline_goals": 3.10, "cups": ["KNVB Cup", "Johan Cruyff Shield"]},
    "norway": {"tier_1": "Eliteserien", "baseline_goals": 3.02, "cups": ["Norwegian Football Cup"]},
    "poland": {"tier_1": "Ekstraklasa", "baseline_goals": 2.62, "cups": ["Polish Cup", "Polish SuperCup"]},
    "portugal": {"tier_1": "Primeira Liga", "baseline_goals": 2.70, "cups": ["Taça de Portugal", "Taça da Liga"]},
    "russia": {"tier_1": "Russian Premier League", "baseline_goals": 2.65, "cups": ["Russian Cup", "Russian Super Cup"]},
    "south_korea": {"tier_1": "K League 1", "baseline_goals": 2.55, "cups": ["Korean FA Cup"]},
    "sweden": {"tier_1": "Allsvenskan", "baseline_goals": 2.74, "cups": ["Svenska Cupen"]},
    "switzerland": {"tier_1": "Swiss Super League", "baseline_goals": 2.95, "cups": ["Swiss Cup"]},
    "canada": {"tier_1": "Canadian Premier League", "baseline_goals": 2.52, "cups": ["Canadian Championship"]},
    "egypt": {"tier_1": "Egyptian Premier League", "baseline_goals": 2.30, "cups": ["Egypt Cup", "Egyptian Super Cup"]},
    "morocco": {"tier_1": "Botola Pro", "baseline_goals": 2.15, "cups": ["Moroccan Throne Cup"]},
    "usa": {"tier_1": "Major League Soccer", "baseline_goals": 2.92, "cups": ["US Open Cup", "MLS Cup Playoffs"]},
    "usa_usl_championship": {"tier_1": "USL Championship", "baseline_goals": 2.74, "cups": ["US Open Cup"]},
    "argentina": {"tier_1": "Liga Profesional", "baseline_goals": 2.11, "cups": ["Copa Argentina", "Copa de la Liga"]},
    "chile": {"tier_1": "Primera División", "baseline_goals": 2.58, "cups": ["Copa Chile"]},
    "south africa": {"tier_1": "Betway Premiership", "baseline_goals": 2.02, "cups": ["Nedbank Cup", "MTN8", "Carling Knockout"]},
    "faroe islands": {"tier_1": "Premier League", "baseline_goals": 3.57, "cups": ["Faroe Islands Cup"]},
    "greece": {"tier_1": "Super League 1", "baseline_goals": 2.45, "cups": ["Greek Football Cup"]},
    "hungary": {"tier_1": "Nemzeti Bajnokság I", "baseline_goals": 2.72, "cups": ["Magyar Kupa"]},
    "latvia": {"tier_1": "Virsliga", "baseline_goals": 2.91, "cups": ["Latvian Football Cup"]},
    "paraguay": {"tier_1": "Primera División", "baseline_goals": 2.48, "cups": ["Copa Paraguay"]},
    "scotland": {"tier_1": "Scottish Premiership", "baseline_goals": 2.78, "cups": ["Scottish Cup", "Scottish League Cup"]}
}

SQUAD_TURNOVER_MATRIX = {
    "stable": {"att_modifier": 1.00, "def_modifier": 1.00},
    "promoted": {"att_modifier": 0.88, "def_modifier": 1.15},
    "relegated": {"att_modifier": 1.12, "def_modifier": 0.90}
}

def calculate_time_decay_weight(days_ago, half_life_days=45, freeze_decay=False):
    if freeze_decay: return 1.0
    return math.exp(-0.69314718056 * (max(0, days_ago) / max(1, half_life_days)))

def calculate_rest_multiplier(days_rest):
    if days_rest <= 2: return 0.88
    elif days_rest == 3: return 0.94
    elif days_rest >= 4 and days_rest <= 6: return 1.00
    elif days_rest >= 7 and days_rest <= 10: return 1.04
    else: return 0.97

def calculate_dynamic_home_advantage(df, current_ts):
    past_df = df[df["match_timestamp"] < current_ts]
    if past_df.empty: return 1.10, 0.90
    avg_h = past_df["home_goals"].mean()
    avg_a = past_df["away_goals"].mean()
    total_avg = (avg_h + avg_a) / 2
    if total_avg == 0 or pd.isna(total_avg): return 1.10, 0.90
    return max(1.0, min(1.3, avg_h / total_avg)), max(0.7, min(1.0, avg_a / total_avg))
def calculate_h2h_psychological_modifier(df, home, away, current_ts):
    h2h = df[(df["match_timestamp"] < current_ts) & 
             (((df["home_team"] == home) & (df["away_team"] == away)) | 
              ((df["home_team"] == away) & (df["away_team"] == home)))]
    if h2h.empty: return 1.0, 1.0
    h_points, a_points = 0, 0
    for _, row in h2h.iterrows():
        if row["home_team"] == home:
            if row["home_goals"] > row["away_goals"]: h_points += 3
            elif row["home_goals"] < row["away_goals"]: a_points += 3
            else: h_points += 1; a_points += 1
        else:
            if row["home_goals"] > row["away_goals"]: a_points += 3
            elif row["home_goals"] < row["away_goals"]: h_points += 3
            else: h_points += 1; a_points += 1
    total = h_points + a_points
    if total == 0: return 1.0, 1.0
    return 1.0 + ((h_points / total) - 0.5) * 0.1, 1.0 + ((a_points / total) - 0.5) * 0.1

def estimate_dixon_coles_tau(df, current_ts):
    past = df[df["match_timestamp"] < current_ts]
    if past.empty: return -0.05
    low_scoring = past[(past["home_goals"] <= 1) & (past["away_goals"] <= 1)]
    if low_scoring.empty: return -0.05
    zeros = len(low_scoring[(low_scoring["home_goals"] == 0) & (low_scoring["away_goals"] == 0)])
    ratio = zeros / len(low_scoring)
    return max(-0.15, min(0.0, -0.05 - (ratio * 0.05)))

def calculate_dixon_coles_adjustment(h, a, lam1, lam2, tau):
    if h == 0 and a == 0: return 1.0 - (lam1 * lam2 * tau)
    elif h == 1 and a == 0: return 1.0 + (lam1 * tau)
    elif h == 0 and a == 1: return 1.0 + (lam2 * tau)
    elif h == 1 and a == 1: return 1.0 - tau
    return 1.0

def parse_live_team_averages(df, team, current_ts, half_life_days=45, status_override="stable", freeze_decay=False):
    past = df[df["match_timestamp"] < current_ts]
    team_matches = past[(past["home_team"] == team) | (past["away_team"] == team)]
    games_played = len(team_matches)
    
    if games_played == 0:
        mod = SQUAD_TURNOVER_MATRIX.get(status_override, {"att_modifier": 1.0, "def_modifier": 1.0})
        return {
            "games_played": 0, "att_strength_goals": 1.0 * mod["att_modifier"], "def_strength_goals": 1.0 * mod["def_modifier"],
            "avg_scored": 1.2, "avg_conceded": 1.2, "box_threat": 1.0, "def_resistance": 1.0
        }
                
    total_w = 0.0
    sum_gf, sum_sot, sum_chances, sum_entries, sum_touches, sum_through = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    sum_ga, sum_intercept, sum_rec, sum_saves, sum_aerial, sum_ground, sum_tackles, sum_pass_f3 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for _, row in team_matches.iterrows():
        days_ago = (current_ts - row["match_timestamp"]).days
        w = calculate_time_decay_weight(days_ago, half_life_days, freeze_decay=freeze_decay)
        total_w += w
        
        if row["home_team"] == team:
            sum_gf += row["home_goals"] * w; sum_ga += row["away_goals"] * w
            sum_sot += row["home_sot"] * w; sum_chances += row["home_big_chances"] * w
            sum_entries += row["home_final_third_entries"] * w; sum_touches += row["home_box_touches"] * w
            sum_through += row["home_through_passes"] * w; sum_intercept += row["home_interceptions"] * w
            sum_rec += row["home_recoveries"] * w; sum_saves += row["home_saves"] * w
            
            # Direct Mobile Percentage Ingestions (Already normalized between 0.0 and 1.0)
            sum_ground += float(row.get("home_ground_duels_won_pct", 0.50)) * w
            sum_aerial += float(row.get("home_aerial_duels_won_pct", 0.50)) * w
            sum_tackles += float(row.get("home_tackles_won_pct", 0.50)) * w
            sum_pass_f3 += float(row.get("home_passes_final_third_pct", 0.70)) * w
        else:
            sum_gf += row["away_goals"] * w; sum_ga += row["home_goals"] * w
            sum_sot += row["away_sot"] * w; sum_chances += row["away_big_chances"] * w
            sum_entries += row["away_final_third_entries"] * w; sum_touches += row["away_box_touches"] * w
            sum_through += row["away_through_passes"] * w; sum_intercept += row["away_interceptions"] * w
            sum_rec += row["away_recoveries"] * w; sum_saves += row["away_saves"] * w
            
            sum_ground += float(row.get("away_ground_duels_won_pct", 0.50)) * w
            sum_aerial += float(row.get("away_aerial_duels_won_pct", 0.50)) * w
            sum_tackles += float(row.get("away_tackles_won_pct", 0.50)) * w
            sum_pass_f3 += float(row.get("away_passes_final_third_pct", 0.70)) * w
            
    avg_gf = sum_gf / total_w; avg_ga = sum_ga / total_w
    avg_sot = sum_sot / total_w; avg_chances = sum_chances / total_w
    avg_entries = sum_entries / total_w; avg_touches = sum_touches / total_w
    avg_through = sum_through / total_w
    avg_intercept = sum_intercept / total_w; avg_rec = sum_rec / total_w
    avg_saves = sum_saves / total_w
    
    avg_ground = sum_ground / total_w; avg_aerial = sum_aerial / total_w
    avg_tackles = sum_tackles / total_w; avg_pass_f3 = sum_pass_f3 / total_w

    league_h = past["home_goals"].mean() if not past.empty else 1.4
    league_a = past["away_goals"].mean() if not past.empty else 1.2
    league_base = (league_h + league_a) / 2
    if pd.isna(league_base) or league_base == 0: league_base = 1.3
    
    # Process-Based Vector Weight Overhaul
    composite_attack_index = (avg_gf * 0.35) + (avg_chances * 0.20) + (avg_sot * 0.15) + (avg_touches * 0.10) + (avg_through * 0.10) + (avg_pass_f3 * 0.10)
    att_strength = composite_attack_index / league_base
    
    composite_defense_index = (avg_ga * 0.40) + ((1.0 - avg_ground) * 0.20) + ((1.0 - avg_tackles) * 0.15) + ((1.0 - avg_aerial) * 0.15) + ((1.0 / max(0.5, avg_saves)) * 0.10)
    def_strength = composite_defense_index / league_base
    
    squad_mod = SQUAD_TURNOVER_MATRIX.get(status_override, {"att_modifier": 1.0, "def_modifier": 1.0})
    
    return {
        "games_played": games_played,
        "att_strength_goals": max(0.15, min(3.5, att_strength * squad_mod["att_modifier"])),
        "def_strength_goals": max(0.15, min(3.5, def_strength * squad_mod["def_modifier"])),
        "avg_scored": round(float(avg_gf), 2), "avg_conceded": round(float(avg_ga), 2),
        "box_threat": round(float(avg_touches + avg_entries), 2), "def_resistance": round(float(avg_ground + avg_intercept), 4)
    }
def predict_match_probabilities(historical_matches, home_team, away_team, current_timestamp, baseline_goals, home_rest_days, away_rest_days, home_status="stable", away_status="stable", max_score=6, vol_dampener=1.0, freeze_decay=False):
    home_hfa, away_hfa = calculate_dynamic_home_advantage(historical_matches, current_timestamp)
    home_rest_mod = calculate_rest_multiplier(home_rest_days)
    away_rest_mod = calculate_rest_multiplier(away_rest_days)
    home_stats = parse_live_team_averages(historical_matches, home_team, current_timestamp, status_override=home_status, freeze_decay=freeze_decay)
    away_stats = parse_live_team_averages(historical_matches, away_team, current_timestamp, status_override=away_status, freeze_decay=freeze_decay)
    home_h2h_mod, away_h2h_mod = calculate_h2h_psychological_modifier(historical_matches, home_team, away_team, current_timestamp)
    
    home_tactical_modifier = max(0.90, min(1.10, 1.0 + (home_stats["box_threat"] * 0.01) + (home_stats["def_resistance"] - away_stats["def_resistance"]) * 0.04))
    away_tactical_modifier = max(0.90, min(1.10, 1.0 + (away_stats["box_threat"] * 0.01) + (away_stats["def_resistance"] - home_stats["def_resistance"]) * 0.04))
    
    lam1 = max(0.05, home_stats["att_strength_goals"] * away_stats["def_strength_goals"] * (baseline_goals / 2) * home_hfa * home_rest_mod * home_tactical_modifier * home_h2h_mod * vol_dampener)
    lam2 = max(0.05, away_stats["att_strength_goals"] * home_stats["def_strength_goals"] * (baseline_goals / 2) * away_hfa * away_rest_mod * away_tactical_modifier * away_h2h_mod * vol_dampener)
    
    tau = estimate_dixon_coles_tau(historical_matches, current_timestamp)
    prob_matrix = np.zeros((max_score + 1, max_score + 1))
    for h in range(max_score + 1):
        for a in range(max_score + 1):
            p_home = (math.exp(-lam1) * (lam1 ** h)) / math.factorial(h) if lam1 > 0 else 0.0
            p_away = (math.exp(-lam2) * (lam2 ** a)) / math.factorial(a) if lam2 > 0 else 0.0
            prob_matrix[h, a] = max(0.0, p_home * p_away * calculate_dixon_coles_adjustment(h, a, lam1, lam2, tau))
            
    matrix_sum = prob_matrix.sum()
    if matrix_sum > 0: prob_matrix /= matrix_sum
    p_home_win, p_away_win, p_draw = 0.0, 0.0, 0.0
    for h in range(max_score + 1):
        for a in range(max_score + 1):
            p = prob_matrix[h, a]
            if h > a: p_home_win += p
            elif a > h: p_away_win += p
            else: p_draw += p
            
    return {
        "lambdas": {"lam1_home": round(lam1, 3), "lam2_away": round(lam2, 3), "dixon_coles_tau": round(tau, 3), "h2h_mods": [round(home_h2h_mod, 2), round(away_h2h_mod, 2)]},
        "market_probabilities": {"1 (Home Win)": round(p_home_win, 4), "X (Draw)": round(p_draw, 4), "2 (Away Win)": round(p_away_win, 4)},
        "secondary_markets": {"BTTS_Yes": round(float(prob_matrix[1:, 1:].sum()), 4), "Home_CS": round(float(prob_matrix[:, 0].sum()), 4), "Away_CS": round(float(prob_matrix[0, :].sum()), 4)},
        "expected_total_goals": round(lam1 + lam2, 2),
        "raw_matrix": prob_matrix
    }

def generate_dynamic_league_table(df):
    if df.empty: return pd.DataFrame()
    clean_df = df.dropna(subset=["home_goals", "away_goals"])
    if clean_df.empty: return pd.DataFrame()
    
    h_df = clean_df.groupby("home_team").agg(
        P=("home_goals", "count"), GF=("home_goals", "sum"), GA=("away_goals", "sum"),
        W=("home_goals", lambda s: (s > clean_df.loc[s.index, "away_goals"]).sum()),
        L=("home_goals", lambda s: (s < clean_df.loc[s.index, "away_goals"]).sum()),
        D=("home_goals", lambda s: (s == clean_df.loc[s.index, "away_goals"]).sum())
    )
    a_df = clean_df.groupby("away_team").agg(
        P=("away_goals", "count"), GF=("away_goals", "sum"), GA=("home_goals", "sum"),
        W=("away_choice", lambda s: (s > clean_df.loc[s.index, "home_goals"]).sum()) if "away_choice" in clean_df.columns else ("away_goals", lambda s: (s > clean_df.loc[s.index, "home_goals"]).sum()),
        L=("away_choice", lambda s: (s < clean_df.loc[s.index, "home_goals"]).sum()) if "away_choice" in clean_df.columns else ("away_goals", lambda s: (s < clean_df.loc[s.index, "home_goals"]).sum()),
        D=("away_choice", lambda s: (s == clean_df.loc[s.index, "home_goals"]).sum()) if "away_choice" in clean_df.columns else ("away_goals", lambda s: (s == clean_df.loc[s.index, "home_goals"]).sum())
    )
    table_df = h_df.add(a_df, fill_value=0).astype(int)
    table_df["PTS"] = (table_df["W"] * 3) + table_df["D"]
    table_df["GD"] = table_df["GF"] - table_df["GA"]
    table_df = table_df.reset_index().rename(columns={"index": "Club Team", "home_team": "Club Team"})
    table_df = table_df.sort_values(by=["PTS", "GD", "GF"], ascending=False).reset_index(drop=True)
    table_df.index += 1
    table_df.index.name = "Pos"
    return table_df
def run_rolling_window_backtest(df, baseline_goals, window_days=180, evaluation_step_days=7, vol_dampener=1.0):
    sorted_df = df.sort_values(by="match_timestamp").reset_index(drop=True)
    sorted_df["match_timestamp"] = pd.to_datetime(sorted_df["match_timestamp"])
    start_date = sorted_df["match_timestamp"].min()
    end_date = sorted_df["match_timestamp"].max()
    first_pivot = start_date + pd.Timedelta(days=window_days)
    if first_pivot >= end_date: return pd.DataFrame()
    
    backtest_records = []
    current_pivot = first_pivot
    while current_pivot <= end_date:
        window_start = current_pivot - pd.Timedelta(days=window_days)
        rolling_training_pool = sorted_df[(sorted_df["match_timestamp"] >= window_start) & (sorted_df["match_timestamp"] < current_pivot)]
        evaluation_bucket = sorted_df[(sorted_df["match_timestamp"] >= current_pivot) & (sorted_df["match_timestamp"] < current_pivot + pd.Timedelta(days=evaluation_step_days))]
        for _, match in evaluation_bucket.dropna(subset=["home_goals", "away_goals"]).iterrows():
            h_team, a_team, match_ts = match["home_team"], match["away_team"], match["match_timestamp"]
            has_home = h_team in rolling_training_pool["home_team"].values or h_team in rolling_training_pool["away_team"].values
            has_away = a_team in rolling_training_pool["home_team"].values or a_team in rolling_training_pool["away_team"].values
            if not (has_home and has_away): continue
            
            res = predict_match_probabilities(
                historical_matches=rolling_training_pool, home_team=h_team, away_team=a_team, current_timestamp=match_ts,
                baseline_goals=baseline_goals, home_rest_days=int(match.get("home_rest_days", 5)), away_rest_days=int(match.get("away_rest_days", 5)),
                home_status=str(match.get("home_status", "stable")), away_status=str(match.get("away_status", "stable")), max_score=6, vol_dampener=vol_dampener
            )
            hg, ag = int(match["home_goals"]), int(match["away_goals"])
            actual = "1 (Home Win)" if hg > ag else ("2 (Away Win)" if ag > hg else "X (Draw)")
            pred_prob = res["market_probabilities"][actual]
            backtest_records.append({
                "match_timestamp": match_ts, "home_team": h_team, "away_team": a_team,
                "home_goals": hg, "away_goals": ag, "actual_outcome": actual,
                "model_probability": pred_prob, "log_loss": round(-math.log(max(1e-15, pred_prob)), 5)
            })
        current_pivot += pd.Timedelta(days=evaluation_step_days)
    return pd.DataFrame(backtest_records)
