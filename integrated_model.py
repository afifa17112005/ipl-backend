import pickle
import pandas as pd
import numpy as np
import os
from player_model import predict_player_performance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load Base Match Model
try:
    with open(os.path.join(BASE_DIR, "pipe.pkl"), "rb") as f:
        match_pipe = pickle.load(f)
except Exception as e:
    print(f"Error loading match model: {e}")
    match_pipe = None

def get_integrated_prediction(match_data, striker_name, bowler_name):
    """
    Combines Match Context + Player Performance to give a refined probability.
    """
    if not match_pipe:
        return {"error": "Model not loaded"}

    # 1. Base Win Probability (Model predicts for Bowling Team [0] and Batting Team [1])
    # The input dataframe must match the training format EXACTLY
    input_df = pd.DataFrame([{
        "batting_team": match_data["batting_team"],
        "bowling_team": match_data["bowling_team"],
        "city": match_data["city"],
        "runs_left": match_data["runs_left"],
        "balls_left": match_data["balls_left"],
        "wickets": match_data["wickets_left"],
        "total_runs_x": match_data["total_runs_x"],
        "cur_run_rate": match_data["cur_run_rate"],
        "req_run_rate": match_data["req_run_rate"]
    }])

    base_proba = match_pipe.predict_proba(input_df)[0]
    bowling_win_prob = base_proba[0]
    batting_win_prob = base_proba[1]

    print(f"Base Prob -> Batting: {batting_win_prob}, Bowling: {bowling_win_prob}")

    # 2. Player Performance Predictions
    striker_runs = 0
    bowler_wickets = 0

    if striker_name:
        try:
            p_res = predict_player_performance("batsman", striker_name, match_data["bowling_team"], match_data["city"])
            striker_runs = p_res["predicted_runs"]
        except:
            pass # Default to 0 impact if model fails

    if bowler_name:
        try:
            p_res = predict_player_performance("bowler", bowler_name, match_data["batting_team"], match_data["city"])
            bowler_wickets = p_res["predicted_wickets"]
        except:
            pass

    print(f"Player Stats -> Striker Runs: {striker_runs}, Bowler Wickets: {bowler_wickets}")

    # 3. Fusion Logic (Heuristic Adjustment)
    # Rationale: 
    # - If Striker is predicted to score many runs (e.g. >30), it improves Batting Win %.
    # - If Bowler is predicted to take wickets (e.g. >1), it improves Bowling Win %.
    
    # Impact Factors (Tune these based on desired sensitivity)
    RUNS_IMPACT_FACTOR = 0.005  # 10 runs = +5% win chance
    WICKETS_IMPACT_FACTOR = 0.10 # 1 wicket = +10% win chance

    # Calculate Adjustments
    batting_boost = (striker_runs * RUNS_IMPACT_FACTOR)
    bowling_boost = (bowler_wickets * WICKETS_IMPACT_FACTOR)

    # Apply to Base Probabilities
    # We apply the boost towards the respective team
    
    new_batting_prob = batting_win_prob + batting_boost - bowling_boost
    
    # Normalize (ensure 0 to 1 range)
    new_batting_prob = max(0.0, min(1.0, new_batting_prob))
    new_bowling_prob = 1.0 - new_batting_prob

    return {
        "batting_win": round(new_batting_prob * 100, 2),
        "bowling_win": round(new_bowling_prob * 100, 2),
        "details": {
            "base_batting_prob": round(batting_win_prob * 100, 2),
            "striker_predicted_runs": striker_runs,
            "bowler_predicted_wickets": bowler_wickets
        }
    }
