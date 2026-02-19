import pickle
import pandas as pd
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load Models
try:
    with open(os.path.join(BASE_DIR, "player_batting.pkl"), "rb") as f:
        batting_pipe = pickle.load(f)
    
    with open(os.path.join(BASE_DIR, "player_bowling.pkl"), "rb") as f:
        bowling_pipe = pickle.load(f)
        
    with open(os.path.join(BASE_DIR, "metadata.json"), "r") as f:
        metadata = json.load(f)
except FileNotFoundError:
    print("⚠️ Player models or metadata not found. Please run train_player_model.py first.")
    batting_pipe = None
    bowling_pipe = None
    metadata = {}

def get_metadata():
    return metadata

def predict_player_performance(player_type, player_name, opposition, venue):
    if not batting_pipe or not bowling_pipe:
        raise Exception("Models not loaded")

    # Construct input dataframe
    # The models expect: 
    # Batting: ['batsman', 'bowling_team', 'city'] -> Target: runs
    # Bowling: ['bowler', 'batting_team', 'city'] -> Target: wickets
    
    data = {}
    
    if player_type == "batsman":
        data = {
            'batsman': [player_name],
            'bowling_team': [opposition],
            'city': [venue]
        }
        df = pd.DataFrame(data)
        prediction = batting_pipe.predict(df)[0]
        return {
            "player_name": player_name,
            "predicted_runs": round(prediction),
            "type": "batsman"
        }
        
    elif player_type == "bowler":
        data = {
            'bowler': [player_name],
            'batting_team': [opposition],
            'city': [venue]
        }
        df = pd.DataFrame(data)
        prediction = bowling_pipe.predict(df)[0]
        return {
            "player_name": player_name,
            "predicted_wickets": round(prediction),
            "type": "bowler"
        }
    
    else:
        raise ValueError("Invalid player_type. Must be 'batsman' or 'bowler'")
