import pickle
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pipe.pkl")

with open(MODEL_PATH, "rb") as f:
    pipe = pickle.load(f)

def predict(match):
    # 🔴 IMPORTANT: column names MUST match training dataset
    data = {
        "batting_team": match["batting_team"],
        "bowling_team": match["bowling_team"],
        "city": match["city"],
        "runs_left": match["runs_left"],
        "balls_left": match["balls_left"],
        "wickets": match["wickets_left"],   # MODEL COLUMN NAME
        "total_runs_x": match["total_runs_x"],
        "cur_run_rate": match["cur_run_rate"],
        "req_run_rate": match["req_run_rate"]
    }

    df = pd.DataFrame([data])

    proba = pipe.predict_proba(df)[0]

    return {
        "win_probability": {
            "batting_win": round(proba[1] * 100, 2),
            "bowling_win": round(proba[0] * 100, 2)
        }
    }
