import pandas as pd
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
MATCHES_PATH = os.path.join(PARENT_DIR, "matches.csv")
DELIVERIES_PATH = os.path.join(PARENT_DIR, "deliveries.csv")

print("Loading datasets...")
matches = pd.read_csv(MATCHES_PATH)
deliveries = pd.read_csv(DELIVERIES_PATH)

print("Processing data...")
# Merge to get match details for every ball
data = deliveries.merge(matches, left_on='match_id', right_on='id')

# --- 1. PREPARE BATTING DATA ---
# Group by Match and Batsman to get total runs per match
batting_data = data.groupby(['match_id', 'batsman', 'batting_team', 'bowling_team', 'city']).agg({
    'batsman_runs': 'sum'
}).reset_index()

# Filter for legitimate teams/players if needed
# For simplicity, we keep all for now, but ensure string consistency

batting_features = batting_data[['batsman', 'bowling_team', 'city']]
batting_target = batting_data['batsman_runs']

# --- 2. PREPARE BOWLING DATA ---
# Filter for wickets (excluding run outs)
# dismissal_kind is NaN if no wicket.
# We consider a wicket for the bowler if dismissal_kind is NOT run out, retired hurt, etc.
valid_wickets = ['caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket']
data['is_wicket'] = data['dismissal_kind'].apply(lambda x: 1 if x in valid_wickets else 0)

bowling_data = data.groupby(['match_id', 'bowler', 'bowling_team', 'batting_team', 'city']).agg({
    'is_wicket': 'sum'
}).reset_index()

# note: for bowling data, 'bowling_team' is the bowler's team, 'batting_team' is the opposition
bowling_features = bowling_data[['bowler', 'batting_team', 'city']]
bowling_target = bowling_data['is_wicket'] # Wickets taken

# --- 3. METADATA EXTRACTION ---
# We need unique lists for the frontend
unique_batsmen = sorted(data['batsman'].unique().tolist())
unique_bowlers = sorted(data['bowler'].unique().tolist())
unique_teams = sorted(matches['team1'].dropna().unique().tolist())
unique_cities = sorted(matches['city'].dropna().unique().tolist())

metadata = {
    "batsmen": unique_batsmen,
    "bowlers": unique_bowlers,
    "teams": unique_teams,
    "cities": unique_cities
}

with open(os.path.join(BASE_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f)
print("Metadata saved.")

# --- 4. TRAIN BATTING MODEL ---
print("Training Batting Model...")

# Pipeline for Batting
# We will OneHotEncode all categorical features.
# Handling unknown categories is important for robustness.
batting_pipeline = Pipeline([
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['batsman', 'bowling_team', 'city'])
        ],
        remainder='passthrough'
    )),
    ('regressor', RandomForestRegressor(n_estimators=10, random_state=42))
])

batting_pipeline.fit(batting_features, batting_target)

with open(os.path.join(BASE_DIR, "player_batting.pkl"), "wb") as f:
    pickle.dump(batting_pipeline, f)

# --- 5. TRAIN BOWLING MODEL ---
print("Training Bowling Model...")

bowling_pipeline = Pipeline([
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['bowler', 'batting_team', 'city'])
        ],
        remainder='passthrough'
    )),
    ('regressor', RandomForestRegressor(n_estimators=10, random_state=42)) 
])

bowling_pipeline.fit(bowling_features, bowling_target)

with open(os.path.join(BASE_DIR, "player_bowling.pkl"), "wb") as f:
    pickle.dump(bowling_pipeline, f)

print("All models trained and saved successfully.")
