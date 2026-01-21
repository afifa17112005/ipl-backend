from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import predict

app = FastAPI(
    title="IPL Win Predictor API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MatchInput(BaseModel):
    batting_team: str
    bowling_team: str
    city: str
    runs_left: int
    balls_left: int
    wickets_left: int
    total_runs_x: int
    cur_run_rate: float
    req_run_rate: float

@app.get("/")
def health():
    return {"status": "Backend is running"}

@app.post("/predict")
def get_prediction(data: MatchInput):
    return {
        "win_probability": predict(data.dict())
    }
