from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
def root():
    return {"status": "IPL backend running"}


@app.post("/predict")
def get_prediction(data: MatchInput):
    return {
        "win_probability": predict(data.dict())
    }
