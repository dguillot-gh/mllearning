"""
FastAPI example for serving sport prediction models.
Run with: uvicorn examples.api:app --reload
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from predict import load_model, predict_single_game


class GamePredictionRequest(BaseModel):
    """Request model for game predictions."""
    sport: str = "nfl"
    task: str = "classification"

    # Common features (will be filtered by sport)
    home_id: str
    away_id: str
    team_favorite_id: Optional[str] = None
    stadium: Optional[str] = None
    weather_detail: Optional[str] = None
    schedule_season: int
    schedule_week: int
    schedule_playoff: bool = False
    stadium_neutral: bool = False
    weather_temperature: Optional[float] = None
    weather_wind_mph: Optional[float] = None
    weather_humidity: Optional[float] = None
    spread_favorite: Optional[float] = None
    over_under_line: Optional[float] = None

    # NBA-specific features
    arena: Optional[str] = None

    # NASCAR-specific features
    driver_id: Optional[str] = None
    track_name: Optional[str] = None
    car_manufacturer: Optional[str] = None
    is_playoff_race: Optional[bool] = None
    night_race: Optional[bool] = None
    track_length: Optional[float] = None
    ambient_temp: Optional[float] = None
    track_temp: Optional[float] = None
    starting_position: Optional[int] = None


app = FastAPI(title="Sports Prediction API", version="1.0.0")


@app.post("/predict")
def predict_game(request: GamePredictionRequest):
    """
    Predict the outcome of a game.

    For NFL classification: Returns probability of home team winning
    For NFL regression: Returns predicted point differential (home - away)
    """
    try:
        # Load the model
        pipeline = load_model(request.sport, request.task)

        # Convert request to dict and filter out None values and sport-specific fields
        game_data = request.dict()
        sport = game_data.pop('sport')
        task = game_data.pop('task')

        # Filter features based on sport (remove features not used by this sport)
        # In a real implementation, you'd load the sport config to know which features to keep
        filtered_data = {k: v for k, v in game_data.items() if v is not None}

        # Make prediction
        result = predict_single_game(pipeline, filtered_data)

        return {
            "sport": sport,
            "task": task,
            "prediction": result["predictions"][0],
            "probability": result.get("probabilities", [None])[0],
            "model_info": f"{sport}_{task}_model"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/")
def root():
    """API root endpoint."""
    return {
        "message": "Sports Prediction API",
        "version": "1.0.0",
        "available_sports": ["nfl"],
        "available_tasks": ["classification", "regression"],
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
