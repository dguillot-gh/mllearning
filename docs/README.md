# Sports ML Learning: Multi-Sport Prediction System

A modular, extensible machine learning system for predicting sports game outcomes. Supports multiple sports with a unified architecture that makes it easy to add new sports and prediction tasks.

## 🎯 Overview

This project provides a complete ML pipeline for sports prediction with the following features:

- **Multi-Sport Support**: NFL (implemented), NBA, NASCAR (ready to implement)
- **Modular Architecture**: Easy to extend with new sports
- **Multiple Tasks**: Classification (win prediction) and regression (point differential)
- **Production Ready**: API, batch scoring, and web UI examples included
- **Configurable**: YAML-based sport configurations

## 📁 Project Structure

```
mllearning/
├── data/                    # Sport-specific data
│   └── {sport}/
│       ├── raw/            # Raw CSV files
│       └── processed/      # Processed data (future)
├── src/                    # Core modules
│   ├── sports/            # Sport implementations
│   │   ├── base.py        # Abstract base class
│   │   └── nfl.py         # NFL implementation
│   ├── data_loader.py     # Data loading utilities
│   ├── model_pipeline.py  # ML pipeline building
│   ├── train.py          # Training orchestration
│   └── predict.py        # Prediction utilities
├── configs/               # Sport configurations
│   ├── nfl_config.yaml
│   ├── nba_config.yaml
│   └── nascar_config.yaml
├── models/                # Trained models
│   └── {sport}/
├── scripts/               # CLI scripts
│   └── train_model.py    # Main training script
├── examples/              # Usage examples
│   ├── api.py            # FastAPI service
│   ├── batch_scoring.py  # Batch prediction
│   └── streamlit_app.py  # Web UI
├── docs/                  # Documentation
├── tests/                 # Unit tests
└── requirements.txt       # Dependencies
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Your First Model

```bash
# Train NFL classification model (predict home team win)
python scripts/train_model.py --sport nfl --task classification

# Train NFL regression model (predict point differential)
python scripts/train_model.py --sport nfl --task regression
```

### 3. Make Predictions

```python
from src.predict import load_model, predict_single_game

# Load trained model
pipeline = load_model('nfl', 'classification')

# Predict a game
game_data = {
    'home_id': 'DAL',
    'away_id': 'PHI',
    'schedule_season': 2025,
    'schedule_week': 5,
    'schedule_playoff': False,
    'stadium_neutral': False,
    'spread_favorite': -3.5,
    'over_under_line': 47.5
}

result = predict_single_game(pipeline, game_data)
print(f"Home win probability: {result['probabilities'][0]:.3f}")
```

## 📖 How to Use

### Training Models

The main training script supports multiple sports and tasks:

```bash
# NFL models (currently supported)
python scripts/train_model.py --sport nfl --task classification
python scripts/train_model.py --sport nfl --task regression

# Custom test period
python scripts/train_model.py --sport nfl --task classification --test-start 2015

# Custom output directory
python scripts/train_model.py --sport nfl --task classification --out-dir my_models
```

### Making Predictions

#### Single Game Prediction

```python
from src.predict import load_model, predict_single_game

# Load model
pipeline = load_model('nfl', 'classification')

# Game features (NFL example)
game = {
    'home_id': 'KC',
    'away_id': 'BUF',
    'schedule_season': 2025,
    'schedule_week': 8,
    'schedule_playoff': False,
    'stadium_neutral': False,
    'spread_favorite': -2.5,
    'over_under_line': 50.5
}

result = predict_single_game(pipeline, game)
print(f"Prediction: {result['predictions'][0]}")  # 0 or 1
print(f"Probability: {result['probabilities'][0]:.3f}")  # Home win prob
```

#### Batch Predictions

```bash
# Create CSV with games to score
python examples/batch_scoring.py --create-sample

# Score the games
python examples/batch_scoring.py --sport nfl --task classification
```

### Web Interface

```bash
# Install streamlit
pip install streamlit

# Run the app
streamlit run examples/streamlit_app.py
```

### API Service

```bash
# Install FastAPI
pip install fastapi uvicorn

# Start the API server
uvicorn examples.api:app --reload

# Make API calls
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "sport": "nfl",
       "task": "classification",
       "home_id": "DAL",
       "away_id": "PHI",
       "schedule_season": 2025,
       "schedule_week": 5
     }'
```

## 🏈 Adding New Sports

The system is designed to be easily extensible. Here's how to add a new sport:

### Step 1: Prepare Data

Create the data directory structure:

```
data/{sport}/raw/
├── {sport}_scores.csv      # Game results
└── {sport}_teams.csv       # Team mappings
```

**Required columns for scores CSV:**
- `team_home`, `team_away`: Team names
- `score_home`, `score_away`: Game scores
- `schedule_season`: Season year
- Plus sport-specific columns

**Required columns for teams CSV:**
- `team_name`: Full team name
- `team_id`: Short ID (e.g., 'LAL', 'BOS')

### Step 2: Create Sport Configuration

Create `configs/{sport}_config.yaml`:

```yaml
name: nba
data:
  scores_file: nba_scores.csv
  teams_file: nba_teams.csv
features:
  categorical:
    - home_id
    - away_id
    - arena
  boolean:
    - playoff_flag
    - neutral_site
  numeric:
    - season
    - game_number
    - temperature
targets:
  classification: home_team_win
  regression: point_diff
preprocessing:
  team_mapping: team_name -> team_id
```

### Step 3: Implement Sport Class

Create `src/sports/{sport}.py`:

```python
from .base import BaseSport
from typing import Dict, List, Any
import pandas as pd

class NBASport(BaseSport):
    def load_data(self) -> pd.DataFrame:
        # Implement data loading logic
        pass

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Implement preprocessing (targets, cleaning, etc.)
        pass

    def get_feature_columns(self) -> Dict[str, List[str]]:
        # Return feature groupings
        pass

    def get_target_columns(self) -> Dict[str, str]:
        # Return target column names
        pass
```

### Step 4: Update Training Script

Add your sport to `scripts/train_model.py`:

```python
from sports.nba import NBASport

def get_sport_instance(sport_name: str):
    config = load_sport_config(sport_name)

    if sport_name == 'nfl':
        return NFLSport(config)
    elif sport_name == 'nba':
        return NBASport(config)  # Add this line
    # ... other sports
```

### Step 5: Test Your Implementation

```bash
# Train your new sport
python scripts/train_model.py --sport nba --task classification

# Test predictions
python examples/batch_scoring.py --sport nba --task classification
```

## 🏀 NBA Implementation Example

Let's walk through adding NBA support:

### 1. Data Files

**nba_scores.csv:**
```csv
team_home,team_away,score_home,score_away,season,game_number,playoff_flag,neutral_site,arena,temperature
Lakers,Celtics,120,115,2024,1,false,false,Crypto.com Arena,72
```

**nba_teams.csv:**
```csv
team_name,team_id
Los Angeles Lakers,LAL
Boston Celtics,BOS
```

### 2. Configuration (configs/nba_config.yaml)

```yaml
name: nba
data:
  scores_file: nba_scores.csv
  teams_file: nba_teams.csv
features:
  categorical:
    - home_id
    - away_id
    - arena
  boolean:
    - playoff_flag
    - neutral_site
  numeric:
    - season
    - game_number
    - temperature
targets:
  classification: home_team_win
  regression: point_diff
```

### 3. Sport Class (src/sports/nba.py)

```python
from .base import BaseSport
from typing import Dict, List, Any
import pandas as pd

class NBASport(BaseSport):
    def load_data(self) -> pd.DataFrame:
        self.validate_data_files()
        paths = self.get_data_paths()

        scores = pd.read_csv(paths['scores_file'])
        teams = pd.read_csv(paths['teams_file'])

        scores.columns = [c.strip() for c in scores.columns]
        teams.columns = [c.strip() for c in teams.columns]

        df = self.preprocess_data(scores)
        df = self._add_team_mappings(df, teams)
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['point_diff'] = df['score_home'] - df['score_away']
        df['home_team_win'] = (df['point_diff'] > 0).astype(int)

        # Convert booleans
        bool_cols = ['playoff_flag', 'neutral_site']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.upper().map({'TRUE': True, 'FALSE': False}).fillna(False)

        return df

    def _add_team_mappings(self, df: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
        team_map = teams.set_index('team_name')['team_id'].to_dict()
        df = df.copy()
        df['home_id'] = df['team_home'].map(team_map)
        df['away_id'] = df['team_away'].map(team_map)
        return df

    def get_feature_columns(self) -> Dict[str, List[str]]:
        return {
            'categorical': ['home_id', 'away_id', 'arena'],
            'boolean': ['playoff_flag', 'neutral_site'],
            'numeric': ['season', 'game_number', 'temperature']
        }

    def get_target_columns(self) -> Dict[str, str]:
        return {
            'classification': 'home_team_win',
            'regression': 'point_diff'
        }
```

## 🏎️ NASCAR Implementation Example

For NASCAR, the process is similar but adapted for race outcomes:

### Configuration (configs/nascar_config.yaml)

```yaml
name: nascar
data:
  races_file: nascar_races.csv
  drivers_file: nascar_drivers.csv
features:
  categorical:
    - driver_id
    - track_name
    - car_manufacturer
  boolean:
    - is_playoff_race
    - night_race
  numeric:
    - season
    - race_number
    - track_length
    - starting_position
targets:
  classification: race_win
  regression: finishing_position
```

### Key Differences for NASCAR

- **Targets**: `race_win` (1/0) instead of home_team_win, `finishing_position` instead of point_diff
- **Features**: Driver-focused rather than team-focused
- **Data Structure**: Race results instead of game scores

## 📊 Model Evaluation

After training, models are evaluated and metrics saved to JSON:

**Classification Metrics:**
- `accuracy`: Overall accuracy
- `roc_auc`: ROC AUC score
- `pred_prob_mean`: Mean predicted probability
- `confusion_matrix`: TN/FP/FN/TP counts

**Regression Metrics:**
- `mae`: Mean absolute error
- `r2`: R-squared coefficient

## 🔧 Configuration Reference

### Sport Configuration Schema

```yaml
name: sport_name
data:
  scores_file: path/to/scores.csv
  teams_file: path/to/teams.csv
features:
  categorical: [list, of, categorical, features]
  boolean: [list, of, boolean, features]
  numeric: [list, of, numeric, features]
targets:
  classification: target_column_name
  regression: target_column_name
preprocessing:
  custom_rules: any custom preprocessing rules
```

### Feature Types

- **categorical**: One-hot encoded (teams, venues, weather types)
- **boolean**: True/false features (playoffs, neutral site)
- **numeric**: Continuous features (scores, temperatures, spreads)

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "examples.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Deployment

The API can be deployed to:
- AWS Lambda + API Gateway
- Google Cloud Run
- Azure Container Instances
- Heroku

## 🤝 Contributing

1. Fork the repository
2. Add your sport implementation
3. Update documentation
4. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Original NFL data from [Pro Football Reference](https://www.pro-football-reference.com/)
- Inspired by various sports analytics projects
