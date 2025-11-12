"""
NFL sport implementation.
"""
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from .base import BaseSport


class NFLSport(BaseSport):
    """NFL-specific sport implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def load_data(self) -> pd.DataFrame:
        """Load NFL data from CSV files."""
        self.validate_data_files()
        paths = self.get_data_paths()

        scores_path = paths.get('scores_file')
        teams_path = paths.get('teams_file')

        if not scores_path or not teams_path:
            raise ValueError("NFL config must specify scores_file and teams_file")

        # Load data
        scores = pd.read_csv(scores_path)
        teams = pd.read_csv(teams_path)

        # Basic cleaning
        scores.columns = [c.strip() for c in scores.columns]
        teams.columns = [c.strip() for c in teams.columns]

        # Apply preprocessing
        df = self.preprocess_data(scores)

        # Add team mappings
        df = self._add_team_mappings(df, teams)

        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply NFL-specific preprocessing."""
        # Create targets
        df = df.copy()
        df['point_diff'] = df['score_home'] - df['score_away']
        df['home_team_win'] = (df['point_diff'] > 0).astype(int)

        # Convert booleans
        def to_bool(s):
            if pd.api.types.is_bool_dtype(s):
                return s
            return s.astype(str).str.upper().map({'TRUE': True, 'FALSE': False}).fillna(False)

        bool_cols = ['schedule_playoff', 'stadium_neutral']
        for col in bool_cols:
            if col in df.columns:
                df[col] = to_bool(df[col])

        # Coerce numeric columns
        numeric_cols = ['schedule_season', 'schedule_week', 'weather_temperature',
                       'weather_wind_mph', 'weather_humidity', 'spread_favorite', 'over_under_line']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def _add_team_mappings(self, df: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
        """Add team ID mappings to the dataframe."""
        # Create team mapping dictionary
        team_map = teams.set_index('team_name')['team_id'].to_dict()

        # Add reverse mappings for short names
        for _, row in teams.iterrows():
            team_map.setdefault(row.get('team_name_short', row['team_name']), row['team_id'])

        def map_team(name):
            return team_map.get(name, name)

        df = df.copy()
        df['home_id'] = df['team_home'].map(map_team)
        df['away_id'] = df['team_away'].map(map_team)

        return df

    def get_feature_columns(self) -> Dict[str, List[str]]:
        """Return NFL feature column groupings."""
        return {
            'categorical': ['home_id', 'away_id', 'team_favorite_id', 'stadium', 'weather_detail'],
            'boolean': ['schedule_playoff', 'stadium_neutral'],
            'numeric': ['schedule_season', 'schedule_week', 'weather_temperature',
                       'weather_wind_mph', 'weather_humidity', 'spread_favorite', 'over_under_line']
        }

    def get_target_columns(self) -> Dict[str, str]:
        """Return NFL target column names."""
        return {
            'classification': 'home_team_win',
            'regression': 'point_diff'
        }
