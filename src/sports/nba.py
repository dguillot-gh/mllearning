"""
NBA sport implementation.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from .base import BaseSport


class NBASport(BaseSport):
    """NBA-specific sport implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Override data_dir to use mllearning/data path (where NBA data actually is)
        project_root = Path(__file__).resolve().parents[2]
        self.data_dir = project_root / 'mllearning' / 'data' / self.name
        self.df = None  # Cache loaded data
        self._team_name_map = None  # Cache team abbrev -> full name mapping
        self._abbrev_to_name = None  # OKC -> Oklahoma City Thunder

    def _load_team_names(self) -> None:
        """Load team abbreviation to full name mapping."""
        if self._team_name_map is not None:
            return
            
        abbrev_file = self.data_dir / 'raw' / 'Team Abbrev.csv'
        if abbrev_file.exists():
            try:
                team_df = pd.read_csv(abbrev_file)
                # Get latest mapping for each abbreviation
                team_df = team_df.sort_values('season', ascending=False)
                team_df = team_df.drop_duplicates('abbreviation', keep='first')
                
                # Create abbrev -> full name mapping
                self._abbrev_to_name = dict(zip(team_df['abbreviation'], team_df['team']))
                # Create full name -> abbrev mapping
                self._team_name_map = dict(zip(team_df['team'], team_df['abbreviation']))
            except Exception as e:
                print(f"Warning: Could not load team names: {e}")
                self._abbrev_to_name = {}
                self._team_name_map = {}
        else:
            self._abbrev_to_name = {}
            self._team_name_map = {}

    def _get_full_team_name(self, abbrev: str) -> str:
        """Convert team abbreviation to full name."""
        self._load_team_names()
        return self._abbrev_to_name.get(abbrev, abbrev)
    
    def _get_team_abbrev(self, full_name: str) -> str:
        """Convert full team name to abbreviation."""
        self._load_team_names()
        return self._team_name_map.get(full_name, full_name)

    def load_data(self) -> pd.DataFrame:
        """Load NBA data from CSV files."""
        if self.df is not None:
            return self.df
            
        # Load player per game stats (main dataset)
        player_file = self.data_dir / 'raw' / 'Player Per Game.csv'
        if not player_file.exists():
            raise FileNotFoundError(f"NBA player data not found at {player_file}")
            
        df = pd.read_csv(player_file)
        
        # Basic cleaning
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        
        # Apply preprocessing
        df = self.preprocess_data(df)
        
        self.df = df
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply NBA-specific preprocessing."""
        df = df.copy()
        
        # Standardize column names
        rename_map = {
            'player_id': 'player_id',
            'player': 'player_name',
            'team': 'team_abbrev',
            'pos': 'position',
            'g': 'games',
            'gs': 'games_started',
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        # Coerce numeric columns
        numeric_cols = ['season', 'age', 'games', 'games_started', 'mp_per_game',
                       'fg_per_game', 'fga_per_game', 'fg_percent',
                       'x3p_per_game', 'x3pa_per_game', 'x3p_percent',
                       'ft_per_game', 'fta_per_game', 'ft_percent',
                       'orb_per_game', 'drb_per_game', 'trb_per_game',
                       'ast_per_game', 'stl_per_game', 'blk_per_game',
                       'tov_per_game', 'pf_per_game', 'pts_per_game']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'schedule_season' not in df.columns and 'season' in df.columns:
            df['schedule_season'] = df['season']
        elif 'schedule_season' not in df.columns:
            df['schedule_season'] = 2024 # Default fallback

        # Create target column 'is_all_star' (approximation: >20 PPG)
        if 'is_all_star' not in df.columns:
            if 'pts_per_game' in df.columns:
                df['is_all_star'] = (df['pts_per_game'] >= 20.0).astype(int)
            else:
                # Fallback if no points data
                df['is_all_star'] = 0

        return df

    def get_feature_columns(self) -> Dict[str, List[str]]:
        """Return NBA feature column groupings."""
        return {
            'categorical': ['player_name', 'team_abbrev', 'position'],
            'boolean': [],
            'numeric': ['season', 'age', 'games', 'games_started', 'mp_per_game',
                       'fg_per_game', 'fga_per_game', 'fg_percent',
                       'x3p_per_game', 'x3pa_per_game', 'x3p_percent',
                       'ft_per_game', 'fta_per_game', 'ft_percent',
                       'orb_per_game', 'drb_per_game', 'trb_per_game',
                       'ast_per_game', 'stl_per_game', 'blk_per_game',
                       'tov_per_game', 'pf_per_game', 'pts_per_game']
        }

    def get_target_columns(self) -> Dict[str, str]:
        """Return NBA target column names."""
        return {
            'classification': 'is_all_star',  # Can be derived
            'regression': 'pts_per_game'
        }

    def get_entities(self) -> List[str]:
        """Return list of all players."""
        df = self.load_data()
        if 'player_name' in df.columns:
            return sorted(df['player_name'].dropna().unique().tolist())
        elif 'player' in df.columns:
            return sorted(df['player'].dropna().unique().tolist())
        return []

    def get_teams(self) -> List[str]:
        """Return list of all teams with FULL NAMES."""
        df = self.load_data()
        col = 'team_abbrev' if 'team_abbrev' in df.columns else 'team'
        if col in df.columns:
            # Get unique abbreviations
            abbrevs = df[col].dropna().unique().tolist()
            # Convert to full names
            full_names = []
            for abbrev in abbrevs:
                full_name = self._get_full_team_name(abbrev)
                full_names.append(full_name)
            return sorted(set(full_names))
        return []

    def get_drivers(self, team_id: Optional[str] = None) -> List[str]:
        """Return list of players (NBA equivalent of 'drivers')."""
        return self.get_players(team_id)

    def get_players(self, team_id: Optional[str] = None) -> List[str]:
        """Return list of players, optionally filtered by team (accepts full name or abbrev)."""
        df = self.load_data()
        
        player_col = 'player_name' if 'player_name' in df.columns else 'player'
        team_col = 'team_abbrev' if 'team_abbrev' in df.columns else 'team'
        
        if team_id and team_col in df.columns:
            # Try to match - could be abbreviation or full name
            self._load_team_names()
            
            # Check if it's a full name and convert to abbrev
            if team_id in self._team_name_map:
                team_abbrev = self._team_name_map[team_id]
            else:
                team_abbrev = team_id  # Assume it's already an abbreviation
            
            df = df[df[team_col] == team_abbrev]
            
        if player_col in df.columns:
            return sorted(df[player_col].dropna().unique().tolist())
        return []

    def get_entity_stats(self, entity_id: str, year: Optional[int] = None) -> Dict[str, Any]:
        """Return comprehensive stats for a player."""
        df = self.load_data()
        
        player_col = 'player_name' if 'player_name' in df.columns else 'player'
        
        # Filter for this player
        player_data = df[df[player_col] == entity_id].copy()
        
        if player_data.empty:
            return {'stats': {}, 'splits': {}, 'history': [], 'years': []}
        
        # Sort by season descending
        if 'season' in player_data.columns:
            player_data = player_data.sort_values('season', ascending=False)
            years = sorted(player_data['season'].dropna().unique().tolist(), reverse=True)
        else:
            years = []
        
        # Filter by year if specified
        if year and 'season' in player_data.columns:
            player_data = player_data[player_data['season'] == year]
        
        if player_data.empty:
            return {'stats': {}, 'splits': {}, 'history': [], 'years': years}
        
        # Get latest season data for stats
        latest = player_data.iloc[0]
        
        # Get full team name
        team_abbrev = latest.get('team_abbrev', latest.get('team', 'N/A'))
        team_full = self._get_full_team_name(team_abbrev) if pd.notna(team_abbrev) else 'N/A'
        
        # Build stats dictionary
        stats = {
            "Season": int(latest.get('season', 0)) if pd.notna(latest.get('season')) else 'N/A',
            "Team": team_full,
            "Position": latest.get('position', latest.get('pos', 'N/A')),
            "Age": int(latest.get('age', 0)) if pd.notna(latest.get('age')) else 'N/A',
            "Games": int(latest.get('games', latest.get('g', 0))) if pd.notna(latest.get('games', latest.get('g'))) else 0,
            "Games Started": int(latest.get('games_started', latest.get('gs', 0))) if pd.notna(latest.get('games_started', latest.get('gs'))) else 0,
            "PPG": f"{latest.get('pts_per_game', 0):.1f}" if pd.notna(latest.get('pts_per_game')) else 'N/A',
            "RPG": f"{latest.get('trb_per_game', 0):.1f}" if pd.notna(latest.get('trb_per_game')) else 'N/A',
            "APG": f"{latest.get('ast_per_game', 0):.1f}" if pd.notna(latest.get('ast_per_game')) else 'N/A',
            "SPG": f"{latest.get('stl_per_game', 0):.1f}" if pd.notna(latest.get('stl_per_game')) else 'N/A',
            "BPG": f"{latest.get('blk_per_game', 0):.1f}" if pd.notna(latest.get('blk_per_game')) else 'N/A',
            "FG%": f"{latest.get('fg_percent', 0)*100:.1f}%" if pd.notna(latest.get('fg_percent')) else 'N/A',
            "3P%": f"{latest.get('x3p_percent', 0)*100:.1f}%" if pd.notna(latest.get('x3p_percent')) else 'N/A',
            "FT%": f"{latest.get('ft_percent', 0)*100:.1f}%" if pd.notna(latest.get('ft_percent')) else 'N/A',
            "MPG": f"{latest.get('mp_per_game', 0):.1f}" if pd.notna(latest.get('mp_per_game')) else 'N/A',
        }
        
        # Splits by team (using full names)
        team_col = 'team_abbrev' if 'team_abbrev' in player_data.columns else 'team'
        splits = {}
        if team_col in player_data.columns:
            for team, group in player_data.groupby(team_col):
                team_name = self._get_full_team_name(str(team))
                splits[team_name] = {
                    "Seasons": len(group),
                    "Avg PPG": f"{group['pts_per_game'].mean():.1f}" if 'pts_per_game' in group.columns else 'N/A',
                    "Avg RPG": f"{group['trb_per_game'].mean():.1f}" if 'trb_per_game' in group.columns else 'N/A',
                    "Avg APG": f"{group['ast_per_game'].mean():.1f}" if 'ast_per_game' in group.columns else 'N/A',
                }
        
        # History (season by season)
        history = []
        for _, row in player_data.head(10).iterrows():
            team_abbrev_hist = row.get('team_abbrev', row.get('team', 'N/A'))
            team_full_hist = self._get_full_team_name(team_abbrev_hist) if pd.notna(team_abbrev_hist) else 'N/A'
            history.append({
                "Season": int(row.get('season', 0)) if pd.notna(row.get('season')) else 'N/A',
                "Team": team_full_hist,
                "Games": int(row.get('games', row.get('g', 0))) if pd.notna(row.get('games', row.get('g'))) else 0,
                "PPG": f"{row.get('pts_per_game', 0):.1f}" if pd.notna(row.get('pts_per_game')) else 'N/A',
                "RPG": f"{row.get('trb_per_game', 0):.1f}" if pd.notna(row.get('trb_per_game')) else 'N/A',
                "APG": f"{row.get('ast_per_game', 0):.1f}" if pd.notna(row.get('ast_per_game')) else 'N/A',
            })
        
        return {
            "stats": stats,
            "splits": splits,
            "history": history,
            "years": years
        }

    def get_year_range(self) -> Dict[str, int]:
        """Return the min and max season years available in the data."""
        df = self.load_data()
        if 'season' in df.columns:
            return {
                "min_year": int(df['season'].min()),
                "max_year": int(df['season'].max())
            }
        return {"min_year": 1947, "max_year": 2026}
