"""
NASCAR sport implementation.
"""
from typing import Dict, List, Any
import pandas as pd
from pathlib import Path

from .base import BaseSport


class NASCARSport(BaseSport):
    """NASCAR-specific sport implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def load_data(self) -> pd.DataFrame:
        """Load NASCAR race results from CSV if present, else fall back to .rda files."""
        paths = self.get_data_paths()

        results_path: Path = paths.get('results_file')  # type: ignore

        # Preferred: CSV as defined in config, if it exists
        if results_path and results_path.exists():
            # If a single RDA file is configured, read it with pyreadr
            if results_path.suffix.lower() == '.rda':
                try:
                    import pyreadr  # type: ignore
                except Exception:
                    raise ImportError(
                        "pyreadr is required to read .rda files. Install with: pip install pyreadr"
                    )
                result = pyreadr.read_r(str(results_path))
                frames = [df for df in result.values() if isinstance(df, pd.DataFrame)]
                if not frames:
                    raise ValueError(f"No tabular data found inside RDA file: {results_path}")
                df = pd.concat(frames, ignore_index=True, sort=False)
            else:
                df = pd.read_csv(results_path)
            df.columns = [c.strip() for c in df.columns]
            return self.preprocess_data(df)

        # Fallback: scan for .rda files in data/nascar/raw
        raw_dir = self.data_dir / 'raw'
        rda_files = sorted(raw_dir.glob('*.rda'))

        if not rda_files:
            # If no .rda files either, raise a clear error mentioning both options
            expected = str(results_path) if results_path else '<configured results_file>'
            raise FileNotFoundError(
                f"NASCAR data not found. Expected CSV at {expected} or .rda files in {raw_dir}"
            )

        try:
            import pyreadr  # type: ignore
        except Exception:
            raise ImportError(
                "pyreadr is required to read .rda files. Install with: pip install pyreadr"
            )

        # Read all data frames from all .rda files, concatenate into a single DataFrame
        frames: List[pd.DataFrame] = []
        for rda in rda_files:
            try:
                result = pyreadr.read_r(str(rda))  # returns a dict-like of dataframes
            except Exception as e:
                raise RuntimeError(f"Failed to read RDA file {rda}: {e}")

            for name, frame in result.items():
                if not isinstance(frame, pd.DataFrame):
                    continue
                # Normalize column names
                frame = frame.copy()
                frame.columns = [str(c).strip() for c in frame.columns]
                frames.append(frame)

        if not frames:
            raise ValueError(f"No tabular data found inside RDA files: {[str(p) for p in rda_files]}")

        # Heuristic: prefer dataframes that contain key NASCAR columns; otherwise concatenate all
        def has_core_cols(df: pd.DataFrame) -> bool:
            core = {'year', 'track', 'fin', 'driver'}
            return core.issubset(set(map(str.lower, df.columns))) or core.issubset(set(df.columns))

        preferred = [f for f in frames if has_core_cols(f)]
        if preferred:
            df = pd.concat(preferred, ignore_index=True, sort=False)
        else:
            df = pd.concat(frames, ignore_index=True, sort=False)

        return self.preprocess_data(df)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply NASCAR-specific preprocessing and target creation."""
        df = df.copy()

        # Ensure expected columns exist; create if missing
        # Map dataset 'fin' to standardized target 'finishing_position'
        if 'fin' in df.columns and 'finishing_position' not in df.columns:
            df['finishing_position'] = pd.to_numeric(df['fin'], errors='coerce')

        # Classification target: race win flag
        if 'race_win' not in df.columns:
            df['race_win'] = (df['finishing_position'] == 1).astype(int)

        # Standardize season column expected by generic trainer
        if 'schedule_season' not in df.columns:
            # Use 'year' column from dataset
            if 'year' in df.columns:
                df['schedule_season'] = pd.to_numeric(df['year'], errors='coerce')
            else:
                # Fallback: try to infer from race identifier if present
                df['schedule_season'] = pd.NA

        # Coerce numerics
        numeric_cols = [
            'year', 'race_num', 'start', 'car_num', 'laps', 'laps_led',
            'points', 'stage_1', 'stage_2', 'stage_3_or_duel', 'stage_points',
            'finishing_position', 'schedule_season'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Fill simple categorical text fields with strings
        for col in ['driver', 'track', 'track_type', 'manu', 'team_name', 'status']:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('Unknown')

        return df

    def get_feature_columns(self) -> Dict[str, List[str]]:
        """Return NASCAR feature columns as configured."""
        feats = self.config.get('features', {})
        return {
            'categorical': feats.get('categorical', []),
            'boolean': feats.get('boolean', []),
            'numeric': feats.get('numeric', []),
        }

    def get_target_columns(self) -> Dict[str, str]:
        """Return target columns mapping for NASCAR."""
        t = self.config.get('targets', {})
        # Defaults if not present
        classification = t.get('classification', 'race_win')
        regression = t.get('regression', 'finishing_position')
        return {'classification': classification, 'regression': regression}
