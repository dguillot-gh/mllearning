
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DatasetManager:
    """Manages dynamic dataset configurations."""
    
    def __init__(self, data_root: Path):
        self.config_path = data_root / 'datasets.json'
        self.data_root = data_root
        self._load_config()
        
    def _load_config(self):
        """Load datasets configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                logger.error(f"Error loading datasets.json: {e}")
                self.config = {}
        else:
            self.config = {}
            
        # Ensure defaults are populated
        self._ensure_defaults()

    def _ensure_defaults(self):
        """Ensure default datasets are configured if missing."""
        defaults = {
            "nfl": "tobycrabtree/nfl-scores-and-betting-data",
            "nba": "sumitrodatta/nba-aba-baa-stats"
        }
        
        updated = False
        for sport, dataset_id in defaults.items():
            if sport not in self.config:
                self.config[sport] = []
                
            # If no datasets configured for this sport, add the default
            # This preserves the "original" behavior while allowing additions or removals
            if not self.config[sport]:
                entry = {
                    "id": dataset_id,
                    "type": "kaggle",
                    "added_at": datetime.utcnow().isoformat(),
                    "last_updated": None
                }
                self.config[sport].append(entry)
                updated = True
                
        if updated:
            self._save_config()

    def _save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving datasets.json: {e}")
            
    def get_datasets(self, sport: str) -> List[Dict[str, Any]]:
        """Get list of configured datasets for a sport."""
        return self.config.get(sport, [])
    
    def add_dataset(self, sport: str, dataset_id: str, type: str = "kaggle") -> Dict[str, Any]:
        """Add a new dataset configuration."""
        if sport not in self.config:
            self.config[sport] = []
            
        # Check if already exists
        for ds in self.config[sport]:
            if ds['id'] == dataset_id:
                return {"success": False, "message": "Dataset already configured"}
                
        # Validate dataset exists (if Kaggle)
        if type == "kaggle":
             if not self._validate_kaggle_dataset(dataset_id):
                 return {"success": False, "message": "Invalid Kaggle dataset ID or not accessible"}
        
        entry = {
            "id": dataset_id,
            "type": type,
            "added_at": datetime.utcnow().isoformat(),
            "last_updated": None
        }
        
        self.config[sport].append(entry)
        self._save_config()
        return {"success": True, "entry": entry}

    def remove_dataset(self, sport: str, dataset_id: str) -> bool:
        """Remove a dataset configuration."""
        if sport not in self.config:
            return False
            
        original_len = len(self.config[sport])
        self.config[sport] = [ds for ds in self.config[sport] if ds['id'] != dataset_id]
        
        if len(self.config[sport]) < original_len:
            self._save_config()
            return True
        return False

    def update_timestamp(self, sport: str, dataset_id: str):
        """Update the last_updated timestamp for a dataset."""
        if sport in self.config:
            for ds in self.config[sport]:
                if ds['id'] == dataset_id:
                    ds['last_updated'] = datetime.utcnow().isoformat()
                    self._save_config()
                    return

    def _validate_kaggle_dataset(self, dataset: str) -> bool:
        """Check if Kaggle dataset exists using generic shell command or python API."""
        try:
            # We can reuse similar logic to KaggleDataSource
            # Assuming Kaggle API credentials are set in env
            import subprocess
            result = subprocess.run(
                ["python", "-m", "kaggle", "datasets", "status", dataset],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
