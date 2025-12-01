import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sports.nascar import NASCARSport

class SimulationEngine:
    def __init__(self, sport: NASCARSport):
        self.sport = sport
        self.model = None # Placeholder for ML model if needed
        
    def calculate_driver_strength(self, driver_id: str, year: int, track_type: str) -> float:
        """
        Calculate a driver's strength rating based on recent history and track type.
        Returns a float where 1.0 is average, >1.0 is better.
        """
        # Get stats for the specific year
        stats = self.sport.get_entity_stats(driver_id, year)
        
        if not stats or 'stats' not in stats:
            return 0.5 # Default low rating for unknown drivers
            
        # Base rating on average finish (inverted, lower is better)
        # Avg finish 15 is roughly 1.0 strength
        avg_finish = float(stats['stats'].get('Avg Finish', 20.0))
        base_strength = 20.0 / max(avg_finish, 1.0)
        
        # Adjust for track type if available
        if 'splits' in stats and track_type in stats['splits']:
            split_avg = float(stats['splits'][track_type].get('Avg Finish', avg_finish))
            track_factor = avg_finish / max(split_avg, 1.0)
            base_strength *= track_factor
            
        return base_strength

    def _simulate_single_race(self, strengths: Dict[str, float]) -> List[str]:
        """
        Simulate a single race using pre-calculated strengths.
        """
        # Add randomness
        # We'll use a Gumbel distribution for the random component to simulate race positions
        # Score = Strength + Noise
        scores = {}
        for driver, strength in strengths.items():
            noise = np.random.gumbel(0, 0.5) # Tunable noise parameter
            scores[driver] = strength + noise
            
        # Sort by score descending (higher score = better finish)
        sorted_drivers = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_drivers

    def run_monte_carlo(self, drivers: List[str], year: int, track_type: str, num_simulations: int = 1000) -> Dict[str, Any]:
        """
        Run multiple simulations and aggregate results.
        """
        # Pre-calculate strengths to avoid repeated lookups
        strengths = {}
        for driver in drivers:
            strengths[driver] = self.calculate_driver_strength(driver, year, track_type)

        results = {driver: [] for driver in drivers}
        
        for _ in range(num_simulations):
            finishing_order = self._simulate_single_race(strengths)
            for pos, driver in enumerate(finishing_order):
                results[driver].append(pos + 1) # 1-based finish position
                
        # Aggregate stats
        aggregated = []
        for driver, finishes in results.items():
            finishes_array = np.array(finishes)
            agg = {
                "driver": driver,
                "avg_finish": float(np.mean(finishes_array)),
                "win_prob": float(np.mean(finishes_array == 1)),
                "top_5_prob": float(np.mean(finishes_array <= 5)),
                "top_10_prob": float(np.mean(finishes_array <= 10)),
                "best_finish": int(np.min(finishes_array)),
                "worst_finish": int(np.max(finishes_array))
            }
            aggregated.append(agg)
            
        # Sort by win probability descending
        aggregated.sort(key=lambda x: x['win_prob'], reverse=True)
        
        return {
            "metadata": {
                "year": year,
                "track_type": track_type,
                "simulations": num_simulations,
                "driver_count": len(drivers)
            },
            "results": aggregated
        }
