"""
Batch scoring example for predicting multiple games at once.
"""
import sys
from pathlib import Path
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from predict import load_model, predict_games


def score_upcoming_games(sport: str = "nfl", task: str = "classification",
                        input_csv: str = "upcoming_games.csv",
                        output_csv: str = "scored_games.csv"):
    """
    Score a batch of upcoming games and save predictions.

    Args:
        sport: Sport name (e.g., 'nfl')
        task: Task type ('classification' or 'regression')
        input_csv: Path to CSV with games to score
        output_csv: Path to save scored results
    """
    print(f"Loading {sport} {task} model...")

    # Load the trained model
    try:
        pipeline = load_model(sport, task)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Make sure you have trained a {sport} {task} model first.")
        return

    # Load games to score
    if not Path(input_csv).exists():
        print(f"Error: Input file {input_csv} not found.")
        print("Create a CSV file with the required feature columns for your sport.")
        return

    print(f"Loading games from {input_csv}...")
    games_df = pd.read_csv(input_csv)

    print(f"Scoring {len(games_df)} games...")

    # Make predictions
    results = predict_games(pipeline, games_df)

    # Add predictions to the dataframe
    output_df = games_df.copy()
    output_df['prediction'] = results['predictions']

    if 'probabilities' in results and results['probabilities']:
        if task == 'classification':
            output_df['home_win_probability'] = results['probabilities']
        else:
            # For regression, probabilities might not be available
            pass

    # Save results
    output_df.to_csv(output_csv, index=False)
    print(f"Saved scored games to {output_csv}")

    # Print summary
    print(f"\nScoring Summary:")
    print(f"- Total games scored: {len(output_df)}")
    print(f"- Sport: {sport}")
    print(f"- Task: {task}")

    if task == 'classification':
        pred_counts = output_df['prediction'].value_counts()
        print(f"- Home wins predicted: {pred_counts.get(1, 0)}")
        print(f"- Away wins predicted: {pred_counts.get(0, 0)}")
        if 'home_win_probability' in output_df.columns:
            avg_prob = output_df['home_win_probability'].mean()
            print(".3f")
    else:
        avg_pred = output_df['prediction'].mean()
        print(".2f")


def create_sample_nfl_games():
    """Create a sample CSV of upcoming NFL games for testing."""
    sample_games = [
        {
            'home_id': 'DAL',
            'away_id': 'PHI',
            'team_favorite_id': 'DAL',
            'stadium': 'AT&T Stadium',
            'weather_detail': None,
            'schedule_season': 2025,
            'schedule_week': 5,
            'schedule_playoff': False,
            'stadium_neutral': False,
            'weather_temperature': None,
            'weather_wind_mph': None,
            'weather_humidity': None,
            'spread_favorite': -3.5,
            'over_under_line': 47.5,
        },
        {
            'home_id': 'KC',
            'away_id': 'BUF',
            'team_favorite_id': 'KC',
            'stadium': 'GEHA Field at Arrowhead',
            'weather_detail': None,
            'schedule_season': 2025,
            'schedule_week': 8,
            'schedule_playoff': False,
            'stadium_neutral': False,
            'weather_temperature': 60,
            'weather_wind_mph': 8,
            'weather_humidity': 55,
            'spread_favorite': -2.5,
            'over_under_line': 50.5,
        }
    ]

    df = pd.DataFrame(sample_games)
    df.to_csv('upcoming_games.csv', index=False)
    print("Created sample upcoming_games.csv for testing")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Batch score upcoming games')
    parser.add_argument('--sport', default='nfl', help='Sport to score (default: nfl)')
    parser.add_argument('--task', default='classification',
                       choices=['classification', 'regression'],
                       help='Task type (default: classification)')
    parser.add_argument('--input', default='upcoming_games.csv',
                       help='Input CSV file with games to score')
    parser.add_argument('--output', default='scored_games.csv',
                       help='Output CSV file for scored results')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create a sample NFL games CSV for testing')

    args = parser.parse_args()

    if args.create_sample:
        create_sample_nfl_games()
    else:
        score_upcoming_games(args.sport, args.task, args.input, args.output)
