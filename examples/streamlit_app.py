"""
Streamlit web app template for sports predictions.
Run with: streamlit run examples/streamlit_app.py
"""
import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from predict import load_model, predict_single_game


def main():
    st.set_page_config(page_title="Sports Prediction App", page_icon="🏈")

    st.title("🏈 Sports Prediction App")
    st.markdown("Predict game outcomes using machine learning models")

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    sport = st.sidebar.selectbox(
        "Select Sport",
        ["nfl"],
        help="Choose the sport for prediction"
    )

    task = st.sidebar.selectbox(
        "Select Task",
        ["classification", "regression"],
        help="Classification: predict winner, Regression: predict point differential"
    )

    # Main content
    st.header(f"{sport.upper()} {task.title()} Prediction")

    # Check if model exists
    try:
        pipeline = load_model(sport, task)
        st.success(f"✅ {sport.upper()} {task} model loaded successfully!")
    except FileNotFoundError:
        st.error(f"❌ {sport.upper()} {task} model not found. Please train the model first.")
        st.info("Run: `python scripts/train_model.py --sport nfl --task classification`")
        return

    # Input form
    st.subheader("Enter Game Details")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            home_team = st.text_input("Home Team ID", "DAL", help="e.g., DAL, KC, PHI")
            away_team = st.text_input("Away Team ID", "PHI", help="e.g., DAL, KC, PHI")

        with col2:
            season = st.number_input("Season", min_value=2000, max_value=2030, value=2025)
            week = st.number_input("Week", min_value=1, max_value=22, value=1)

        col3, col4 = st.columns(2)

        with col3:
            favorite = st.text_input("Favorite Team ID", "", help="Leave empty if no favorite")
            spread = st.number_input("Spread (Favorite)", value=0.0, step=0.5,
                                   help="Point spread for favorite (negative = favorite)")

        with col4:
            over_under = st.number_input("Over/Under", value=0.0, step=0.5,
                                       help="Total points line")

        # NFL-specific fields
        if sport == "nfl":
            st.subheader("NFL-Specific Details")

            col5, col6 = st.columns(2)
            with col5:
                stadium = st.text_input("Stadium", "", help="e.g., AT&T Stadium")
                weather = st.selectbox("Weather Detail", ["", "dome", "indoor", "outdoor"],
                                     help="Type of weather condition")

            with col6:
                temp = st.number_input("Temperature (°F)", value=70, min_value=0, max_value=120)
                wind = st.number_input("Wind Speed (mph)", value=0, min_value=0)
                humidity = st.number_input("Humidity (%)", value=50, min_value=0, max_value=100)

            playoff = st.checkbox("Playoff Game")
            neutral = st.checkbox("Neutral Site")

        # Submit button
        submitted = st.form_submit_button("Predict Outcome")

    if submitted:
        # Prepare game data
        game_data = {
            'home_id': home_team,
            'away_id': away_team,
            'schedule_season': season,
            'schedule_week': week,
            'schedule_playoff': playoff if sport == "nfl" else False,
            'stadium_neutral': neutral if sport == "nfl" else False,
        }

        # Add optional fields if provided
        if favorite:
            game_data['team_favorite_id'] = favorite
        if spread != 0:
            game_data['spread_favorite'] = spread
        if over_under != 0:
            game_data['over_under_line'] = over_under

        # Add NFL-specific fields
        if sport == "nfl":
            if stadium:
                game_data['stadium'] = stadium
            if weather:
                game_data['weather_detail'] = weather
            game_data['weather_temperature'] = temp
            game_data['weather_wind_mph'] = wind
            game_data['weather_humidity'] = humidity

        # Make prediction
        with st.spinner("Making prediction..."):
            try:
                result = predict_single_game(pipeline, game_data)

                st.success("Prediction Complete!")

                # Display results
                col_result1, col_result2 = st.columns(2)

                with col_result1:
                    st.metric("Prediction",
                             "Home Win" if result['predictions'][0] == 1 else "Away Win")

                with col_result2:
                    if result.get('probabilities'):
                        prob = result['probabilities'][0]
                        st.metric("Home Win Probability", ".1%")
                    else:
                        st.metric("Point Differential", ".1f")

                # Show input summary
                st.subheader("Game Summary")
                summary_data = {
                    "Home Team": home_team,
                    "Away Team": away_team,
                    "Season": season,
                    "Week": week,
                    "Favorite": favorite or "None",
                    "Spread": spread,
                    "Over/Under": over_under
                }

                if sport == "nfl":
                    summary_data.update({
                        "Stadium": stadium or "Unknown",
                        "Weather": weather or "Unknown",
                        "Temperature": f"{temp}°F",
                        "Wind": f"{wind} mph",
                        "Humidity": f"{humidity}%",
                        "Playoff": "Yes" if playoff else "No",
                        "Neutral Site": "Yes" if neutral else "No"
                    })

                st.json(summary_data)

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.info("Check that all required fields are filled and try again.")


if __name__ == "__main__":
    main()
