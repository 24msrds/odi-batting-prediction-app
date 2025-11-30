import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import project_model  # our ML backend


# -----------------------------
# Load trained objects
# -----------------------------
trained = project_model.get_trained_objects()

MODEL = trained["best_model"]
X_COLS = trained["x_columns"]
LE = trained["label_encoder"]
DF_FULL = trained["df_full"]
PLAYER_ROLES = trained["player_roles"]
PLAYER_NATIONALITY = trained["player_nationality"]
MODEL_ACC = trained["model_accuracy"]
MODEL_NAME = trained["model_name"]

# -----------------------------
# Clean dropdown values
# -----------------------------

# Oppositions: use team codes from PLAYER_NATIONALITY (IND, AUS, ENG, etc.)
OPPONENTS = sorted(set(PLAYER_NATIONALITY.values()))

# Grounds: use a static list of popular venues
GROUNDS = [
    "Delhi", "Mumbai", "Chennai", "Kolkata", "Ahmedabad",
    "Melbourne", "Sydney", "Adelaide", "Perth",
    "Lords", "The Oval", "Birmingham",
    "Abu Dhabi", "Dubai", "Sharjah",
    "Johannesburg", "Cape Town", "Durban",
    "Wellington", "Auckland", "Christchurch",
]
DEFAULT_GROUND = "Neutral Venue"


# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="ODI Batting Prediction Dashboard",
    page_icon="🏏",
    layout="wide",
)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("⚙ Match Settings")

target_opp = st.sidebar.selectbox(
    "Select Opposition (Team)",
    options=OPPONENTS,
    index=0,
)

ground_choice = st.sidebar.selectbox(
    "Select Ground",
    options=[DEFAULT_GROUND] + GROUNDS,
)

target_ground = ground_choice

top_n = st.sidebar.slider(
    "Number of top players to display",
    min_value=5,
    max_value=25,
    value=10,
    step=1,
)

# -----------------------------
# Main header
# -----------------------------
st.title("🏏 ODI Batting Performance Prediction Dashboard")
st.subheader(f"vs {target_opp} at {target_ground}")

st.markdown(
    f"**Model:** {MODEL_NAME} &nbsp;&nbsp;|&nbsp;&nbsp; "
    f"**Accuracy:** {MODEL_ACC:.2%}"
)

st.markdown("---")

# -----------------------------
# Run model analysis
# -----------------------------
ranked_df = project_model.analyze_and_rank_players(
    MODEL,
    X_COLS,
    LE,
    target_opp,      # e.g. 'IND'
    target_ground,   # e.g. 'Delhi'
)

if ranked_df.empty:
    st.warning("No player performance data available for this selection.")
else:
    # -------------------------
    # Overview table
    # -------------------------
    st.subheader("📋 Ranked Player Performance")
    st.dataframe(
        ranked_df.head(top_n),
        use_container_width=True,
    )

    # -------------------------
    # Bar chart of Avg Runs
    # -------------------------
    st.subheader("📊 Average Runs (Top Players)")
    chart_df = ranked_df.head(top_n).sort_values("Avg Runs", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(chart_df) * 0.4)))
    ax.barh(chart_df["Player"], chart_df["Avg Runs"])
    ax.set_xlabel("Average Runs")
    ax.set_ylabel("Player")
    ax.set_title(f"Average Runs vs {target_opp} at {target_ground}")

    for i, v in enumerate(chart_df["Avg Runs"]):
        ax.text(v, i, f"{v:.1f}", va="center")

    st.pyplot(fig)

    # -------------------------
    # Actionable tips
    # -------------------------
    st.subheader("🧠 Actionable Batting Insights")

    for idx, row in ranked_df.head(top_n).iterrows():
        with st.expander(f"{int(idx)}. {row['Player']} — {row['Predicted Category']}"):
            st.write(f"**Role:** {row['Role']}")
            st.write(f"**Avg Runs:** {row['Avg Runs']}")
            st.write(row["Actionable Tip"])
