import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import project_model  # our ML backend

# -----------------------------
# Page Config + Theme Styles
# -----------------------------
st.set_page_config(
    page_title="ODI Batting Prediction Dashboard",
    page_icon="🏏",
    layout="wide",
)

# ---- Custom styling ----
st.markdown(
    """
    <style>
    .main {
        background-color: #020617;
        color: #e5e7eb;
    }
    section[data-testid="stSidebar"] {
        background-color: #030712;
        border-right: 1px solid #1f2937;
    }
    .metric-card {
        padding: 1rem 1.5rem;
        border-radius: 0.75rem;
        background: #020617;
        border: 1px solid #1f2937;
        box-shadow: 0 18px 45px rgba(15,23,42,0.9);
    }
    .metric-label {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-bottom: 0.15rem;
    }
    .metric-value {
        font-size: 1.15rem;
        font-weight: 600;
    }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
    }
    .badge-excellent { background: #16a34a; }
    .badge-good { background: #22c55e; }
    .badge-moderate { background: #eab308; }
    .badge-low { background: #ef4444; }
    </style>
    """,
    unsafe_allow_html=True,
)

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
# Main header + KPI Cards
# -----------------------------
st.title("🏏 ODI Performance Intelligence")
st.subheader(f"Match Context: vs {target_opp} at {target_ground}")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">ML Model</div>
            <div class="metric-value">{MODEL_NAME}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{MODEL_ACC:.2%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Players Evaluated</div>
            <div class="metric-value">{len(DF_FULL['Player'].unique())}</div>
        </div>
        """,
        unsafe_allow_html=True,
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
    # Overview Display Table
    # -------------------------
    st.subheader("📋 Ranked Player Performance")
    st.dataframe(
        ranked_df.head(top_n),
        use_container_width=True,
    )

    # -------------------------
    # Bar Chart of Avg Runs
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
    # Actionable tips (Styled Expanders)
    # -------------------------
    st.subheader("🧠 Actionable Batting Insights")

    category_colors = {
        "Excellent": "badge-excellent",
        "Good": "badge-good",
        "Moderate": "badge-moderate",
        "Low": "badge-low",
    }

    for idx, row in ranked_df.head(top_n).iterrows():
        cat = str(row["Predicted Category"])
        base_cat = "Low"
        if "Excellent" in cat:
            base_cat = "Excellent"
        elif "Good" in cat:
            base_cat = "Good"
        elif "Moderate" in cat:
            base_cat = "Moderate"

        badge_class = category_colors.get(base_cat, "badge-low")

        header_html = (
            f"<span><b>{int(idx)}. {row['Player']}</b></span> "
            f"&nbsp;&nbsp;<span class='badge {badge_class}'>{cat}</span>"
        )

        with st.expander("", expanded=False):
            st.markdown(header_html, unsafe_allow_html=True)
            st.write(f"**Role:** {row['Role']}")
            st.write(f"**Avg Runs:** {row['Avg Runs']}")
            col_name = f"Actual Runs vs {target_opp}"
            if col_name in row:
                st.write(f"**Actual vs {target_opp}:** {row[col_name]}")
            st.write(f"**Insight:** {row['Actionable Tip']}")

    # -------------------------
    # Bowling impact section
    # -------------------------
    if hasattr(project_model, "get_bowling_impact_df"):
        st.markdown("---")
        st.subheader(f"🎯 Bowling Impact vs {target_opp}")

        bowling_df = project_model.get_bowling_impact_df(target_opp)

        if bowling_df is None or bowling_df.empty:
            st.info("No bowling profiles available for this opposition.")
        else:
            # Table
            st.subheader("📋 Bowling Impact – Key Bowlers & All-rounders")
            st.dataframe(bowling_df.head(top_n), use_container_width=True)

            # Bar chart for Impact Score
            st.subheader("📈 Bowling Impact Score")
            bowl_chart = bowling_df.head(top_n).sort_values("Impact Score", ascending=True)

            fig_bowl, ax_bowl = plt.subplots(figsize=(8, max(4, len(bowl_chart) * 0.4)))
            ax_bowl.barh(bowl_chart["Player"], bowl_chart["Impact Score"])
            ax_bowl.set_xlabel("Impact Score")
            ax_bowl.set_ylabel("Player")
            ax_bowl.set_title(f"Bowling Impact vs {target_opp}")

            for i, v in enumerate(bowl_chart["Impact Score"]):
                ax_bowl.text(v, i, f"{v:.2f}", va="center")

            st.pyplot(fig_bowl)

            # Bowling insights using badges
            st.subheader("🧠 Bowling Strategy Insights")

            bowl_badge_map = {
                "Strike": "badge-excellent",
                "Control": "badge-good",
                "Support": "badge-moderate",
            }

            for idx, row in bowling_df.head(top_n).iterrows():
                cat_text = str(row["Impact Category"])
                if "Strike" in cat_text:
                    base_cat = "Strike"
                elif "Control" in cat_text:
                    base_cat = "Control"
                else:
                    base_cat = "Support"

                badge_class = bowl_badge_map.get(base_cat, "badge-moderate")

                header_html = (
                    f"<span><b>{int(idx)}. {row['Player']} ({row['Role']})</b></span> "
                    f"&nbsp;&nbsp;<span class='badge {badge_class}'>{cat_text}</span>"
                )

                with st.expander("", expanded=False):
                    st.markdown(header_html, unsafe_allow_html=True)
                    st.write(f"**Type / Phase:** {row['Type']} | {row['Phase']}")
                    st.write(
                        f"**Overs / Match:** {row['Overs / Match']}  "
                        f"| **Wickets / Match:** {row['Wickets / Match']}  "
                        f"| **Economy:** {row['Economy']}"
                    )
                    st.write(f"**Impact Score:** {row['Impact Score']}")
                    st.write(f"**Strategy Tip:** {row['Bowling Tip']}")
