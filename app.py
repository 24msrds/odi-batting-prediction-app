import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import project_model  # our ML backend

# -----------------------------
# Page Config + Theme Styles
# -----------------------------
st.set_page_config(
    page_title="ODI Performance Studio",
    page_icon="🏏",
    layout="wide",
)

# ---- Custom styling ----
st.markdown(
    """
    <style>

    /* -------- GLOBAL PAGE BACKGROUND & TEXT -------- */
    .main, .block-container {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 35%, #93c5fd 70%, #e0f2fe 100%);
        color: #0f172a;
    }

    body {
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, "Roboto", sans-serif;
    }

    /* Center content a bit nicer */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* -------- SIDEBAR -------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1d4ed8 55%, #0ea5e9 100%);
        border-right: 2px solid rgba(191, 219, 254, 0.6);
        color: #e5f4ff;
    }

    section[data-testid="stSidebar"] .css-1d391kg,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #e0f2fe !important;
        font-weight: 500;
    }

    /* -------- KPI Metric Cards -------- */
    .metric-card {
        padding: 1.1rem 1.6rem;
        border-radius: 1rem;
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.6);
        box-shadow: 0 14px 40px rgba(15, 23, 42, 0.25);
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 24px 60px rgba(30, 64, 175, 0.45);
        border-color: rgba(59, 130, 246, 0.9);
    }
    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        margin-bottom: 0.15rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f172a;
    }

    /* -------- CHIP BADGES (Opposition, Venue, Top N) -------- */
    .chip {
        display: inline-flex;
        align-items: center;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 8px;
        background: linear-gradient(120deg, #0ea5e9, #2563eb);
        border: 1px solid rgba(191, 219, 254, 0.9);
        color: #f9fafb;
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.45);
    }

    .chip-label {
        opacity: 0.9;
        margin-right: 6px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.68rem;
    }

    /* -------- CATEGORY BADGES -------- */
    .badge {
        display: inline-block;
        padding: 3px 11px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        color: white;
    }
    .badge-excellent { background: linear-gradient(90deg, #22c55e, #16a34a); }
    .badge-good { background: linear-gradient(90deg, #0ea5e9, #2563eb); }
    .badge-moderate { background: linear-gradient(90deg, #facc15, #eab308); color:#111827; }
    .badge-low { background: linear-gradient(90deg, #fb7185, #ef4444); }

    /* -------- SECTION TITLES -------- */
    .section-heading {
        font-size: 1.05rem;
        font-weight: 650;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        background: rgba(239, 246, 255, 0.95);
        border: 1px solid rgba(59, 130, 246, 0.8);
        margin-bottom: 0.4rem;
        color: #0f172a;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.2);
    }

    /* -------- TABS STYLING -------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 0.55rem 1.1rem;
        background: rgba(219, 234, 254, 0.9);
        color: #1e3a8a;
        font-weight: 500;
        border: 1px solid transparent;
        transition: 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(120deg, #60a5fa, #3b82f6);
        color: white;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(120deg, #2563eb, #1d4ed8);
        color: white;
        border-color: rgba(191, 219, 254, 0.9);
        font-weight: 650;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.45);
    }

    /* -------- DATAFRAME TABLE -------- */
    .dataframe {
        border-radius: 0.75rem;
        overflow: hidden;
        background: #eff6ff;
    }

    /* -------- EXPANDER STYLING -------- */
    .streamlit-expanderHeader {
        background: #dbeafe !important;
        color: #0f172a !important;
        font-weight: 650;
        border-radius: 0.5rem !important;
    }

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
st.title("🏏 ODI Performance Studio")

st.markdown(
    f"""
    <div>
        <span class="chip">
            <span class="chip-label">Opposition</span>
            <span>{target_opp}</span>
        </span>
        <span class="chip">
            <span class="chip-label">Venue</span>
            <span>{target_ground}</span>
        </span>
        <span class="chip">
            <span class="chip-label">Top N</span>
            <span>{top_n} players</span>
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("")

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
            <div class="metric-label">Validation Accuracy</div>
            <div class="metric-value">{MODEL_ACC:.2%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Unique Players Analysed</div>
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
    target_opp,
    target_ground,
)

# Pre-compute bowling DF if available
bowling_df = None
if hasattr(project_model, "get_bowling_impact_df"):
    bowling_df = project_model.get_bowling_impact_df(target_opp)

# Tabs for Batting / Bowling
bat_tab, bowl_tab = st.tabs(["🏏 Batting Analysis", "🎯 Bowling Analysis"])

# -------------------------------------------------
# BAT TAB
# -------------------------------------------------
with bat_tab:
    st.markdown('<div class="section-heading">📋 Ranked Player Performance</div>', unsafe_allow_html=True)

    if ranked_df.empty:
        st.warning("No player performance data available for this selection.")
    else:
        st.dataframe(
            ranked_df.head(top_n),
            use_container_width=True,
        )

        st.markdown('<div class="section-heading">📊 Average Runs (Top Players)</div>', unsafe_allow_html=True)

        chart_df = ranked_df.head(top_n).sort_values("Avg Runs", ascending=True)

        fig, ax = plt.subplots(figsize=(8, max(4, len(chart_df) * 0.4)))
        ax.barh(chart_df["Player"], chart_df["Avg Runs"])
        ax.set_xlabel("Average Runs")
        ax.set_ylabel("Player")
        ax.set_title(f"Average Runs vs {target_opp} at {target_ground}")

        for i, v in enumerate(chart_df["Avg Runs"]):
            ax.text(v, i, f"{v:.1f}", va="center")

        st.pyplot(fig)

        # Actionable tips
        st.markdown('<div class="section-heading">🧠 Actionable Batting Insights</div>', unsafe_allow_html=True)

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

# -------------------------------------------------
# BOWLING TAB
# -------------------------------------------------
with bowl_tab:
    st.markdown('<div class="section-heading">🎯 Bowling Impact vs Opposition</div>', unsafe_allow_html=True)

    if bowling_df is None or bowling_df.empty:
        st.info("No bowling profiles available for this opposition yet.")
    else:
        st.dataframe(bowling_df.head(top_n), use_container_width=True)

        st.markdown('<div class="section-heading">📈 Bowling Impact Score</div>', unsafe_allow_html=True)

        bowl_chart = bowling_df.head(top_n).sort_values("Impact Score", ascending=True)

        fig_bowl, ax_bowl = plt.subplots(figsize=(8, max(4, len(bowl_chart) * 0.4)))
        ax_bowl.barh(bowl_chart["Player"], bowl_chart["Impact Score"])
        ax_bowl.set_xlabel("Impact Score")
        ax_bowl.set_ylabel("Player")
        ax_bowl.set_title(f"Bowling Impact vs {target_opp}")

        for i, v in enumerate(bowl_chart["Impact Score"]):
            ax_bowl.text(v, i, f"{v:.2f}", va="center")

        st.pyplot(fig_bowl)

        st.markdown('<div class="section-heading">🧠 Bowling Strategy Insights</div>', unsafe_allow_html=True)

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
