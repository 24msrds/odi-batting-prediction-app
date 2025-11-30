import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pydeck as pdk  # 🔹 for the animated venue map
import numpy as np     # 🔹 for gradient colors

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
        background: linear-gradient(140deg, #0a192f 0%, #112d4e 40%, #1b3b5f 100%);
        color: #e2e8f0;
    }

    body {
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, "Roboto", sans-serif;
    }

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
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.7);
        box-shadow: 0 16px 45px rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 26px 70px rgba(37, 99, 235, 0.75);
        border-color: rgba(59, 130, 246, 0.95);
    }
    .metric-label {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-bottom: 0.15rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e5e7eb;
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
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.6);
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
        background: rgba(15, 23, 42, 0.88);
        border: 1px solid rgba(59, 130, 246, 0.9);
        margin-bottom: 0.4rem;
        color: #e5e7eb;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.8);
    }

    /* -------- TABS STYLING -------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 0.55rem 1.1rem;
        background: rgba(30, 64, 175, 0.55);
        color: #e5e7eb;
        font-weight: 500;
        border: 1px solid transparent;
        transition: 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(120deg, #3b82f6, #1d4ed8);
        color: white;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(120deg, #1d4ed8, #0ea5e9);
        color: white;
        border-color: rgba(191, 219, 254, 0.9);
        font-weight: 650;
        box-shadow: 0 8px 22px rgba(37, 99, 235, 0.7);
    }

    /* -------- DATAFRAME TABLE -------- */
    .dataframe {
        border-radius: 0.75rem;
        overflow: hidden;
        background: #020617;
    }

    /* -------- EXPANDER STYLING -------- */
    .streamlit-expanderHeader {
        background: #0b1220 !important;
        color: #e5e7eb !important;
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

# ---- Map metadata: lat/lon, team, country ----
VENUE_META = {
    "Delhi":       {"lat": 28.6139,  "lon": 77.2090,  "team": "IND", "country": "India"},
    "Mumbai":      {"lat": 19.0760,  "lon": 72.8777,  "team": "IND", "country": "India"},
    "Chennai":     {"lat": 13.0827,  "lon": 80.2707,  "team": "IND", "country": "India"},
    "Kolkata":     {"lat": 22.5726,  "lon": 88.3639,  "team": "IND", "country": "India"},
    "Ahmedabad":   {"lat": 23.0225,  "lon": 72.5714,  "team": "IND", "country": "India"},

    "Melbourne":   {"lat": -37.8136, "lon": 144.9631, "team": "AUS", "country": "Australia"},
    "Sydney":      {"lat": -33.8688, "lon": 151.2093, "team": "AUS", "country": "Australia"},
    "Adelaide":    {"lat": -34.9285, "lon": 138.6007, "team": "AUS", "country": "Australia"},
    "Perth":       {"lat": -31.9505, "lon": 115.8605, "team": "AUS", "country": "Australia"},

    "Lords":       {"lat": 51.5299,  "lon": -0.1722,  "team": "ENG", "country": "England"},
    "The Oval":    {"lat": 51.4837,  "lon": -0.1147,  "team": "ENG", "country": "England"},
    "Birmingham":  {"lat": 52.4862,  "lon": -1.8904,  "team": "ENG", "country": "England"},

    "Abu Dhabi":   {"lat": 24.4539,  "lon": 54.3773,  "team": "UAE", "country": "UAE"},
    "Dubai":       {"lat": 25.2048,  "lon": 55.2708,  "team": "UAE", "country": "UAE"},
    "Sharjah":     {"lat": 25.3463,  "lon": 55.4209,  "team": "UAE", "country": "UAE"},

    "Johannesburg": {"lat": -26.2041, "lon": 28.0473,   "team": "SA", "country": "South Africa"},
    "Cape Town":    {"lat": -33.9249, "lon": 18.4241,   "team": "SA", "country": "South Africa"},
    "Durban":       {"lat": -29.8587, "lon": 31.0218,   "team": "SA", "country": "South Africa"},

    "Wellington":  {"lat": -41.2865, "lon": 174.7762, "team": "NZ", "country": "New Zealand"},
    "Auckland":    {"lat": -36.8485, "lon": 174.7633, "team": "NZ", "country": "New Zealand"},
    "Christchurch":{"lat": -43.5321, "lon": 172.6362, "team": "NZ", "country": "New Zealand"},
}

def venue_color(team: str):
    """Neon-ish RGBA colors per team."""
    mapping = {
        "IND": [37, 99, 235, 220],     # royal blue
        "AUS": [234, 179, 8, 220],     # yellow
        "ENG": [248, 113, 113, 220],   # red
        "UAE": [45, 212, 191, 220],    # teal
        "SA":  [74, 222, 128, 220],    # green
        "NZ":  [129, 140, 248, 220],   # indigo
    }
    return mapping.get(team, [56, 189, 248, 220])  # default cyan

def build_venue_df():
    rows = []
    for venue in GROUNDS:
        if venue == "Neutral Venue":
            continue
        meta = VENUE_META.get(venue)
        if not meta:
            continue
        rows.append(
            {
                "venue": venue,
                "lat": meta["lat"],
                "lon": meta["lon"],
                "team": meta["team"],
                "country": meta["country"],
                "color": venue_color(meta["team"]),
                "icon_name": "stadium",
            }
        )
    return pd.DataFrame(rows)

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

# Tabs for Batting / Bowling / Map
bat_tab, bowl_tab, map_tab = st.tabs(
    ["🏏 Batting Analysis", "🎯 Bowling Analysis", "🗺 Venue Map"]
)

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

        # ---------- PREMIUM THEME BAR CHART (BATTING) ----------
        fig, ax = plt.subplots(figsize=(11, max(4, len(chart_df) * 0.6)))

        fig.patch.set_facecolor("#0b1220")
        ax.set_facecolor("#020617")

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Gradient blue bars
        colors = np.linspace(0.3, 0.85, len(chart_df))
        bars = ax.barh(
            chart_df["Player"],
            chart_df["Avg Runs"],
            height=0.55,
            color=[plt.cm.Blues(c) for c in colors],
            edgecolor="#0f172a",
            linewidth=1.2,
        )

        # Grid
        ax.grid(axis="x", linestyle="--", alpha=0.35, color="#475569")
        ax.set_axisbelow(True)

        # Titles & labels
        ax.set_title(
            f"Average Runs vs {target_opp} at {target_ground}",
            fontsize=18,
            fontweight="bold",
            color="#e2e8f0",
            pad=15,
        )
        ax.set_xlabel("Average Runs", fontsize=13, color="#cbd5e1")
        ax.set_ylabel("Player", fontsize=13, color="#cbd5e1")
        ax.tick_params(colors="#e2e8f0", labelsize=12)

        # Values at end of bars
        for i, (bar, val) in enumerate(zip(bars, chart_df["Avg Runs"])):
            ax.text(
                val + 1,
                i,
                f"{val:.1f}",
                va="center",
                fontsize=12,
                color="#e5e7eb",
                fontweight="bold",
            )

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

        # ---------- PREMIUM THEME BAR CHART (BOWLING) ----------
        fig_bowl, ax_bowl = plt.subplots(figsize=(11, max(4, len(bowl_chart) * 0.6)))

        fig_bowl.patch.set_facecolor("#0b1220")
        ax_bowl.set_facecolor("#020617")

        for spine in ax_bowl.spines.values():
            spine.set_visible(False)

        colors_b = np.linspace(0.2, 0.9, len(bowl_chart))
        bars_b = ax_bowl.barh(
            bowl_chart["Player"],
            bowl_chart["Impact Score"],
            height=0.55,
            color=[plt.cm.Greens(c) for c in colors_b],
            edgecolor="#022c22",
            linewidth=1.2,
        )

        ax_bowl.grid(axis="x", linestyle="--", alpha=0.35, color="#4b5563")
        ax_bowl.set_axisbelow(True)

        ax_bowl.set_title(
            f"Bowling Impact vs {target_opp}",
            fontsize=18,
            fontweight="bold",
            color="#e2e8f0",
            pad=15,
        )
        ax_bowl.set_xlabel("Impact Score", fontsize=13, color="#cbd5e1")
        ax_bowl.set_ylabel("Player", fontsize=13, color="#cbd5e1")
        ax_bowl.tick_params(colors="#e2e8f0", labelsize=12)

        for i, (bar, v) in enumerate(zip(bars_b, bowl_chart["Impact Score"])):
            ax_bowl.text(
                v + 0.02,
                i,
                f"{v:.2f}",
                va="center",
                fontsize=12,
                color="#e5e7eb",
                fontweight="bold",
            )

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

# -------------------------------------------------
# VENUE MAP TAB
# -------------------------------------------------
with map_tab:
    st.markdown('<div class="section-heading">🗺 Animated Venue Impact Map</div>', unsafe_allow_html=True)

    venue_df = build_venue_df()
    if venue_df.empty:
        st.info("No venue coordinates configured yet.")
    else:
        # Focus dropdown
        focus_choice = st.selectbox(
            "Focus view",
            ["All venues"] + venue_df["venue"].tolist(),
            index=0,
        )

        # Animation slider for pulsing effect
        frame = st.slider(
            "Pulse animation frame",
            min_value=0,
            max_value=100,
            value=0,
            help="Slide to see neon pulses around venues.",
        )

        df_map = venue_df.copy()
        base_radius = 40000
        step = 4000
        frame_mod = frame % 20

        pulse_radii = []
        heights = []
        for i in range(len(df_map)):
            offset = (frame_mod + i * 5) % 20
            pulse_radii.append(base_radius + offset * step)
            heights.append(100000 + i * 30000)
        df_map["pulse_radius"] = pulse_radii
        df_map["height"] = heights

        # View state
        if focus_choice == "All venues":
            view_state = pdk.ViewState(
                latitude=df_map["lat"].mean(),
                longitude=df_map["lon"].mean(),
                zoom=1.8,
                pitch=45,
                bearing=15,
            )
        else:
            row = df_map[df_map["venue"] == focus_choice].iloc[0]
            view_state = pdk.ViewState(
                latitude=row["lat"],
                longitude=row["lon"],
                zoom=7,
                pitch=60,
                bearing=30,
            )

        # Country borders overlay
        country_layer = pdk.Layer(
            "GeoJsonLayer",
            data="https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json",
            stroked=True,
            filled=False,
            get_line_color=[96, 165, 250],
            line_width_min_pixels=0.5,
            opacity=0.25,
        )

        # 3D columns (stadium peaks)
        column_layer = pdk.Layer(
            "ColumnLayer",
            data=df_map,
            get_position=["lon", "lat"],
            get_elevation="height",
            elevation_scale=1,
            radius=30000,
            get_fill_color="color",
            pickable=True,
            extruded=True,
        )

        # Neon pulsing scatter
        neon_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position=["lon", "lat"],
            get_radius="pulse_radius",
            get_fill_color="color",
            get_line_color=[255, 255, 255],
            line_width_min_pixels=1,
            stroked=True,
            filled=True,
            opacity=0.35,
            pickable=True,
        )

        deck = pdk.Deck(
            layers=[country_layer, column_layer, neon_layer],
            initial_view_state=view_state,
            map_style="mapbox://styles/mapbox/navigation-night-v1",
            tooltip={
                "html": "<b>{venue}</b><br/>Team: {team}<br/>Country: {country}",
                "style": {"color": "white"},
            },
        )

        st.pydeck_chart(deck)

        st.markdown(
            """
            <div style="
                margin-top: 0.5rem;
                padding: 0.5rem 0.75rem;
                background: rgba(15, 23, 42, 0.85);
                border-radius: 0.5rem;
                border: 1px solid rgba(96, 165, 250, 0.8);
                font-size: 0.8rem;
                color: #e5e7eb;
            ">
                <b>Legend:</b>
                🔵 <b>Neon rings</b> = active venue pulse &nbsp; | &nbsp;
                ⬛ <b>3D columns</b> = relative venue prominence &nbsp; | &nbsp;
                📍 <b>Center</b> = venue location
            </div>
            """,
            unsafe_allow_html=True,
        )
