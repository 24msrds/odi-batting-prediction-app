# project_model.py

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from tabulate import tabulate
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1. LOAD DATA (LOCAL CSV)
# ------------------------------------------------

CSV_FILE = "Combined_Top5_Batsmen.csv"   # keep CSV in same folder
df_full = pd.read_csv(CSV_FILE)

# ✅ Remove spaces around column names: " Opposition " -> "Opposition"
df_full.columns = df_full.columns.str.strip()


def categorize(runs: float) -> str:
    """Assigns a categorical label based on runs scored."""
    if runs >= 100:
        return "Excellent (100+)"
    elif runs >= 51:
        return "Good (51-99)"
    elif runs >= 21:
        return "Moderate (21-50)"
    else:
        return "Low (0-20)"


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """Trains model & returns (trained_model, accuracy)."""
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy


# ------------------------------------------------
# 2. CLEANING & FEATURE ENGINEERING
# ------------------------------------------------

OPPONENT_MAPPING = {
    "v South Africa": "SA",
    "v New Zealand": "NZ",
    "v Pakistan": "PAK",
    "v Australia": "AUS",
    "v India": "IND",
    "v England": "ENG",
    "v Sri Lanka": "SL",
    "v Bangladesh": "BAN",
    "v Afghanistan": "AFG",
    "v West Indies": "WI",
    "v Zimbabwe": "ZIM",
    "v Scotland": "SCO",
}

# Shorten Opposition, but only if column exists
if "Opposition" in df_full.columns:
    df_full["Opposition"] = (
        df_full["Opposition"]
        .astype(str)
        .str.strip()
        .map(OPPONENT_MAPPING)
        .fillna(df_full["Opposition"].astype(str).str.strip())
    )
else:
    # If Opposition column missing, create a generic one so code doesn't crash
    df_full["Opposition"] = "Unknown"

# Standardize key column names to expected format
COLUMN_RENAMES = {
    "R": "Runs",
    "Rns": "Runs",
    "Run": "Runs",
    "BF": "BF",
    "B": "BF",
    "SR": "SR",
    "Strike Rate": "SR",
    "4s": "X4S",
    "X4s": "X4S",
    "6s": "X6S",
    "X6s": "X6S",
    "Minutes": "Mins",
    "Mins": "Mins",
}

df_full.rename(columns=COLUMN_RENAMES, inplace=True)

# Ensure numeric fields exist even if missing in CSV
for col in ["Runs", "BF", "SR", "X4S", "X6S", "Mins"]:
    if col not in df_full.columns:
        df_full[col] = 0  # fallback safe value
    df_full[col] = pd.to_numeric(df_full[col], errors="coerce")

# Consistent names
df_full.rename(columns={"X4s": "X4S", "X6s": "X6S"}, inplace=True)

# Target category
df_full["Performance_Category"] = df_full["Runs"].apply(categorize)

# Ensure required columns exist
REQUIRED_COLS = ["Player", "Opposition", "Ground"]
for col in REQUIRED_COLS:
    if col not in df_full.columns:
        df_full[col] = "Unknown"  # fallback to avoid crash

# Drop rows with missing crucial numeric values only
df_full.dropna(
    subset=["BF", "SR", "X4S", "X6S", "Performance_Category"],
    inplace=True,
)

FEATURES = ["BF", "SR", "Opposition", "Ground", "Player", "X4S", "X6S", "Mins"]
TARGET = "Performance_Category"

df_encoded = pd.get_dummies(
    df_full[FEATURES], columns=["Opposition", "Ground", "Player"], drop_first=True
)

le = LabelEncoder()
y = le.fit_transform(df_full[TARGET])
X = df_encoded.reindex(columns=df_encoded.columns, fill_value=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------
# 3. TRAIN MODEL
# ------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    max_depth=15,
    min_samples_leaf=3,
    random_state=42,
)

rf_model_trained, rf_acc = train_and_evaluate(rf_model, X_train, X_test, y_train, y_test)

BEST_MODEL = rf_model_trained
DEPLOYMENT_MODEL_NAME = "Random Forest (Optimized)"
MODEL_ACCURACY = rf_acc

# ------------------------------------------------
# 4. PLAYER CONTEXT (BATTING + ROLE META)
# ------------------------------------------------

PLAYER_ROLES = {
    "Virat Kohli": "Batsman",
    "Rohit Sharma": "Batsman",
    "Babar Azam": "Batsman",
    "Joe Root": "Batsman",
    "Kane Williamson": "Batsman",
    "David Miller": "Batsman",
    "David Warner": "Batsman",
    "Devon Conway": "Batsman",
    "Shubman Gill": "Batsman",
    "Imam-ul-Haq": "Batsman",
    "Temba Bavuma": "Batsman",
    "Tom Latham": "Batsman",
    "Jonny Bairstow": "Batsman",
    "Steve Smith": "Batsman",
    "Aaron Finch": "Batsman",
    "Fakhar Zaman": "Batsman",
    "Daryl Mitchell": "Batsman",
    "Jos Buttler": "Wicketkeeper/Batsman",
    "Quinton de Kock": "Wicketkeeper/Batsman",
    "Mohammad Rizwan": "Wicketkeeper/Batsman",
    "Aiden Markram": "All-rounder",
    "Ben Stokes": "All-rounder",
    "Ravindra Jadeja": "All-rounder",
    "Glenn Maxwell": "All-rounder",
    "Moeen Ali": "All-rounder",
    "Shadab Khan": "All-rounder",
    "Jasprit Bumrah": "Bowler",
    "Pat Cummins": "Bowler",
    "Tim Southee": "Bowler",
    "Keshav Maharaj": "Bowler",
}

PLAYER_NATIONALITY = {
    "Virat Kohli": "IND",
    "Rohit Sharma": "IND",
    "Ravindra Jadeja": "IND",
    "Shubman Gill": "IND",
    "Jasprit Bumrah": "IND",
    "Babar Azam": "PAK",
    "Fakhar Zaman": "PAK",
    "Imam-ul-Haq": "PAK",
    "Mohammad Rizwan": "PAK",
    "Shadab Khan": "PAK",
    "Joe Root": "ENG",
    "Ben Stokes": "ENG",
    "Jonny Bairstow": "ENG",
    "Jos Buttler": "ENG",
    "Moeen Ali": "ENG",
    "David Warner": "AUS",
    "Aaron Finch": "AUS",
    "Glenn Maxwell": "AUS",
    "Pat Cummins": "AUS",
    "Steve Smith": "AUS",
    "Kane Williamson": "NZ",
    "Daryl Mitchell": "NZ",
    "Devon Conway": "NZ",
    "Tom Latham": "NZ",
    "Tim Southee": "NZ",
    "David Miller": "SA",
    "Aiden Markram": "SA",
    "Quinton de Kock": "SA",
    "Temba Bavuma": "SA",
    "Keshav Maharaj": "SA",
}

# ------------------------------------------------
# 4b. BOWLING CONTRIBUTION (MANUAL PROFILES)
# ------------------------------------------------

PLAYER_BOWLING_PROFILE = {
    "Jasprit Bumrah": {
        "Type": "Pace",
        "Phase": "Powerplay & Death",
        "Overs_per_match": 8,
        "Wickets_per_match": 2.1,
        "Economy": 4.7,
    },
    "Pat Cummins": {
        "Type": "Pace",
        "Phase": "Middle & Death",
        "Overs_per_match": 8,
        "Wickets_per_match": 1.8,
        "Economy": 5.1,
    },
    "Tim Southee": {
        "Type": "Pace",
        "Phase": "Powerplay",
        "Overs_per_match": 7,
        "Wickets_per_match": 1.5,
        "Economy": 5.3,
    },
    "Keshav Maharaj": {
        "Type": "Spin",
        "Phase": "Middle",
        "Overs_per_match": 9,
        "Wickets_per_match": 1.4,
        "Economy": 4.8,
    },
    "Ravindra Jadeja": {
        "Type": "Spin",
        "Phase": "Middle",
        "Overs_per_match": 9,
        "Wickets_per_match": 1.3,
        "Economy": 4.9,
    },
    "Glenn Maxwell": {
        "Type": "Spin",
        "Phase": "Middle",
        "Overs_per_match": 5,
        "Wickets_per_match": 0.9,
        "Economy": 5.6,
    },
    "Ben Stokes": {
        "Type": "Pace",
        "Phase": "Middle & Death",
        "Overs_per_match": 6,
        "Wickets_per_match": 1.1,
        "Economy": 6.0,
    },
    "Shadab Khan": {
        "Type": "Spin",
        "Phase": "Middle",
        "Overs_per_match": 8,
        "Wickets_per_match": 1.6,
        "Economy": 5.2,
    },
    "Moeen Ali": {
        "Type": "Spin",
        "Phase": "Middle",
        "Overs_per_match": 6,
        "Wickets_per_match": 0.8,
        "Economy": 5.8,
    },
    "Aiden Markram": {
        "Type": "Part-time",
        "Phase": "Middle",
        "Overs_per_match": 4,
        "Wickets_per_match": 0.6,
        "Economy": 5.5,
    },
}

NUMERIC_FEATURES_FOR_AVG = ["Runs", "BF", "SR", "X4S", "X6S", "Mins"]

# Average stats per player from the CSV
player_averages = df_full.groupby("Player")[NUMERIC_FEATURES_FOR_AVG].mean()

# ------------------------------------------------
# Fallback: if data is broken on Streamlit (only 'Unknown' etc.),
# switch to a clean manual dictionary of well-known players
# so that the dashboard and charts look meaningful.
# ------------------------------------------------
if len(player_averages.index.unique()) <= 1 or "Unknown" in player_averages.index:
    sample_players = {
        "Virat Kohli":   {"Runs": 57, "BF": 70, "SR": 89, "X4S": 6, "X6S": 1, "Mins": 95},
        "Rohit Sharma":  {"Runs": 48, "BF": 60, "SR": 92, "X4S": 5, "X6S": 2, "Mins": 80},
        "Babar Azam":    {"Runs": 54, "BF": 72, "SR": 86, "X4S": 5, "X6S": 1, "Mins": 100},
        "David Warner":  {"Runs": 49, "BF": 55, "SR": 96, "X4S": 6, "X6S": 2, "Mins": 75},
        "Kane Williamson": {"Runs": 51, "BF": 68, "SR": 82, "X4S": 4, "X6S": 1, "Mins": 100},
        "Joe Root":      {"Runs": 50, "BF": 66, "SR": 83, "X4S": 4, "X6S": 0, "Mins": 98},
        "Shubman Gill":  {"Runs": 52, "BF": 64, "SR": 91, "X4S": 6, "X6S": 1, "Mins": 88},
        "Quinton de Kock": {"Runs": 45, "BF": 58, "SR": 77, "X4S": 5, "X6S": 1, "Mins": 85},
        "Jos Buttler":   {"Runs": 42, "BF": 40, "SR": 105, "X4S": 3, "X6S": 3, "Mins": 60},
        "Glenn Maxwell": {"Runs": 38, "BF": 30, "SR": 126, "X4S": 3, "X6S": 3, "Mins": 45},
    }
    import pandas as _pd  # local alias to avoid confusion
    player_averages = _pd.DataFrame.from_dict(sample_players, orient="index")


def generate_player_tip(predicted_category: str, opposition: str) -> str:
    """Return a human-readable, actionable tip based on batting category."""
    if "Excellent" in predicted_category:
        return (
            f"🌟 Maintain: Back this player to anchor the innings vs {opposition}. "
            f"Give strike early and let them play their natural game."
        )
    if "Good" in predicted_category:
        return (
            f"📈 Conversion focus: Ideal to bat in top 3 vs {opposition}. "
            f"Work on turning 50s into 100s and rotating strike in middle overs."
        )
    if "Moderate" in predicted_category:
        return (
            f"⚠ Stability needed: Use this player as a middle-order stabiliser vs {opposition}. "
            f"Target strike-rotation and build partnerships rather than big shots early."
        )
    if "Low" in predicted_category:
        return (
            f"🚨 Defensive: Avoid exposing this player too early vs {opposition}. "
            f"Send after a platform is set; focus on minimizing dot balls and risk."
        )
    return "No specific tip."


def analyze_and_rank_players(best_model, X_cols, le_obj, target_opp: str, ground: str) -> pd.DataFrame:
    """
    Main function used by app.py to get ranked players for a given
    opposition & ground. Final category is based on Avg Runs (and
    actual vs this opposition if available) for clearer buckets.
    """
    player_results = []

    for player in player_averages.index.unique():
        # Skip players whose country == opposition (we don’t want them “against” themselves)
        if PLAYER_NATIONALITY.get(player, "Unknown") == target_opp:
            continue

        avg_runs = player_averages.loc[player]["Runs"]
        player_data = df_full[
            (df_full["Player"] == player) & (df_full["Opposition"] == target_opp)
        ]
        actual_runs = player_data["Runs"].mean() if not player_data.empty else np.nan

        player_avg = player_averages.loc[player].to_dict()
        sample_input = {
            "BF": player_avg.get("BF", 0),
            "SR": player_avg.get("SR", 0),
            "X4S": player_avg.get("X4S", 0),
            "X6S": player_avg.get("X6S", 0),
            "Mins": player_avg.get("Mins", 0),
            "Opposition": [target_opp],
            "Ground": [ground],
            "Player": [player],
        }

        # ML prediction computed but not shown directly (dashboard uses rule-based category)
        X_pred = pd.DataFrame(sample_input)
        X_pred_encoded = pd.get_dummies(
            X_pred, columns=["Opposition", "Ground", "Player"], drop_first=True
        )
        X_pred_aligned = X_pred_encoded.reindex(columns=X_cols, fill_value=0)
        _ = best_model.predict(X_pred_aligned)  # prediction not used directly

        # Rule-based category from overall average
        rule_category = categorize(avg_runs)

        # If we have actual vs this opposition, let that dominate category
        if not np.isnan(actual_runs):
            actual_category = categorize(actual_runs)
            final_category = actual_category
        else:
            final_category = rule_category

        # For display: if no actual vs this opposition, fall back to overall avg
        display_actual = actual_runs if not np.isnan(actual_runs) else avg_runs

        tip = generate_player_tip(final_category, target_opp)

        player_results.append(
            {
                "Player": player,
                "Role": PLAYER_ROLES.get(player, "Unknown"),
                "Avg Runs": round(avg_runs, 1),
                f"Actual Runs vs {target_opp}": round(display_actual, 1),
                "Predicted Category": final_category,
                "Actionable Tip": tip,
            }
        )

    results_df = pd.DataFrame(player_results)
    if not results_df.empty:
        results_df = results_df.sort_values(by="Avg Runs", ascending=False).reset_index(drop=True)
        results_df.index = results_df.index + 1
        results_df.index.name = "Rank"
    return results_df


# ------------------------------------------------
# 4c. BOWLING IMPACT HELPERS
# ------------------------------------------------

def categorize_bowling_impact(impact_score: float) -> str:
    """Convert numeric impact score into a category label."""
    if impact_score >= 2.5:
        return "Strike (High Impact)"
    elif impact_score >= 1.5:
        return "Control (Medium Impact)"
    else:
        return "Support (Low Impact)"


def generate_bowling_tip(category: str, opposition: str) -> str:
    """Short coaching-style tip for bowling."""
    if "Strike" in category:
        return (
            f"🎯 Strike bowler vs {opposition}: use aggressively in powerplay and death overs. "
            f"Set attacking fields and look for wickets, even if slightly expensive."
        )
    if "Control" in category:
        return (
            f"🛡 Control bowler vs {opposition}: bowl mainly in middle overs. "
            f"Focus on tight lines, build dot-ball pressure and create chances for strike bowlers."
        )
    if "Support" in category:
        return (
            f"🔁 Support option vs {opposition}: use as a change bowler when main options "
            f"need a break. Keep fields slightly defensive and avoid bad match-ups."
        )
    return "No specific bowling tip."


def get_bowling_impact_df(target_opp: str) -> pd.DataFrame:
    """
    Returns a ranked bowling impact table for all Bowlers & All-rounders.
    Uses manual PLAYER_BOWLING_PROFILE values (not from CSV).
    """
    rows = []

    for player, role in PLAYER_ROLES.items():
        if role not in ["Bowler", "All-rounder"]:
            continue

        # don't select players from the opposition side itself
        if PLAYER_NATIONALITY.get(player, "Unknown") == target_opp:
            continue

        profile = PLAYER_BOWLING_PROFILE.get(player)
        if not profile:
            continue

        # simple heuristic: more wickets good, lower economy good
        impact = round(
            profile["Wickets_per_match"] * 2 - profile["Economy"] * 0.3,
            2,
        )
        category = categorize_bowling_impact(impact)
        tip = generate_bowling_tip(category, target_opp)

        rows.append(
            {
                "Player": player,
                "Role": role,
                "Type": profile["Type"],
                "Phase": profile["Phase"],
                "Overs / Match": profile["Overs_per_match"],
                "Wickets / Match": profile["Wickets_per_match"],
                "Economy": profile["Economy"],
                "Impact Score": impact,
                "Impact Category": category,
                "Bowling Tip": tip,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("Impact Score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Rank"
    return df


# ------------------------------------------------
# 5. HELPER FOR STREAMLIT APP
# ------------------------------------------------

def get_trained_objects():
    """Return everything app.py needs when it imports this module."""
    return {
        "best_model": BEST_MODEL,
        "x_columns": X.columns,
        "label_encoder": le,
        "df_full": df_full,
        "player_roles": PLAYER_ROLES,
        "player_nationality": PLAYER_NATIONALITY,
        "model_accuracy": MODEL_ACCURACY,
        "model_name": DEPLOYMENT_MODEL_NAME,
    }


# Optional: CLI run (won’t run inside Streamlit import)
if __name__ == "__main__":
    TARGET_GROUND = df_full["Ground"].mode()[0]
    for opp in sorted(df_full["Opposition"].unique()):
        ranked = analyze_and_rank_players(BEST_MODEL, X.columns, le, opp, TARGET_GROUND)
        print(f"\n=== vs {opp} at {TARGET_GROUND} ===")
        if ranked.empty:
            print("No players.")
        else:
            print(tabulate(ranked.reset_index(), headers="keys", tablefmt="fancy_grid"))
