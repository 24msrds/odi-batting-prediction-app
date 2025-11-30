import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate

# -------------------------------------------------
# 1. LOAD FULL RAW DATASET (YOUR BIG CSV)
# -------------------------------------------------

CSV_FILE = "Combined_Top5_Batsmen.csv"  # must be in same folder
df_raw = pd.read_csv(CSV_FILE)

# Expected columns in your original file:
# ['X', 'Runs', 'Mins', 'BF', 'X4s', 'X6s', 'SR', 'Pos', 'Dismissal', 'Inns',
#  'Opposition', 'Ground', 'Start.Date', 'Player', 'Country', 'Type',
#  'Overs', 'Mdns', 'Wkts', 'Econ', 'Source_Sheet']

# -------------------------------------------------
# 2. BASIC CLEANING
# -------------------------------------------------

# Convert Runs & SR to numeric
df_raw["Runs"] = pd.to_numeric(df_raw["Runs"], errors="coerce")
df_raw["SR"] = pd.to_numeric(df_raw["SR"], errors="coerce")

# Keep only rows with valid runs & strike rate
df_raw = df_raw.dropna(subset=["Runs", "SR"])

# Keep only batting innings (Type == 'Batting' if present)
if "Type" in df_raw.columns:
    df_raw = df_raw[df_raw["Type"].str.contains("Bat", case=False, na=False)]

# -------------------------------------------------
# 3. ASSIGN REALISTIC ROLES PER PLAYER
# -------------------------------------------------

role_mapping = {
    # OPENERS
    "Rohit Sharma": "Opener",
    "Shubman Gill": "Opener",
    "David Warner": "Opener",
    "Travis Head": "Opener",
    "Imam-ul-Haq": "Opener",
    "Jonny Bairstow": "Opener",
    "Fakhar Zaman": "Opener",
    "Aaron Finch": "Opener",
    "Devon Conway": "Opener",
    "Will Young": "Opener",

    # MIDDLE ORDER
    "Virat Kohli": "Middle-order",
    "Babar Azam": "Middle-order",
    "Joe Root": "Middle-order",
    "Kane Williamson": "Middle-order",
    "Daryl Mitchell": "Middle-order",
    "Rassie van der Dussen": "Middle-order",
    "David Miller": "Middle-order",
    "Aiden Markram": "Middle-order",
    "Steve Smith": "Middle-order",

    # FINISHERS
    "Jos Buttler": "Finisher",
    "Glenn Maxwell": "Finisher",
    "Moeen Ali": "Finisher",

    # KEEPERS
    "Mohammad Rizwan": "Wicket-Keeper",

    # ALL-ROUNDERS
    "Ben Stokes": "All-Rounder",
    "Ravindra Jadeja": "All-Rounder",
    "Rachin Ravindra": "All-Rounder",
}

df_raw["Role"] = df_raw["Player"].map(role_mapping)

# Remove players without defined role (mainly bowlers)
df_full = df_raw.dropna(subset=["Role"]).copy()

# -------------------------------------------------
# 4. CREATE PERFORMANCE CATEGORY (TARGET)
# -------------------------------------------------

def categorize_runs(runs):
    if runs >= 80:
        return "Excellent"
    elif runs >= 50:
        return "Good"
    elif runs >= 20:
        return "Moderate"
    else:
        return "Low"

df_full["Performance_Category"] = df_full["Runs"].apply(categorize_runs)

# -------------------------------------------------
# 5. FEATURE ENGINEERING (BIGGER & BETTER MODEL)
# -------------------------------------------------

# Convert boundaries columns safely
for col in ["BF", "X4s", "X6s"]:
    if col in df_full.columns:
        df_full[col] = pd.to_numeric(df_full[col], errors="coerce").fillna(0)
    else:
        df_full[col] = 0  # if missing, create as 0

# Label encode categorical fields
le_opp = LabelEncoder()
le_ground = LabelEncoder()
le_role = LabelEncoder()

df_full["Opp_encoded"] = le_opp.fit_transform(df_full["Opposition"])
df_full["Ground_encoded"] = le_ground.fit_transform(df_full["Ground"])
df_full["Role_encoded"] = le_role.fit_transform(df_full["Role"])

# FEATURE SET: using more features (bigger model)
feature_cols = [
    "Opp_encoded",
    "Ground_encoded",
    "Role_encoded",
    "BF",       # balls faced
    "X4s",      # 4s
    "X6s",      # 6s
    "SR"        # strike rate
]

X = df_full[feature_cols]
y = df_full["Performance_Category"]

# -------------------------------------------------
# 6. TRAIN / TEST SPLIT + MODEL TRAINING
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

BEST_MODEL = RandomForestClassifier(
    n_estimators=250,
    max_depth=None,
    random_state=42
)
BEST_MODEL.fit(X_train, y_train)

MODEL_ACCURACY = BEST_MODEL.score(X_test, y_test)
DEPLOYMENT_MODEL_NAME = "Random Forest (Batting Performance)"
UNIQUE_OPPONENTS = sorted(df_full["Opposition"].unique())
TARGET_GROUND = df_full["Ground"].value_counts().idxmax()

# Dummy label encoder (not really used by dashboard, but kept for compatibility)
le = le_opp

# -------------------------------------------------
# 7. ANALYSIS FUNCTION FOR DASHBOARD
# -------------------------------------------------

def analyze_and_rank_players(model, feature_cols_index, le_obj, target_opp, target_ground):
    """
    For a given opposition & ground:
    - Filters matches
    - Predicts performance category
    - Aggregates by player
    - Returns ranked dataframe with tips
    """
    subset = df_full[
        (df_full["Opposition"] == target_opp) &
        (df_full["Ground"] == target_ground)
    ]
    if subset.empty:
        return pd.DataFrame()

    # Prepare features
    subset_feat = subset.copy()
    # (feature_cols_index is not used; we use our local feature_cols)
    X_sub = subset_feat[feature_cols]

    subset_feat["Predicted Category"] = model.predict(X_sub)

    performance_rank = (
        subset_feat.groupby(["Player", "Role"])
        .agg({"Runs": ["mean", "count"], "SR": "mean"})
        .reset_index()
    )
    performance_rank.columns = ["Player", "Role", "Avg Runs", "Innings", "Avg SR"]
    performance_rank = performance_rank.sort_values("Avg Runs", ascending=False)

    def give_tip(category):
        if category == "Excellent":
            return "Maintain aggression, rotate strike smartly, anchor the innings."
        elif category == "Good":
            return "Convert 50s into 100s, focus on building long partnerships."
        elif category == "Moderate":
            return "Work on shot selection and rotating strike; avoid dot-ball pressure."
        else:
            return "Improve temperament; spend more time at the crease before attacking."

    merged = pd.merge(
        performance_rank,
        subset_feat[["Player", "Predicted Category"]],
        on="Player",
        how="left"
    ).drop_duplicates("Player")

    merged["Actionable Tip"] = merged["Predicted Category"].apply(give_tip)

    return merged

# -------------------------------------------------
# 8. HELPER FOR STREAMLIT APP
# -------------------------------------------------

def get_trained_objects():
    """
    Returns everything the Streamlit app needs, without printing.
    """
    return {
        "model": BEST_MODEL,
        "feature_columns": feature_cols,
        "label_encoder": le,
        "unique_opponents": UNIQUE_OPPONENTS,
        "target_ground": TARGET_GROUND,
        "df_full": df_full
    }

# -------------------------------------------------
# 9. OPTIONAL: COMMAND-LINE PREVIEW
# -------------------------------------------------

if __name__ == "__main__":
    print(f"\n✅ Training completed using: {DEPLOYMENT_MODEL_NAME}")
    print(f"🎯 Model Accuracy: {MODEL_ACCURACY:.2%}\n")

    print("Sample cleaned data:")
    print(tabulate(df_full.head(10), headers="keys", tablefmt="psql"))
