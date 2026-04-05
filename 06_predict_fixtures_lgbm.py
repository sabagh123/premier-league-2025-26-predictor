import pandas as pd, numpy as np
from pathlib import Path
from joblib import load

# Paths
CLEAN = Path("data/processed/pl_matches_clean.csv")
HIST  = Path("data/processed/train_frame.parquet")   # features we trained on
FIX   = Path("data/fixtures_2526.csv")               # you create this file
ART   = Path("artifacts")
OUTD  = Path("predictions"); OUTD.mkdir(parents=True, exist_ok=True)

# --- feature selection (same rules as training) ---
def feature_cols(df):
    drop_exact = {"Date","HomeTeam","AwayTeam","FTR","season","B365H","B365D","B365A","FTHG","FTAG"}
    drop_prefixes = ("FT","HT")
    cols=[]
    for c in df.columns:
        if c in drop_exact: continue
        if any(c.startswith(p) for p in drop_prefixes): continue
        if df[c].dtype=="O": continue
        cols.append(c)
    return cols

# Map long names to canonical (same as earlier)
def canon_team(name: str) -> str:
    mapping = {
        "Tottenham Hotspur":"Tottenham", "Spurs":"Tottenham",
        "Brighton & Hove Albion":"Brighton","Brighton and Hove Albion":"Brighton",
        "Wolves":"Wolverhampton","Wolverhampton Wanderers":"Wolverhampton",
        "West Ham United":"West Ham","Newcastle United":"Newcastle","Leeds United":"Leeds",
        "Nott'm Forest":"Nottingham Forest","Nott Forest":"Nottingham Forest",
        "AFC Bournemouth":"Bournemouth",
        "Man Utd":"Manchester United","Man United":"Manchester United","Manchester Utd":"Manchester United",
        "Man City":"Manchester City",
    }
    return mapping.get(name, name)

def most_recent_row(df_feat: pd.DataFrame, team: str, before_date: pd.Timestamp) -> pd.Series | None:
    # last match for team BEFORE fixture date
    sub = df_feat[(df_feat["Date"] < before_date) & ((df_feat["HomeTeam"]==team) | (df_feat["AwayTeam"]==team))]
    if sub.empty: return None
    return sub.iloc[-1]

def build_fixture_vector(row_h: pd.Series | None, row_a: pd.Series | None, Xcols: list[str]) -> dict:
    """
    Build a single prediction vector matching Xcols.
    Uses venue-agnostic (overall) form when available, and maps venue-specific if possible.
    Handles cases where the last game was away by swapping prefixes.
    """
    vec = {c: np.nan for c in Xcols}

    def put_from(source: pd.Series, needed_prefix: str):
        if source is None: return
        if needed_prefix == "home_":
            # If source row is home for that team, take home_*; else map away_* -> home_*
            if source["HomeTeam"] == team_h:
                src_pref, dst_pref = "home_", "home_"
            else:
                src_pref, dst_pref = "away_", "home_"
        else:
            if source["AwayTeam"] == team_a:
                src_pref, dst_pref = "away_", "away_"
            else:
                src_pref, dst_pref = "home_", "away_"

        for c in Xcols:
            if c.startswith(dst_pref):
                alt = c.replace(dst_pref, src_pref, 1)
                if alt in source.index:
                    vec[c] = source[alt]

    # overall/venue features
    if row_h is not None:
        put_from(row_h, "home_")
    if row_a is not None:
        put_from(row_a, "away_")

    # Elo handling (always present in hist)
    if row_h is not None:
        vec["elo_home"] = row_h["elo_home"] if row_h["HomeTeam"]==team_h else row_h["elo_away"]
    if row_a is not None:
        vec["elo_away"] = row_a["elo_away"] if row_a["AwayTeam"]==team_a else row_a["elo_home"]
    if "elo_diff" in vec:
        vec["elo_diff"] = vec.get("elo_home", np.nan) - vec.get("elo_away", np.nan)

    # Rest difference if column exists
    if "rest_diff" in vec:
        hr = row_h["home_rest"] if (row_h is not None and row_h["HomeTeam"]==team_h and "home_rest" in row_h) \
             else (row_h["away_rest"] if (row_h is not None and "away_rest" in row_h) else np.nan)
        ar = row_a["away_rest"] if (row_a is not None and row_a["AwayTeam"]==team_a and "away_rest" in row_a) \
             else (row_a["home_rest"] if (row_a is not None and "home_rest" in row_a) else np.nan)
        vec["rest_diff"] = (hr - ar) if pd.notna(hr) and pd.notna(ar) else np.nan

    # Month-in-season if present
    if "mos" in vec and "Date" in globals():
        vec["mos"] = ((fix_date.month - 8) % 12) + 1

    return vec

if __name__ == "__main__":
    # Load artifacts & data
    cal = load(ART / "lgbm_calibrated.joblib")
    hist = pd.read_parquet(HIST).sort_values("Date").reset_index(drop=True)
    Xcols = feature_cols(hist)

    # Fixtures file: Date,HomeTeam,AwayTeam
    fix = pd.read_csv(FIX, parse_dates=["Date"])
    fix["HomeTeam"] = fix["HomeTeam"].map(canon_team)
    fix["AwayTeam"] = fix["AwayTeam"].map(canon_team)

    preds = []
    for _, r in fix.iterrows():
        team_h, team_a = r["HomeTeam"], r["AwayTeam"]
        fix_date = r["Date"]

        row_h = most_recent_row(hist, team_h, fix_date)
        row_a = most_recent_row(hist, team_a, fix_date)

        vec = build_fixture_vector(row_h, row_a, Xcols)
        X = pd.DataFrame([vec])[Xcols]
        proba = cal.predict_proba(X)[0]

        # map class order safely
        classes = list(cal.classes_)
        preds.append({
            "Date": fix_date, "HomeTeam": team_h, "AwayTeam": team_a,
            "P(H)": float(proba[classes.index("H")]) if "H" in classes else np.nan,
            "P(D)": float(proba[classes.index("D")]) if "D" in classes else np.nan,
            "P(A)": float(proba[classes.index("A")]) if "A" in classes else np.nan,
        })

    out = pd.DataFrame(preds).sort_values("Date")
    out_path = OUTD / "predictions_2526_lgbm.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved -> {out_path} ({len(out)} rows)")