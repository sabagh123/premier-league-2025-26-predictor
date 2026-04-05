import numpy as np, pandas as pd
from pathlib import Path

CLEAN = Path("data/processed/pl_matches_clean.csv")
OUT_PARQ = Path("data/processed/train_frame.parquet")
OUT_CSV  = Path("data/processed/train_frame.csv")

# ---------- helpers ----------
def long_team_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match: one from each team's perspective. Contains only past info after we shift."""
    df = df.copy().reset_index(drop=True)
    df["mid"] = df.index  # stable match id

    home = pd.DataFrame({
        "mid": df["mid"], "Date": df["Date"],
        "team": df["HomeTeam"].astype(str),
        "opp":  df["AwayTeam"].astype(str),
        "is_home": 1,
        "gf": df["FTHG"].astype(float),
        "ga": df["FTAG"].astype(float),
    })
    away = pd.DataFrame({
        "mid": df["mid"], "Date": df["Date"],
        "team": df["AwayTeam"].astype(str),
        "opp":  df["HomeTeam"].astype(str),
        "is_home": 0,
        "gf": df["FTAG"].astype(float),
        "ga": df["FTHG"].astype(float),
    })
    long = pd.concat([home, away], ignore_index=True).sort_values(["team","Date","mid"])
    long["gd"] = long["gf"] - long["ga"]
    long["win"]  = (long["gf"] > long["ga"]).astype(int)
    long["draw"] = (long["gf"] == long["ga"]).astype(int)
    long["loss"] = (long["gf"] < long["ga"]).astype(int)
    long["points"] = 3*long["win"] + long["draw"]
    long["cs"] = (long["ga"]==0).astype(int)            # clean sheet
    long["btts"] = ((long["gf"]>0) & (long["ga"]>0)).astype(int)
    long["rest_days"] = long.groupby("team")["Date"].diff().dt.days
    return long

def add_rolling_features(long: pd.DataFrame, windows=(5,10)) -> pd.DataFrame:
    """Rolling means using ONLY past games (shifted), for overall and home/away split."""
    long = long.copy()
    def past_roll(g, col, w):  # shift() ensures no leakage
        return g[col].shift().rolling(w, min_periods=1).mean()

    # overall (no venue split)
    g_all = long.groupby("team", group_keys=False)
    for w in windows:
        long[f"ovr{w}_gf"]   = past_roll(g_all, "gf", w)
        long[f"ovr{w}_ga"]   = past_roll(g_all, "ga", w)
        long[f"ovr{w}_gd"]   = past_roll(g_all, "gd", w)
        long[f"ovr{w}_ppg"]  = past_roll(g_all, "points", w)
        long[f"ovr{w}_wr"]   = past_roll(g_all, "win", w)
        long[f"ovr{w}_dr"]   = past_roll(g_all, "draw", w)
        long[f"ovr{w}_lr"]   = past_roll(g_all, "loss", w)
        long[f"ovr{w}_cs"]   = past_roll(g_all, "cs", w)
        long[f"ovr{w}_btts"] = past_roll(g_all, "btts", w)

    # venue-specific
    g_venue = long.groupby(["team","is_home"], group_keys=False)
    for w in windows:
        long[f"ven{w}_gf"]   = past_roll(g_venue, "gf", w)
        long[f"ven{w}_ga"]   = past_roll(g_venue, "ga", w)
        long[f"ven{w}_gd"]   = past_roll(g_venue, "gd", w)
        long[f"ven{w}_ppg"]  = past_roll(g_venue, "points", w)
        long[f"ven{w}_wr"]   = past_roll(g_venue, "win", w)
        long[f"ven{w}_dr"]   = past_roll(g_venue, "draw", w)
        long[f"ven{w}_lr"]   = past_roll(g_venue, "loss", w)
        long[f"ven{w}_cs"]   = past_roll(g_venue, "cs", w)
        long[f"ven{w}_btts"] = past_roll(g_venue, "btts", w)

    return long

def compute_elo(df: pd.DataFrame, k=24.0, home_adv=70.0) -> pd.DataFrame:
    """Pre-match Elo for home/away (no leakage)."""
    df = df.sort_values("Date").copy()
    teams = pd.unique(df[["HomeTeam","AwayTeam"]].values.ravel("K"))
    elo = {t:1500.0 for t in teams}
    pre_h, pre_a = [], []

    for _, r in df.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        Rh, Ra = elo[h], elo[a]
        pre_h.append(Rh); pre_a.append(Ra)

        # expected with home advantage
        exp_h = 1.0 / (1.0 + 10 ** (-((Rh + home_adv) - Ra)/400))
        exp_a = 1.0 - exp_h

        # actual result
        if   r["FTHG"] > r["FTAG"]: s_h, s_a = 1.0, 0.0
        elif r["FTHG"] < r["FTAG"]: s_h, s_a = 0.0, 1.0
        else:                        s_h, s_a = 0.5, 0.5

        margin = abs(float(r["FTHG"]) - float(r["FTAG"]))
        mov = np.log1p(margin) if margin>0 else 1.0

        elo[h] = Rh + k * mov * (s_h - exp_h)
        elo[a] = Ra + k * mov * (s_a - exp_a)

    out = pd.DataFrame({"elo_home": pre_h, "elo_away": pre_a}, index=df.index)
    out["elo_diff"] = out["elo_home"] - out["elo_away"]
    return out

def odds_to_probs(row):
    if not set(["B365H","B365D","B365A"]).issubset(row.index):  # odds may not exist
        return pd.Series([np.nan,np.nan,np.nan], index=["pH","pD","pA"])
    try:
        inv = np.array([1/row["B365H"], 1/row["B365D"], 1/row["B365A"]], dtype=float)
        s = np.nansum(inv)
        return pd.Series(inv/s, index=["pH","pD","pA"]) if np.isfinite(s) and s>0 else pd.Series([np.nan,np.nan,np.nan], index=["pH","pD","pA"])
    except Exception:
        return pd.Series([np.nan,np.nan,np.nan], index=["pH","pD","pA"])

# ---------- main ----------
def main():
    assert CLEAN.exists(), "Run 02_lock_base.py first."
    df = pd.read_csv(CLEAN, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    # Build long frame + rolling features (all computed on shifted past)
    long = long_team_frame(df)
    long = add_rolling_features(long, windows=(5,10))

    # Split back to match rows: pick the home-side row for home_* features and away-side row for away_* features
    def pick_cols(prefixes, side_is_home):
        cols = [c for c in long.columns if any(c.startswith(p) for p in prefixes)]
        side = long[long["is_home"]==side_is_home].set_index("mid")[cols]
        side = side.add_prefix("home_" if side_is_home==1 else "away_")
        return side

    home_ovr = pick_cols(["ovr5_","ovr10_"], 1)
    away_ovr = pick_cols(["ovr5_","ovr10_"], 0)
    home_ven = pick_cols(["ven5_","ven10_"], 1)
    away_ven = pick_cols(["ven5_","ven10_"], 0)

    rest_home = long[long["is_home"]==1].set_index("mid")[["rest_days"]].rename(columns={"rest_days":"home_rest"})
    rest_away = long[long["is_home"]==0].set_index("mid")[["rest_days"]].rename(columns={"rest_days":"away_rest"})

    X = df.copy()
    X.index.name = "mid"
    X = X.join([home_ovr, away_ovr, home_ven, away_ven, rest_home, rest_away])

    # Elo (pre-match)
    elo = compute_elo(df)
    X = X.join(elo)

    # Odds → implied probabilities (optional)
    if {"B365H","B365D","B365A"}.issubset(X.columns):
        probs = X.apply(odds_to_probs, axis=1).add_prefix("odds_")
        X = pd.concat([X, probs], axis=1)

    # Require minimal history (avoid early-season noise)
    need = ["home_ovr5_ppg","away_ovr5_ppg"]
    keep = X[need].notna().all(axis=1)
    X = X.loc[keep].copy()
    X["mos"] = ((X["Date"].dt.month - 8) % 12) + 1
    X["rest_diff"] = (X["home_rest"] - X["away_rest"]).clip(-10, 10)

    # Save
    OUT_PARQ.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(OUT_PARQ, index=False)
    X.to_csv(OUT_CSV, index=False)
    print(f"Saved features: {OUT_PARQ}  rows={len(X)}  cols={len(X.columns)}")

if __name__ == "__main__":
    main()