# src/07_table_projection.py
import numpy as np
import pandas as pd
from pathlib import Path

PRED_PATH = Path("predictions/predictions_2526_lgbm.csv")
OUTDIR = Path("predictions"); OUTDIR.mkdir(parents=True, exist_ok=True)

def load_predictions(path: Path) -> pd.DataFrame:
    assert path.exists() and path.stat().st_size > 0, f"Missing/empty predictions: {path}"
    df = pd.read_csv(path, parse_dates=["Date"])
    need = {"Date","HomeTeam","AwayTeam","P(H)","P(D)","P(A)"}
    missing = need - set(df.columns)
    assert not missing, f"Predictions must include {missing}"
    # normalize tiny drift so probs sum to 1
    s = df["P(H)"] + df["P(D)"] + df["P(A)"]
    df[["P(H)","P(D)","P(A)"]] = df[["P(H)","P(D)","P(A)"]].div(s, axis=0)
    return df.sort_values("Date").reset_index(drop=True)

def expected_points_table(pred: pd.DataFrame) -> pd.DataFrame:
    pred = pred.copy()
    pred["home_ep"] = 3*pred["P(H)"] + 1*pred["P(D)"]
    pred["away_ep"] = 3*pred["P(A)"] + 1*pred["P(D)"]
    ep_home = pred.groupby("HomeTeam")["home_ep"].sum()
    ep_away = pred.groupby("AwayTeam")["away_ep"].sum()
    table = ep_home.add(ep_away, fill_value=0.0).rename("exp_points").reset_index()
    table = table.rename(columns={"HomeTeam":"Team"})
    table["exp_points"] = table["exp_points"].round(3)
    table = table.sort_values(["exp_points","Team"], ascending=[False, True]).reset_index(drop=True)
    table.insert(0, "Rank_exp", np.arange(1, len(table)+1))
    return table

def simulate_table(pred: pd.DataFrame, n_sims=20000, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = sorted(set(pred["HomeTeam"]).union(set(pred["AwayTeam"])))
    t2i = {t:i for i,t in enumerate(teams)}
    nT = len(teams); nM = len(pred)

    pH = pred["P(H)"].to_numpy(); pD = pred["P(D)"].to_numpy(); pA = pred["P(A)"].to_numpy()
    cum1 = pH; cum2 = pH + pD
    U = rng.random((n_sims, nM))
    outcomes = np.where(U < cum1, 0, np.where(U < cum2, 1, 2))  # 0=H,1=D,2=A

    pts = np.zeros((n_sims, nT), dtype=np.float64)
    hidx = pred["HomeTeam"].map(t2i).to_numpy()
    aidx = pred["AwayTeam"].map(t2i).to_numpy()
    for m in range(nM):
        o = outcomes[:, m]
        pts[:, hidx[m]] += (o == 0)*3 + (o == 1)*1
        pts[:, aidx[m]] += (o == 2)*3 + (o == 1)*1

    # break ties with tiny noise so ranks are well-defined
    pts_tb = pts + rng.uniform(0, 1e-6, size=pts.shape)
    order = np.argsort(-pts_tb, axis=1)  # indices sorted by points desc
    ranks = np.empty_like(order)
    for s in range(n_sims):
        ranks[s, order[s]] = np.arange(1, nT+1)

    out = pd.DataFrame({
        "Team": teams,
        "mean_points": pts.mean(axis=0),
        "median_points": np.median(pts, axis=0),
        "std_points": pts.std(axis=0),
        "avg_rank": ranks.mean(axis=0),
        "rank_5pct": np.percentile(ranks, 5, axis=0),
        "rank_95pct": np.percentile(ranks, 95, axis=0),
        "p_title": (ranks == 1).mean(axis=0),
        "p_top4":  (ranks <= 4).mean(axis=0),
        "p_top6":  (ranks <= 6).mean(axis=0),
        "p_relegation": (ranks >= (nT-2)).mean(axis=0)  # bottom 3
    }).sort_values(["mean_points","Team"], ascending=[False, True]).reset_index(drop=True)

    for c in ["mean_points","median_points","std_points","avg_rank","rank_5pct","rank_95pct"]:
        out[c] = out[c].round(2)
    for c in ["p_title","p_top4","p_top6","p_relegation"]:
        out[c] = (out[c]*100).round(1)  # percentages
    out.insert(0, "Rank_mean", np.arange(1, len(out)+1))
    return out

def main():
    pred = load_predictions(PRED_PATH)

    ep = expected_points_table(pred)
    ep_path = OUTDIR / "table_expected_points.csv"
    ep.to_csv(ep_path, index=False)
    print(f"Saved expected-points table -> {ep_path}")

    sim = simulate_table(pred, n_sims=20000, seed=42)  # drop to 5000 if your PC is slow
    sim_path = OUTDIR / "table_sim_summary.csv"
    sim.to_csv(sim_path, index=False)
    print(f"Saved simulation summary -> {sim_path}")

    print("\nTop 5 by mean points:")
    print(sim.head(5)[["Rank_mean","Team","mean_points","p_title","p_top4","p_relegation"]])

if __name__ == "__main__":
    main()