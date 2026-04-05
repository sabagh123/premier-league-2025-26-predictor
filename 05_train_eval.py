# src/05_train_eval.py
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score
from joblib import dump

IN = Path("data/processed/train_frame.parquet")
ART = Path("artifacts"); ART.mkdir(parents=True, exist_ok=True)
REP = Path("reports");   REP.mkdir(parents=True, exist_ok=True)

TARGET = "FTR"

def feature_cols(df):
    # columns we must NEVER use as features
    drop_exact = {
        "Date","HomeTeam","AwayTeam","FTR","season",
        "B365H","B365D","B365A",          # raw odds (we use normalized probs instead)
        "FTHG","FTAG"                     # *** leak: final goals ***
    }
    drop_prefixes = ("FT", "HT")          # any other full/half-time stat variants

    cols = []
    for c in df.columns:
        if c in drop_exact: 
            continue
        if any(c.startswith(p) for p in drop_prefixes):
            continue
        if df[c].dtype == "O":            # non-numerics
            continue
        cols.append(c)
    return cols

def season_cv(df, min_history_seasons=3):
    seasons = sorted(df["season"].unique())
    # start testing only after we have enough prior seasons
    folds = [s for s in seasons if s > seasons[0] + min_history_seasons]
    if len(folds) < 3:
        raise SystemExit("Not enough seasons for CV; reduce min_history_seasons or add more data.")
    holdout = folds[-1]           # most recent season as holdout
    folds = folds[:-1]            # earlier folds for CV
    return folds, holdout

def fit_rf(X, y, seed=42):
    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed
    )
    rf.fit(X, y)
    return rf

def main():
    df = pd.read_parquet(IN).sort_values("Date").reset_index(drop=True)
    Xcols = feature_cols(df)
    folds, holdout = season_cv(df, min_history_seasons=3)

    rows = []
    for s in folds:
        train = df[df["season"] < s]
        test  = df[df["season"] == s]
        # time-aware split for calibration: last 20% of train acts as calibrator
        split = int(len(train)*0.8)
        fit_set, cal_set = train.iloc[:split], train.iloc[split:]

        rf = fit_rf(fit_set[Xcols], fit_set[TARGET])

        # raw
        proba_raw = rf.predict_proba(test[Xcols])
        ll_raw = log_loss(test[TARGET], proba_raw, labels=rf.classes_)
        acc_raw = accuracy_score(test[TARGET], rf.predict(test[Xcols]))

        # calibrated
        cal = CalibratedClassifierCV(rf, method="isotonic", cv="prefit")
        cal.fit(cal_set[Xcols], cal_set[TARGET])
        proba_cal = cal.predict_proba(test[Xcols])
        ll_cal = log_loss(test[TARGET], proba_cal, labels=cal.classes_)
        acc_cal = accuracy_score(test[TARGET], cal.predict(test[Xcols]))

        rows.append({"test_season": s, "n_train": len(train), "n_test": len(test),
                     "logloss_raw": ll_raw, "logloss_cal": ll_cal,
                     "accuracy_raw": acc_raw, "accuracy_cal": acc_cal})
        print(f"Season {s}: logloss {ll_raw:.3f}→{ll_cal:.3f} | acc {acc_raw:.3f}→{acc_cal:.3f}")

    cv = pd.DataFrame(rows).sort_values("test_season")
    cv.to_csv(REP/"cv_metrics_rf.csv", index=False)
    print(f"CV metrics -> {REP/'cv_metrics_rf.csv'}")

    # final fit on all pre-holdout seasons
    train_all = df[df["season"] < holdout]
    split = int(len(train_all)*0.8)
    fit_set, cal_set = train_all.iloc[:split], train_all.iloc[split:]
    rf_final = fit_rf(fit_set[Xcols], fit_set[TARGET])
    cal_final = CalibratedClassifierCV(rf_final, method="isotonic", cv="prefit")
    cal_final.fit(cal_set[Xcols], cal_set[TARGET])

    dump(rf_final, ART/"rf_model.joblib")
    dump(cal_final, ART/"rf_calibrated.joblib")

    # holdout evaluation
    hold = df[df["season"] == holdout]
    ll_hold_raw = log_loss(hold[TARGET], rf_final.predict_proba(hold[Xcols]), labels=rf_final.classes_)
    ll_hold_cal = log_loss(hold[TARGET],  cal_final.predict_proba(hold[Xcols]), labels=cal_final.classes_)
    acc_hold_raw = accuracy_score(hold[TARGET], rf_final.predict(hold[Xcols]))
    acc_hold_cal = accuracy_score(hold[TARGET], cal_final.predict(hold[Xcols]))

    with open(REP/"holdout_rf.txt","w") as f:
        f.write(f"Holdout season: {holdout}\nRows: {len(hold)}\n")
        f.write(f"LogLoss raw: {ll_hold_raw:.4f}\nLogLoss cal: {ll_hold_cal:.4f}\n")
        f.write(f"Accuracy raw: {acc_hold_raw:.4f}\nAccuracy cal: {acc_hold_cal:.4f}\n")
    print(f"Holdout report -> {REP/'holdout_rf.txt'}")

if __name__ == "__main__":
    main()