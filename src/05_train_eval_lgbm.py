# src/05_train_eval_lgbm.py
import numpy as np, pandas as pd
from pathlib import Path
from lightgbm import LGBMClassifier 
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score
from joblib import dump

IN = Path("data/processed/train_frame.parquet")
ART = Path("artifacts"); ART.mkdir(parents=True, exist_ok=True)
REP = Path("reports");   REP.mkdir(parents=True, exist_ok=True)
TARGET = "FTR"

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

def season_cv(df, min_history_seasons=3):
    seasons = sorted(df["season"].unique())
    folds = [s for s in seasons if s > seasons[0] + min_history_seasons]
    if len(folds) < 3: raise SystemExit("Not enough seasons for CV.")
    holdout = folds[-1]; folds = folds[:-1]
    return folds, holdout

def fit_lgbm(X_train, y_train, X_val, y_val, seed=42):
    clf = LGBMClassifier(
        objective="multiclass",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_data_in_leaf=25,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        random_state=seed,
        n_jobs=-1,
    )
    # Handle both LightGBM API styles
    try:
        # newer style: use callbacks
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
        )
    except TypeError:
        # older style: early_stopping_rounds kwarg
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            early_stopping_rounds=150
        )
    return clf

def main():
    # 1) Load features and set up columns/splits
    df = pd.read_parquet(IN).sort_values("Date").reset_index(drop=True)
    Xcols = feature_cols(df)
    folds, holdout = season_cv(df)  # expanding-time CV; last full season is holdout

    # 2) Cross-validation over seasons
    rows = []
    for s in folds:
        train = df[df["season"] < s].copy()
        test  = df[df["season"] == s].copy()

        # 70% fit  | 15% early-stop validation | 15% separate calibration
        n = len(train)
        i_fit = int(n * 0.70)
        i_val = int(n * 0.85)
        fit_set = train.iloc[:i_fit]
        val_set = train.iloc[i_fit:i_val]
        cal_set = train.iloc[i_val:]

        lgbm = fit_lgbm(
            fit_set[Xcols], fit_set[TARGET],
            val_set[Xcols], val_set[TARGET]
        )

        # raw metrics on the test season
        proba_raw = lgbm.predict_proba(test[Xcols])
        ll_raw = log_loss(test[TARGET], proba_raw, labels=lgbm.classes_)
        acc_raw = accuracy_score(test[TARGET], lgbm.predict(test[Xcols]))

        # calibrate with a separate set (Platt/sigmoid is safer than isotonic here)
        cal = CalibratedClassifierCV(lgbm, method="sigmoid", cv="prefit")
        cal.fit(cal_set[Xcols], cal_set[TARGET])
        proba_cal = cal.predict_proba(test[Xcols])
        ll_cal = log_loss(test[TARGET], proba_cal, labels=cal.classes_)
        acc_cal = accuracy_score(test[TARGET], cal.predict(test[Xcols]))

        rows.append({
            "test_season": s, "n_train": len(train), "n_test": len(test),
            "logloss_raw": ll_raw, "logloss_cal": ll_cal,
            "accuracy_raw": acc_raw, "accuracy_cal": acc_cal
        })
        print(f"Season {s}: logloss {ll_raw:.3f}→{ll_cal:.3f} | acc {acc_raw:.3f}→{acc_cal:.3f}")

    # save CV summary
    cv = pd.DataFrame(rows).sort_values("test_season")
    cv.to_csv(REP / "cv_metrics_lgbm.csv", index=False)
    print(f"CV metrics -> {REP / 'cv_metrics_lgbm.csv'}")

    # 3) Final model on all pre-holdout seasons, with separate val & calib
    train_all = df[df["season"] < holdout].copy()
    n = len(train_all)
    i_fit = int(n * 0.70)
    i_val = int(n * 0.85)
    fit_set = train_all.iloc[:i_fit]
    val_set = train_all.iloc[i_fit:i_val]
    cal_set = train_all.iloc[i_val:]

    lgbm_final = fit_lgbm(
        fit_set[Xcols], fit_set[TARGET],
        val_set[Xcols], val_set[TARGET]
    )
    cal_final = CalibratedClassifierCV(lgbm_final, method="sigmoid", cv="prefit")
    cal_final.fit(cal_set[Xcols], cal_set[TARGET])

    dump(lgbm_final, ART / "lgbm_model.joblib")
    dump(cal_final, ART / "lgbm_calibrated.joblib")

    # 4) Evaluate on holdout season
    hold = df[df["season"] == holdout].copy()
    ll_hold_raw = log_loss(hold[TARGET], lgbm_final.predict_proba(hold[Xcols]), labels=lgbm_final.classes_)
    ll_hold_cal = log_loss(hold[TARGET], cal_final.predict_proba(hold[Xcols]), labels=cal_final.classes_)
    acc_hold_raw = accuracy_score(hold[TARGET], lgbm_final.predict(hold[Xcols]))
    acc_hold_cal = accuracy_score(hold[TARGET], cal_final.predict(hold[Xcols]))

    with open(REP / "holdout_lgbm.txt", "w") as f:
        f.write(f"Holdout season: {holdout}\nRows: {len(hold)}\n")
        f.write(f"LogLoss raw: {ll_hold_raw:.4f}\nLogLoss cal: {ll_hold_cal:.4f}\n")
        f.write(f"Accuracy raw: {acc_hold_raw:.4f}\nAccuracy cal: {acc_hold_cal:.4f}\n")
    print(f"Holdout report -> {REP / 'holdout_lgbm.txt'}")

if __name__ == "__main__":
    main()