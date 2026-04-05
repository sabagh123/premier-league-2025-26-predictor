# Premier League 2025-26 Predictor

A machine learning project that forecasts the **2025-26 Premier League season** using historical football data, feature engineering, probability calibration, and season simulation.

This project builds match-level predictions and then uses those probabilities to simulate the whole season and project the final table, including estimated chances of winning the title, finishing in the top 4 or top 6, and being relegated.

## Project overview

The pipeline does the following:

1. collects or parses fixture data
2. builds a Premier League dataset from historical information
3. creates predictive features
4. trains and evaluates multiple models
5. calibrates probabilities
6. predicts upcoming fixtures
7. simulates the full season to estimate final table outcomes

The main models used are:

- **Random Forest**
- **LightGBM**
- calibrated probability outputs for better simulation quality

## Repository structure

- `src/` – main pipeline scripts
- `data/` – raw and processed datasets
- `notebooks/` – exploration and experimentation
- `reports/` – evaluation outputs and metrics
- `predictions/` – fixture predictions and season table projections
- `artifacts/` – saved trained models and calibration files

## Main scripts

The `src/` folder contains the core workflow:

- `00_parse_espn_fixtures.py` – parses fixture data
- `01_make_pl_dataset.py` – builds the main dataset
- `02_lock_base.py` – locks or freezes a base dataset version
- `03_make_features.py` – creates model features
- `05_train_eval.py` – trains and evaluates a Random Forest model
- `05_train_eval_lgbm.py` – trains and evaluates a LightGBM model
- `06_predict_fixtures_lgbm.py` – predicts 2025-26 fixtures
- `07_table_projection.py` – simulates season outcomes and builds projected tables

## Outputs

This project produces outputs such as:

- fixture-level prediction files
- cross-validation metrics
- holdout evaluation summaries
- expected points table
- simulated final table summary
- probabilities for:
  - title win
  - top 4 finish
  - top 6 finish
  - relegation

## Example use case

Instead of only predicting one match at a time, this project tries to answer bigger season-level questions such as:

- Who is most likely to win the league?
- Which teams are most likely to finish in the Champions League places?
- Which teams are most at risk of relegation?
- How stable is each team’s projected finish?

## Tech stack

- Python
- pandas
- NumPy
- scikit-learn
- LightGBM
- joblib
- Excel / CSV outputs for reports and predictions

## How to run

A typical workflow is:

```bash
python src/00_parse_espn_fixtures.py
python src/01_make_pl_dataset.py
python src/02_lock_base.py
python src/03_make_features.py
python src/05_train_eval.py
python src/05_train_eval_lgbm.py
python src/06_predict_fixtures_lgbm.py
python src/07_table_projection.py
