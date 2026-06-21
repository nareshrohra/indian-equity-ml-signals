# indian-equity-ml-signals

ML pipeline for generating trading signals on Indian equities — trains, compares, and
runs classifier models (XGBoost, LightGBM, CatBoost, Random Forest, Logistic Regression)
over technical indicators and Nifty 50 / India VIX market context.

Currently ships a **bullish** classifier (`Target = stock gains ≥5% within the next 15
sessions, before a -2.5% stop-loss is hit`). The model/feature/experiment framework is
direction-agnostic, so a bearish (or other) classifier can be added the same way.

## Repository layout

```
src/
  AGENTS.MD              Project conventions for AI coding agents (imported by CLAUDE.MD)
  config/
    feature_configs.json Named feature groups + presets (eq_core, eq_all, eq_nifty, eq_vix, eq_market, eq_full)
    filter_configs.json  Named training-data filters (e.g. high_turnover_only)
  modules/
    data_helper.py        Fetches/caches stock + index (Nifty 50, India VIX) data, runs calculation pipelines
    calc_workers.py       Custom feature/target calculation workers (price, volume, target/stop-loss)
    model_helper.py       Feature column selection, feature prep, target creation, training-data filters
    models.py             ClassifierModel base class + XGBoost/LightGBM/CatBoost/RandomForest/LogisticRegression
                           subclasses, registered via ClassifierModelsRegistry
  ui/
    model_training.ipynb          Batch-trains a model × feature-config grid, optionally under a named experiment
    model_training_ui.py
    trained_model_dashboard.ipynb View/compare trained models, ROC AUC, feature importance, per experiment
    trained_model_dashboard_ui.py
    run_predictions.ipynb         Load a saved model and run it against live data
    ui_helper.py

trained_models/   (gitignored) Saved models + metadata.json per model×config, optionally grouped by experiment
cache-data/       (gitignored) Cached raw/processed market data (large)
```

Notebooks `chdir` into `src/` on the first cell, so all relative paths in the `modules`/`ui`
code are resolved from `src/` (e.g. `trained_models` is referenced as `../trained_models`).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r src/requirements.txt
```

Requires a Jupyter kernel (VS Code Jupyter extension or `jupyter lab`) pointed at `.venv`.

## Workflow

1. **Train** — open `src/ui/model_training.ipynb` → `model_training.ipynb` calls
   `model_training_ui.render()`. Pick a data end date / training cutoff, optionally name an
   experiment and apply a training filter (e.g. `high_turnover_only`), select which
   model × feature-config combinations to run, and start training. Each run is saved to
   `trained_models/bullish/classifiers/[<experiment>/]<model>_<config>/` with
   `model.joblib`, `metadata.json` (stats, feature importances), and evaluation plots.
2. **Compare** — open `src/ui/trained_model_dashboard.ipynb` to inspect ROC AUC,
   classification reports, confusion matrices, and feature importance across all
   models in an experiment, with CSV export.
3. **Predict** — open `src/ui/run_predictions.ipynb` to load a saved model and score
   live data, then filter results by date/probability in the same notebook.

## Models & feature configs

| Model key | Algorithm |
|---|---|
| `xgboost` | XGBClassifier |
| `lgbm` | LGBMClassifier |
| `catboost` | CatBoostClassifier |
| `random_forest` | RandomForestClassifier |
| `logistic_regression` | LogisticRegression (+ StandardScaler) |

| Feature config | Adds on top of `eq_core` |
|---|---|
| `eq_core` | momentum, trend, volume, price/gap (no Bollinger, no market context) |
| `eq_all` | `eq_core` + Bollinger Band features |
| `eq_nifty` | `eq_core` + Nifty 50 context/momentum/trend |
| `eq_vix` | `eq_core` + India VIX context/momentum/trend |
| `eq_market` | `eq_core` + Nifty + VIX context/momentum |
| `eq_full` | everything |

See `src/AGENTS.MD` for full feature/target/data-leakage conventions.
