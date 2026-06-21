---
name: analyze-results
description: Analyze trained-model comparison results (models_comparison.csv and features_comparison.csv) for the bullish-stock classifier project. Use when the user asks to compare models, find the best model/feature config, explain why a model performs the way it does, or identify which features matter most.
---

# Analyzing model & feature comparison results

## Project context

This project trains classifiers to predict whether an Indian equity will be
**bullish**: `Target = (PeakPercInNext15Sessions >= 5)` — i.e. the stock gains
at least 5% within the next 15 trading sessions (class 1 = bullish, class 0 =
not). The goal is to **rank** stocks by probability of hitting that target,
not just to classify accurately — ranking metrics (ROC AUC, precision at the
top of the probability distribution) matter more than raw accuracy.

Training data is daily OHLCV + technical indicators (RSI, StochRSI, SMA50/100/200,
Bollinger Bands, volume/turnover ratios, gap%, daily range%) per stock, optionally
joined with **Nifty 50** (broad market) and **India VIX** (volatility) context
columns suffixed `__Nifty 50` / `__India VIX`. Train/test split is chronological
(a cutoff date), never random — this is a time-series problem and there must be
no leakage from the future into training.

Full feature/target/data-leakage conventions are documented in `src/agents.md`
(imported into `src/CLAUDE.md`) — read it if you need definitions of a specific
feature name.

## What was trained

Every row in `models_comparison.csv` is one **(model family) × (feature config)**
combination, all trained/evaluated on the *same* chronological split so they're
directly comparable.

**Model families** (`src/modules/models.py`, `ClassifierModelsRegistry`):

| Key | Algorithm | Notes |
|---|---|---|
| `xgboost` | XGBClassifier | 300 trees, depth 5, lr 0.05 |
| `lgbm` | LGBMClassifier | 300 trees, depth 5, lr 0.05 |
| `catboost` | CatBoostClassifier | 300 iterations, depth 5, lr 0.05 |
| `random_forest` | RandomForestClassifier | 300 trees, depth 10, min_leaf 5 |
| `logistic_regression` | LogisticRegression (+StandardScaler) | linear baseline |

All use class-imbalance weighting (`scale_pos_weight` / `class_weight`) since
bullish setups are the minority class.

**Feature configs** (`src/config/feature_configs.json`), each a superset of the
last:

| Config | Adds on top of `eq_core` |
|---|---|
| `eq_core` | base: OHLCV, momentum (RSI/StochRSI), trend (SMA), volume, price/gap — no Bollinger, no market context |
| `eq_all` | `eq_core` + Bollinger Band features |
| `eq_nifty` | `eq_core` + Nifty 50 context/momentum/trend |
| `eq_vix` | `eq_core` + India VIX context/momentum/trend |
| `eq_market` | `eq_core` + Nifty context/momentum + VIX context/momentum |
| `eq_full` | everything: `eq_all` + Nifty + VIX (context/momentum/trend/Bollinger) |

A model name like `catboost_eq_vix` = CatBoost trained on the `eq_vix` feature set.

## `models_comparison.csv` schema

One row per model×config. Key columns:

- `Model` — `{model_family}_{feature_config}`
- `Saved`, `Cutoff` — when saved, and the train/test chronological split date
- `Train rows` / `Test rows`
- `ROC AUC (train)` / `ROC AUC (test)` — **primary metric**; compare test, and
  watch the train→test gap as an overfitting signal (e.g. `eq_full`/`eq_market`
  often score highest on train but drop more on test — classic overfitting from
  added market-context features)
- `F1 (1, train)` / `F1 (1, test)` — F1 for the bullish class
- `Prec (0)/Rec (0)/F1 (0)` and `Prec (1)/Rec (1)` — precision/recall per class
  at the default 0.5 threshold (class 1 = bullish, the minority/interesting class)
- `Accuracy` — least important metric here; the classes are imbalanced so a
  high accuracy can still mean a model that rarely flags bullish setups
- `TN, FP, FN, TP` — raw confusion-matrix counts on the test set

When ranking models, prefer **`ROC AUC (test)`** first, then check the
train/test gap and `Prec (1)`/`Rec (1)` trade-off depending on whether the
user cares more about precision (fewer false bullish calls) or recall
(catching more real opportunities).

## `features_comparison.csv` schema

A pivot table: **rows = feature names**, **columns = model×config combos**
(same naming as above), **values = that feature's importance for that
specific trained model**.

Important caveats:

- A **blank/NaN cell** means the feature wasn't part of that config (e.g.
  `Close__Nifty 50` is blank for all `eq_core`/`eq_all`/`eq_vix` columns since
  those configs exclude Nifty context) — it does *not* mean zero importance.
- Importance scales differ by model family and are only safely comparable
  **within a column**, not across columns:
  - `xgboost`, `catboost` — native gain-based importance
  - `lgbm`, `random_forest` — split-count/gain importance, normalized to ~100
  - `logistic_regression` — `|coefficient|` magnitude, normalized to ~100
- To find genuinely robust signals, look for features that rank highly
  **across many model families and configs** (not just one column) — a
  feature that only one model likes is more likely overfitting/noise.

## Which experiment to analyze

Trained models are saved under `trained_models/bullish/classifiers/` (project root), optionally
inside an **experiment** folder the user names at training time (see
`model_training_ui.py`'s "Experiment name" field). An experiment groups one
batch of model×config combinations trained under the same setup — it can also
mark a deliberate change in dataset/universe (e.g. "high turnover stocks
only"), features, or target definition, not just a retrain date.
`models_comparison.csv` / `features_comparison.csv` are generated *inside
whichever folder is currently selected* in the Trained Model Dashboard, so
there can be one such pair per experiment, plus possibly one at the root.

**Before analyzing, always ask the user which experiment they mean** — don't
assume the root. List the candidates first:

```python
import os
base = 'trained_models/bullish/classifiers'
for d in sorted(os.listdir(base)):
    full = os.path.join(base, d)
    if os.path.isdir(full):
        print(d)
```

Folders containing `metadata.json` directly are individual models saved at the
root (no experiment folder used); folders containing *other* folders (each
with their own `metadata.json`) are experiments worth asking about. If the
user says "the latest" or doesn't know the name, sort candidates by
modification time and confirm with them before proceeding rather than
silently guessing.

## Suggested analysis workflow

1. Load both CSVs with pandas (`index_col=0` for `features_comparison.csv`)
   from the confirmed experiment's path.
2. Sort `models_comparison.csv` by `ROC AUC (test)` descending — identify the
   best model overall, and the best model *per family* (does the best config
   change by model type?) and *per config* (does the best model type change
   by config?).
3. Compute `ROC AUC (train) - ROC AUC (test)` as an overfitting gap column;
   flag configs/models where it's unusually large.
4. For features: for each feature, average its rank (not raw value, since
   scales differ) across columns to find consistently important features;
   separately note features that are highly important in only one model
   family (interesting outliers, possibly model-specific quirks).
5. Compare configs holding model family fixed (e.g. all `xgboost_*` rows) to
   answer "does adding Nifty/VIX context actually help this model?" — a
   config that helps `xgboost` but hurts `logistic_regression` suggests an
   interaction the linear model can't exploit, not a useless feature.
6. Frame conclusions in terms of the project's actual goal: ranking quality
   (ROC AUC, top-feature consistency) over raw accuracy, per `src/agents.md`.

If the user wants a before/after comparison across training runs, ask which
two experiments to compare (e.g. an older run like `_old/` vs. a newer named
experiment) and repeat the workflow for each before contrasting results.
