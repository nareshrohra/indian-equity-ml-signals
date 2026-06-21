# AGENTS.md

## Project Overview

This project builds machine learning models to predict trading signals for stocks — currently a
**bullish** classifier, with the model/feature/experiment framework designed so a bearish (or other
direction) classifier can be added the same way. Models are trained on historical market data,
technical indicators, and index-level (Nifty 50 / India VIX) features.

Current model families, registered in `modules/models.py` via `ClassifierModelsRegistry`:

* `xgboost` — XGBClassifier
* `lgbm` — LGBMClassifier
* `catboost` — CatBoostClassifier
* `random_forest` — RandomForestClassifier
* `logistic_regression` — LogisticRegression (+ StandardScaler)

All use class-imbalance weighting (`scale_pos_weight` / `class_weight`) since bullish setups are
the minority class.

Current target (computed in `calc_workers.py` / `model_helper.py::create_target`):

```python
TargetHitDay = first session in next 15 where PeakPercInNextNSessions >= +5%
StopHitDay   = first session in next 15 where TroughPercInNextNSessions <= stop-loss threshold

Target = 1 if TargetHitDay is reached, AND
              (StopHitDay was never reached OR TargetHitDay comes before StopHitDay)
Target = 0 otherwise
```

Meaning:

* Class 1 = Stock gains at least 5% within the next 15 trading sessions *before* hitting its
  stop-loss.
* Class 0 = Otherwise (target never hit, or stop-loss hit first).

The primary objective is to rank stocks by probability of achieving the target.

---

# Repository Structure

## modules/data_helper.py

Responsible for:

* Fetching stock + index (Nifty 50, India VIX) data via `markets_insights`
* Running calculation pipelines (technical indicators, target/stop-loss, growth)
* Caching processed data under `cache-data/` (gitignored, can be regenerated)
* Merging index data onto stock rows with suffix `__{identifier}` (e.g. `Rsi__Nifty 50`)

## modules/calc_workers.py

Custom `CalculationWorker` implementations not provided by `markets_insights`:
price/gap features, volume/turnover liquidity features, RSI/Stoch-RSI/SMA threshold
flags, and the `TargetHitDay`/`StopHitDay` workers used to build the target.

## modules/model_helper.py

Responsible for:

* Feature column selection
* Feature preparation (numeric coercion, inf→NaN, dropna)
* Target generation
* Training-data filtering (`filter_training_data`, used for experiments like
  `high_turnover_only`)

Key functions:

```python
get_feature_columns(training_data)
prepare_features(data, feature_cols)
create_target(data)
filter_training_data(data, criteria)
```

## modules/models.py

`ClassifierModel` base class + one subclass per model family (`XGBoostModel`,
`LGBMModel`, `CatBoostModel`, `RandomForestModel`, `LogisticRegressionModel`), registered
in `ClassifierModelsRegistry` under keys `xgboost`, `lgbm`, `catboost`, `random_forest`,
`logistic_regression`.

Key methods:

```python
train(training_data)
predict(data)
evaluate(data)
get_feature_importance()
save_to_dir(directory, metadata)
load_from_dir(directory)          # staticmethod
```

## ui/

Jupyter-notebook-backed UIs (each notebook just `chdir`s to `src/` then calls
`render()` from its matching `_ui.py` module):

* `model_training.ipynb` / `model_training_ui.py` — batch-trains a model × feature-config
  grid, optionally under a named experiment + training filter(s); saves to `trained_models/`.
* `trained_model_dashboard.ipynb` / `trained_model_dashboard_ui.py` — compares trained
  models within an experiment (ROC AUC, classification report, feature importance).
* `run_predictions.ipynb` / `ui_helper.py` — loads a saved model and scores live data.

---

# Data Model

Input data is expected to contain:

```text
Date
Identifier
Open
High
Low
Close
Volume
Turnover
PreviousClose
```

Technical indicators may include:

```text
Rsi
StochRsi_K
StochRsi_D

Sma50
Sma100
Sma200

PriceAboveSma50
PriceAboveSma100
PriceAboveSma200
```

---

# Index Features

Index features are merged into stock-level data.

Current benchmarks:

```text
Nifty 50
India VIX
```

Index columns are suffixed using `__{identifier}` (double underscore), via
`pd.merge(..., suffixes=('', f'__{identifier}'))`:

```text
Rsi
Rsi__Nifty 50
Rsi__India VIX

Sma50
Sma50__Nifty 50
```

Only rows for the named identifier should be used when creating index features.

Avoid many-to-many joins.

Preferred pattern:

```python
nifty_data = index_data[
    index_data["Identifier"] == "Nifty 50"
]
```

---

# Feature Engineering

Generated features include:

## Volatility

```python
DailyRangePerc =
(High - Low) / Close
```

## Gap

```python
GapPerc =
(Open - PreviousClose) / PreviousClose
```

## RSI States

```python
RsiBelow30
RsiAbove75
```

## Stochastic RSI States

```python
StochBelow20
StochAbove80
```

## Trend Features

```python
PriceAboveSma50
PriceAboveSma100
PriceAboveSma200
```

---

# Feature Naming Conventions

Use:

```text
Sma50
Sma100
Sma200

PriceAboveSma50
PriceAboveSma100
PriceAboveSma200
```

Do NOT mix with:

```text
SMA50
SMA100
SMA200
```

Column names are case-sensitive.

---

# Feature Configs

The authoritative feature lists for training live in `config/feature_configs.json`, not
`model_helper.get_feature_columns()` (legacy/unused by the current multi-config training
flow — kept for reference only).

`feature_configs.json` has two sections:

* `features` — named groups of column names (e.g. `eq_momentum`, `nifty_trend`,
  `vix_bollinger`).
* `configs` — named presets, each a list of group names, resolved (de-duplicated) into a
  flat feature-column list by `model_training_ui.py::_load_configs()`.

Current presets, each a superset of `eq_core`:

| Config | Adds on top of `eq_core` |
|---|---|
| `eq_core` | momentum, trend, volume, price/gap |
| `eq_all` | + Bollinger Band features |
| `eq_nifty` | + Nifty 50 context/momentum/trend |
| `eq_vix` | + India VIX context/momentum/trend |
| `eq_market` | + Nifty + VIX context/momentum |
| `eq_full` | everything |

A trained model's name is `{model_key}_{config_key}`, e.g. `catboost_eq_vix`.

Training-data row filters (separate from feature selection) live in
`config/filter_configs.json` as `{name: pandas_query_string}`, applied via
`model_helper.filter_training_data()`. These mark deliberate changes to the training
universe (e.g. `high_turnover_only` → `` `TurnoverSma20PercentileRank` > 0.75 ``) and are
typically paired with a same-named experiment folder under `trained_models/`.

---

# Feature List Rules

FEATURE_COLUMNS must not contain duplicates.

Before model training:

```python
FEATURE_COLUMNS = list(
    dict.fromkeys(FEATURE_COLUMNS)
)
```

Recommended validation:

```python
assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))
```

---

# Missing Value Policy

Feature preparation performs:

```python
replace([np.inf, -np.inf], np.nan)
```

and later:

```python
dropna(subset=FEATURE_COLUMNS)
```

When adding new features:

* Measure row loss before and after dropna.
* Check NA counts per feature.
* Avoid accidentally removing the entire dataset.

Debug pattern:

```python
df[FEATURE_COLUMNS].isna().sum()
```

---

# Model Evaluation

Primary metrics:

## ROC-AUC

Measures ranking quality.

Interpretation:

```text
0.50 = Random
0.60 = Useful signal
0.70 = Strong signal
```

## Precision

Measures accuracy of positive predictions.

## Recall

Measures capture rate of positive opportunities.

---

# Preferred Evaluation Metrics

In addition to ROC-AUC:

* Precision@Top5%
* Precision@Top10%
* Precision@Top20%
* Average PeakPercInNext15Sessions by probability decile
* Top-decile hit rate
* Top-minus-bottom performance spread

For stock-selection use cases, ranking metrics are more important than raw accuracy.

---

# Model Persistence

Models are saved (joblib pickle of the whole `ClassifierModel` instance, plus a
`metadata.json` with stats/feature importances/eval plots) under
`trained_models/bullish/classifiers/[<experiment>/]<model>_<config>/` at the project
root (gitignored — not checked into version control).

Saving:

```python
model.save_to_dir(directory, metadata)
```

Loading:

```python
model = ClassifierModel.load_from_dir(directory)
```

An **experiment** is just a named subfolder grouping one batch of model×config runs —
used to mark a deliberate change in training universe (e.g. `high_turnover_only`,
filtered via `config/filter_configs.json`), features, or target definition.

Model comparisons must always be performed on the same evaluation dataset.

---

# Common Failure Modes

## Duplicate Feature Names

Symptoms:

```text
AttributeError:
'DataFrame' object has no attribute 'dtype'
```

Cause:

Duplicate columns in FEATURE_COLUMNS.

---

## Empty Evaluation Dataset

Symptoms:

```text
ValueError:
Found empty input array
```

Cause:

dropna removed all rows.

---

## Many-to-Many Merge Explosion

Symptoms:

```text
Unable to allocate memory
```

Cause:

Merging stock data with multiple index rows per date.

Fix:

Filter index data to Nifty 50 before merging.

---

# Development Principles

1. Prevent data leakage.
2. Preserve chronological ordering.
3. Train only on historical information available at prediction time.
4. Use the same feature engineering logic for:

   * Training
   * Evaluation
   * Inference
5. Keep feature names consistent.
6. Validate feature lists before model training.
7. Prefer ranking quality over raw classification accuracy.
