# AGENTS.md

## Project Overview

This project builds machine learning models to predict whether a stock is bullish or not. The model will be trained on historical market data, technical indicators, and index-level features.

Current model family:

* XGBoost Classifier

Current target:

```python
Target = (PeakPercInNext15Sessions >= 5)
```

Meaning:

* Class 1 = Stock gains at least 5% within the next 15 trading sessions.
* Class 0 = Otherwise.

The primary objective is to rank stocks by probability of achieving the target.

---

# Repository Structure

## model_helper.py

Responsible for:

* Feature engineering
* Target generation
* Feature selection

Key functions:

```python
prepare_features(data, feature_cols)
create_target(data)
get_features(training_data)
```

## xgboost_model.py

Responsible for:

* Model training
* Model persistence
* Prediction
* Evaluation
* Reporting

Key methods:

```python
train()
predict()
evaluate()
save()
load()
```

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

Current benchmark:

```text
Nifty 50
```

Index columns are suffixed using:

```python
suffixes=('', '_index')
```

Example:

```text
Rsi
Rsi_index

Sma50
Sma50_index
```

Only Nifty 50 rows should be used when creating index features.

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

Models should be saved after training.

Preferred approach:

```python
model.save_model(path)
```

Loading:

```python
model.load_model(path)
```

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
