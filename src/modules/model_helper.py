# =========================================================
# FEATURE ENGINEERING METHOD
# Reusable for:
# - Training Data
# - Evaluation Data
# - Live/Inference Data
# =========================================================

import pandas as pd
import numpy as np

# =========================================================
# FEATURE COLUMN LIST
# Use SAME list during training & inference
# =========================================================

def get_feature_columns(training_data: pd.DataFrame):
    feature_cols = [
        'Open',
        'High',
        'Low',
        'Close',
        'Volume',
        'Turnover'
    ]

    feature_cols += [
        col for col in training_data.columns
        if (
            "above" in col
            or "below" in col
            or "Crossed" in col
            or col.replace('__', '') in [
                'Rsi',
                'StochRsi_K',
                'StochRsi_D',
                'SMA50',
                'SMA100',
                'SMA200',
                'PriceAboveSMA50',
                'PriceAboveSMA100',
                'PriceAboveSMA200'
            ]
        )
    ]
    feature_cols += [
        # Custom Features
        'DailyRangePerc',
        'GapPerc',

        # RSI states
        'RsiBelow30',
        'RsiAbove75',

        # Stoch states
        'StochBelow20',
        'StochAbove80',

        # Trend Features
        'Sma50',
        'Sma100',
        'Sma200',

        'PriceAboveSma50',
        'PriceAboveSma100',
        'PriceAboveSma200'
    ]
    return feature_cols

# =========================================================
# MAIN FEATURE PREPARATION FUNCTION
# =========================================================
def prepare_features(data: pd.DataFrame, feature_cols: list):

    df = data.copy()

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # -----------------------------------------
    # Convert features to numeric
    # -----------------------------------------

    for col in feature_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        )

    # -----------------------------------------
    # Remove bad rows
    # -----------------------------------------

    df = df.dropna(
        subset=feature_cols
    )

    # -----------------------------------------
    # Return cleaned dataframe
    # -----------------------------------------

    return df

# =========================================================
# TARGET CREATION METHOD
# ONLY FOR TRAINING DATA
# =========================================================
def create_target(data: pd.DataFrame):
    if 'Target' in data.columns:
        return data

    df = data.copy()
    target_hit_no_stop = df['TargetHitDay'].notna() & df['StopHitDay'].isna()
    target_hits_first = (
        df['TargetHitDay'].notna() &
        df['StopHitDay'].notna() &
        (df['TargetHitDay'] < df['StopHitDay'])
    )
    df['Target'] = (target_hit_no_stop | target_hits_first).astype(int)
    return df

# =========================================================
# TRAINING DATA FILTERING
# ONLY FOR TRAINING DATA
# =========================================================
def filter_training_data(data: pd.DataFrame, criteria: list[str]) -> pd.DataFrame:
    df = data
    for query in criteria:
        df = df.query(query)
    return df