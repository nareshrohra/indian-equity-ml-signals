from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import matplotlib.pyplot as plt

import os
import json
import joblib
import pandas as pd
import numpy as np
from modules.model_helper import get_feature_columns, prepare_features, create_target

class ClassifierModel:

    def __init__(self, feature_columns: list):
        self.feature_columns = feature_columns
        self.model = None

    def train(self, training_data: pd.DataFrame):
        # =========================================================
        # PREPARE DATA
        # =========================================================

        training_data = create_target(training_data)

        training_data = prepare_features(training_data, self.feature_columns)

        # =========================================================
        # FEATURE MATRIX + TARGET
        # =========================================================

        X = training_data[self.feature_columns]

        y = training_data['Target']

        # =========================================================
        # TIME SERIES TRAIN / TEST SPLIT
        # VERY IMPORTANT
        # =========================================================

        X_train = X
        y_train = y
        
        # =========================================================
        # HANDLE CLASS IMBALANCE
        # =========================================================

        negative_count = (y_train == 0).sum()

        positive_count = (y_train == 1).sum()

        self.scale_pos_weight = scale_pos_weight = negative_count / positive_count

        print("\nscale_pos_weight:", scale_pos_weight)

        # =========================================================
        # BUILD MODEL
        # =========================================================

        model = self.get_model()

        # =========================================================
        # TRAIN MODEL
        # =========================================================

        print("\nTraining model...\n")

        model.fit(X_train, y_train)

        print("\nModel training completed.\n")
        self.model = model

        self.show_training_report()

    def get_feature_importance(self) -> dict:
        return dict(zip(self.feature_columns,
                        self.model.feature_importances_.tolist()))

    def get_model(self):
        raise NotImplementedError("get_model() is not implemented")

    def save_to_dir(self, directory: str, metadata: dict):
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self, os.path.join(directory, 'model.joblib'))
        with open(os.path.join(directory, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    def save_eval_plots(self, directory: str, y_true, y_preds, y_probs):
        from sklearn.metrics import (
            roc_curve, auc,
            precision_recall_curve, average_precision_score,
        )
        os.makedirs(directory, exist_ok=True)

        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        ap_score = average_precision_score(y_true, y_probs)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0].set_title('ROC Curve (Bullish Signals)')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].legend(loc='lower right')
        axes[0].grid(True)
        axes[1].plot(recall, precision, color='green', lw=2, label=f'AP = {ap_score:.3f}')
        axes[1].set_title('Precision-Recall Curve')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].legend(loc='lower left')
        axes[1].grid(True)
        plt.tight_layout()
        fig.savefig(os.path.join(directory, 'roc_pr_curves.png'), dpi=100, bbox_inches='tight')
        plt.close(fig)

        cm = confusion_matrix(y_true, y_preds)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14,
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Bearish', 'Bullish'])
        ax.set_yticklabels(['Bearish', 'Bullish'])
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.tight_layout()
        fig.savefig(os.path.join(directory, 'confusion_matrix.png'), dpi=100, bbox_inches='tight')
        plt.close(fig)    
            
    def evaluate(self, data: pd.DataFrame):
        data = prepare_features(data, self.feature_columns)
        data = create_target(data)
        y_preds, y_probs = self.predict(data)
        y_true = data['Target']
        self.show_prediction_report(y_true, y_preds, y_probs, data)
        return y_true, y_preds, y_probs

    @staticmethod
    def load_from_dir(directory: str):
        return joblib.load(os.path.join(directory, 'model.joblib'))

    def predict(self, data: pd.DataFrame):
        data = prepare_features(data, self.feature_columns)
        x_tests = data[self.feature_columns]
        y_preds = pd.Series(self.model.predict(x_tests), index=data.index)
        y_probs = pd.Series(self.model.predict_proba(x_tests)[:, 1], index=data.index)
        return y_preds, y_probs

    def show_training_report(self):
        # =========================================================
        # FEATURE IMPORTANCE
        # =========================================================
        importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.model.feature_importances_
        })
        importance_df = importance_df.sort_values(
            by='Importance',
            ascending=False
        )
        print("\n===================================")
        print("FEATURE IMPORTANCE")
        print("===================================")
        print(importance_df)

    def show_prediction_report(self, y_tests, y_preds, y_probs, data):
        self.show_stats(y_tests, y_preds, y_probs, data)
        self.display_roc_curve(y_tests, y_probs)

    def show_stats(self, y_tests, y_preds, y_probs, data):
        print("===================================")
        print("CLASSIFICATION REPORT")
        print("===================================")
        print(classification_report(y_tests, y_preds))

        print("\n===================================")
        print("CONFUSION MATRIX")
        print("===================================")
        print(confusion_matrix(y_tests, y_preds))

        print("\n===================================")
        print("ROC AUC SCORE")
        print("===================================")
        print(roc_auc_score(y_tests, y_probs))

        # =========================================================
        # TOP PREDICTIONS
        # =========================================================
        test_df = data.copy()
        test_df['PredictionProbability'] = y_probs
        top_predictions = test_df[[
            'Identifier',
            'Date',
            'Close',
            'PredictionProbability'
        ]].sort_values(
            by='PredictionProbability',
            ascending=False
        )
        print("\n===================================")
        print("TOP PREDICTED BULLISH SETUPS")
        print("===================================")
        print(top_predictions.head(20))

    def display_roc_curve(self, y_tests, y_probs):
        from sklearn.metrics import (
            roc_curve,
            auc,
            precision_recall_curve,
            average_precision_score
        )

        # =========================
        # ROC CURVE
        # =========================
        fpr, tpr, roc_thresholds = roc_curve(y_tests, y_probs)
        roc_auc = auc(fpr, tpr)

        # =========================
        # PRECISION-RECALL CURVE
        # =========================
        precision, recall, pr_thresholds = precision_recall_curve(y_tests, y_probs)
        ap_score = average_precision_score(y_tests, y_probs)

        # =========================
        # PLOT SIDE-BY-SIDE
        # =========================
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # -------------------------
        # ROC Curve
        # -------------------------
        axes[0].plot(
            fpr,
            tpr,
            color='darkorange',
            lw=2,
            label=f'AUC = {roc_auc:.3f}'
        )

        # Random baseline
        axes[0].plot(
            [0, 1],
            [0, 1],
            color='navy',
            lw=2,
            linestyle='--'
        )

        axes[0].set_title('ROC Curve (Bullish Signals)')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].legend(loc='lower right')
        axes[0].grid(True)

        # -------------------------
        # Precision-Recall Curve
        # -------------------------
        axes[1].plot(
            recall,
            precision,
            color='green',
            lw=2,
            label=f'AP = {ap_score:.3f}'
        )

        axes[1].set_title('Precision-Recall Curve')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].legend(loc='lower left')
        axes[1].grid(True)

        plt.tight_layout()
        from IPython.display import display as _ipy_display
        _ipy_display(fig)
        plt.close(fig)

class XGBoostModel(ClassifierModel):
    def get_model(self):
        from xgboost import XGBClassifier
        return XGBClassifier(

            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,

            subsample=0.8,
            colsample_bytree=0.8,

            objective='binary:logistic',

            scale_pos_weight=self.scale_pos_weight,

            random_state=42,

            eval_metric='logloss',

            n_jobs=-1
        )

class LGBMModel(ClassifierModel):
    def get_model(self):
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=self.scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    def get_feature_importance(self) -> dict:
        importances = self.model.feature_importances_
        total = importances.sum()
        normalized = (importances / total * 100) if total > 0 else importances
        return dict(zip(self.feature_columns, normalized.tolist()))


class CatBoostModel(ClassifierModel):
    def get_model(self):
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=300,
            depth=5,
            learning_rate=0.05,
            subsample=0.8,
            class_weights={0: 1.0, 1: self.scale_pos_weight},
            random_seed=42,
            verbose=0,
        )

    def get_feature_importance(self) -> dict:
        return dict(zip(self.feature_columns, self.model.get_feature_importance().tolist()))


class RandomForestModel(ClassifierModel):
    def get_model(self):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            class_weight={0: 1, 1: self.scale_pos_weight},
            random_state=42,
            n_jobs=-1,
        )

    def get_feature_importance(self) -> dict:
        importances = self.model.feature_importances_
        return dict(zip(self.feature_columns, (importances * 100).tolist()))


class LogisticRegressionModel(ClassifierModel):
    def get_model(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                C=1.0,
                class_weight={0: 1, 1: self.scale_pos_weight},
                max_iter=1000,
                solver='lbfgs',
                random_state=42,
                n_jobs=-1,
            )),
        ])

    def get_feature_importance(self) -> dict:
        coefs = np.abs(self.model.named_steps['clf'].coef_[0])
        total = coefs.sum()
        normalized = (coefs / total * 100) if total > 0 else coefs
        return dict(zip(self.feature_columns, normalized.tolist()))

    def show_training_report(self):
        importance = self.get_feature_importance()
        importance_df = pd.DataFrame(
            list(importance.items()), columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False)
        print("\n===================================")
        print("FEATURE IMPORTANCE (|coef| %)")
        print("===================================")
        print(importance_df)


class ClassifierModelsRegistry:
    _registry: dict = {}

    @classmethod
    def register(cls, name: str, model_class: type) -> None:
        cls._registry[name] = model_class

    @classmethod
    def unregister(cls, name: str) -> None:
        if name not in cls._registry:
            raise KeyError(f'No model registered under "{name}"')
        del cls._registry[name]

    @classmethod
    def get(cls, name: str) -> type:
        if name not in cls._registry:
            raise KeyError(f'No model registered under "{name}"')
        return cls._registry[name]

    @classmethod
    def list(cls) -> dict:
        return dict(cls._registry)


ClassifierModelsRegistry.register('xgboost', XGBoostModel)
ClassifierModelsRegistry.register('lgbm', LGBMModel)
ClassifierModelsRegistry.register('catboost', CatBoostModel)
ClassifierModelsRegistry.register('random_forest', RandomForestModel)
ClassifierModelsRegistry.register('logistic_regression', LogisticRegressionModel)
