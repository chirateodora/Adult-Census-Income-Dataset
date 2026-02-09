"""
Naive Bayes module for Task 6.
Simple, fast baseline classification using Gaussian Naive Bayes.
Compares scenarios with and without feature engineering (same as Task 1).
"""

import logging
import os
from typing import Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from .preprocessing import DataPreprocessor
from .utils import ModelEvaluator


class NaiveBayesBaseline:
    """
    Task 6: Naive Bayes baseline.
    Runs two scenarios:
      - without feature engineering
      - with feature engineering
    Saves trained models and returns a results table.
    """

    def __init__(
        self,
        data_dir: str = "data",
        results_dir: str = "results",
        logger: Optional[logging.Logger] = None
    ):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.logger = logger or logging.getLogger("DataMining")

        self.models_dir = os.path.join(self.results_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        self.preprocessor = DataPreprocessor(data_dir=self.data_dir, logger=self.logger)
        self.evaluator = ModelEvaluator(logger=self.logger)

    def _run_scenario(self, scenario_name: str, with_fe: bool) -> Dict[str, Any]:
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"Running Naive Bayes scenario: {scenario_name}")
        self.logger.info("=" * 60)

        X_train, X_test, y_train, y_test, feature_names = self.preprocessor.preprocess_for_classification(
            with_feature_engineering=with_fe
        )

        model = GaussianNB()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # GaussianNB supports predict_proba
        y_prob = None
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except Exception:
            pass

        metrics = self.evaluator.calculate_metrics(y_test, y_pred, y_prob)
        self.evaluator.log_metrics(f"GaussianNB ({scenario_name})", metrics)

        # Save model
        model_path = os.path.join(self.models_dir, f"GaussianNB_{scenario_name}.joblib")
        joblib.dump(model, model_path)
        self.logger.info(f"Saved model: {model_path}")

        return {
            "Model": "GaussianNB",
            "Scenario": scenario_name,
            **metrics
        }

    def run(self) -> pd.DataFrame:
        """
        Run Task 6 and return a summary DataFrame.
        """
        self.logger.info("Starting Naive Bayes Baseline (Task 6)")

        results = []
        results.append(self._run_scenario("without_fe", with_fe=False))
        results.append(self._run_scenario("with_fe", with_fe=True))

        df = pd.DataFrame(results)

        # If calculate_metrics returns rounded floats already, fine; if not, keep consistent:
        for col in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
            if col in df.columns:
                df[col] = df[col].astype(float).round(4)

        return df
