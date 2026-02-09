"""
Task 7: k-Nearest Neighbors (KNN) classifier.
Simple and intuitive classification technique.
Runs two scenarios: without_fe and with_fe (same preprocessing pipeline).
"""

import logging
import os
from typing import Optional, Dict, Any, List

import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from .preprocessing import DataPreprocessor
from .utils import ModelEvaluator


class KNNStudy:
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

    def _train_and_eval(self, scenario_name: str, with_fe: bool, k_values: List[int]) -> Dict[str, Any]:
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"Running KNN scenario: {scenario_name}")
        self.logger.info("=" * 60)

        X_train, X_test, y_train, y_test, feature_names = self.preprocessor.preprocess_for_classification(
            with_feature_engineering=with_fe
        )

        # KNN is distance-based => scaling matters (you already scale in preprocessing)
        model = KNeighborsClassifier()

        param_grid = {"n_neighbors": k_values}
        grid = GridSearchCV(model, param_grid=param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        self.logger.info(f"KNN best params: {grid.best_params_}")
        self.logger.info(f"KNN best CV score: {grid.best_score_:.4f}")

        y_pred = best_model.predict(X_test)

        # KNN has predict_proba if weights are uniform/distance and algorithm supports it (it does)
        y_prob = None
        try:
            y_prob = best_model.predict_proba(X_test)[:, 1]
        except Exception:
            pass

        metrics = self.evaluator.calculate_metrics(y_test, y_pred, y_prob)
        self.evaluator.log_metrics(f"KNN ({scenario_name})", metrics)

        model_path = os.path.join(self.models_dir, f"KNN_{scenario_name}.joblib")
        joblib.dump(best_model, model_path)
        self.logger.info(f"Saved model: {model_path}")

        return {
            "Model": "KNN",
            "Scenario": scenario_name,
            "Best_k": grid.best_params_.get("n_neighbors"),
            **metrics
        }

    def run(self) -> pd.DataFrame:
        self.logger.info("Starting KNN Classifier (Task 7)")

        k_values = [3, 5, 7, 9, 11]

        results = []
        results.append(self._train_and_eval("without_fe", with_fe=False, k_values=k_values))
        results.append(self._train_and_eval("with_fe", with_fe=True, k_values=k_values))

        df = pd.DataFrame(results)
        for col in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
            if col in df.columns:
                df[col] = df[col].astype(float).round(4)


        return df
