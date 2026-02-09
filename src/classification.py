"""
Classification module for Task 1.
Multi-algorithm classification with feature engineering comparison and hyperparameter tuning.
"""

import logging
import os
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from .preprocessing import DataPreprocessor
from .utils import ModelEvaluator, PlotUtils, save_results_csv, setup_logging


class ClassificationPipeline:
    """
    Multi-algorithm classification pipeline with GridSearchCV tuning.
    Compares scenarios with and without feature engineering.
    """
    
    # Hyperparameter grids for each algorithm
    PARAM_GRIDS = {
        'DecisionTree': {
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'LogisticRegression': {
            # sklearn 1.8+: use l1_ratio instead of deprecated penalty
            # l1_ratio=0 -> L2 regularization
            # l1_ratio=1 -> L1 regularization  
            # 0<l1_ratio<1 -> ElasticNet (mix of L1 and L2)
            'C': [0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.0, 0.5, 1.0],
            'max_iter': [5000]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
    }
    
    def __init__(
        self,
        data_dir: str = 'data',
        results_dir: str = 'results',
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the classification pipeline.
        
        Args:
            data_dir: Directory containing the data files
            results_dir: Directory to save results
            logger: Optional logger instance
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.models_dir = os.path.join(results_dir, 'models')
        self.plots_dir = os.path.join(results_dir, 'plots')
        self.reports_dir = os.path.join(results_dir, 'reports')
        
        self.logger = logger or logging.getLogger('DataMining')
        self.preprocessor = DataPreprocessor(data_dir, logger)
        self.evaluator = ModelEvaluator(logger)
        self.plot_utils = PlotUtils(self.plots_dir, logger)
        
        # Create output directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        self.results = []
        self.best_models = {}
        
    def _get_base_models(self) -> Dict[str, Any]:
        """
        Get base model instances.
        
        Returns:
            Dictionary of model name -> model instance
        """
        return {
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'LogisticRegression': LogisticRegression(
                random_state=42, 
                max_iter=5000,  # Higher for convergence with SAGA
                solver='saga'  # saga supports l1_ratio parameter
            ),
            'XGBoost': XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1
            )
        }
    
    def train_with_gridsearch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str,
        model: Any,
        cv: int = 5
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train a model with GridSearchCV hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_name: Name of the model
            model: Model instance
            cv: Number of cross-validation folds
            
        Returns:
            Tuple of (best model, best parameters)
        """
        self.logger.info(f"Training {model_name} with GridSearchCV...")
        
        param_grid = self.PARAM_GRIDS[model_name]
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        self.logger.info(f"{model_name} best params: {grid_search.best_params_}")
        self.logger.info(f"{model_name} best CV score: {grid_search.best_score_:.4f}")
        self.logger.info(f"{model_name} training time: {train_time:.2f}s")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        scenario: str
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            scenario: Scenario name (e.g., 'without_fe', 'with_fe')
            
        Returns:
            Dictionary with evaluation results
        """
        y_pred = model.predict(X_test)
        
        # Get probability predictions if available
        y_prob = None
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = self.evaluator.calculate_metrics(y_test, y_pred, y_prob)
        self.evaluator.log_metrics(f"{model_name} ({scenario})", metrics)
        
        result = {
            'Model': model_name,
            'Scenario': scenario,
            **metrics
        }
        
        return result
    
    def run_scenario(
        self,
        with_feature_engineering: bool
    ) -> List[Dict[str, Any]]:
        """
        Run classification for a specific scenario.
        
        Args:
            with_feature_engineering: Whether to use feature engineering
            
        Returns:
            List of results for all models
        """
        scenario = 'with_fe' if with_feature_engineering else 'without_fe'
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Running scenario: {scenario}")
        self.logger.info(f"{'='*60}")
        
        # Preprocess data
        X_train, X_test, y_train, y_test, feature_names = self.preprocessor.preprocess_for_classification(
            with_feature_engineering=with_feature_engineering
        )
        
        scenario_results = []
        base_models = self._get_base_models()
        
        for model_name, model in base_models.items():
            # Train with GridSearch
            best_model, best_params = self.train_with_gridsearch(
                X_train, y_train, model_name, model
            )
            
            # Evaluate
            result = self.evaluate_model(
                best_model, X_test, y_test, model_name, scenario
            )
            result['Best_Params'] = str(best_params)
            scenario_results.append(result)
            
            # Store best model
            key = f"{model_name}_{scenario}"
            self.best_models[key] = {
                'model': best_model,
                'feature_names': feature_names,
                'params': best_params
            }
            
            # Save model
            model_path = os.path.join(self.models_dir, f"{key}.joblib")
            joblib.dump(best_model, model_path)
            self.logger.info(f"Saved model: {model_path}")
        
        return scenario_results
    
    def plot_feature_importance(self, scenario: str = 'without_fe') -> None:
        """
        Plot feature importance for tree-based models.
        
        Args:
            scenario: Scenario to plot for
        """
        for model_name in ['RandomForest', 'XGBoost']:
            key = f"{model_name}_{scenario}"
            if key in self.best_models:
                model_data = self.best_models[key]
                model = model_data['model']
                feature_names = model_data['feature_names']
                
                importances = model.feature_importances_
                
                fig = self.plot_utils.plot_feature_importance(
                    feature_names,
                    importances,
                    title=f"{model_name} Feature Importance ({scenario})",
                    top_n=10
                )
                self.plot_utils.save_figure(fig, f"feature_importance_{model_name}_{scenario}")
    
    def run(self) -> pd.DataFrame:
        """
        Run the complete classification pipeline.
        
        Returns:
            DataFrame with all results
        """
        self.logger.info("Starting Classification Pipeline (Task 1)")
        
        # Run both scenarios
        results_without_fe = self.run_scenario(with_feature_engineering=False)
        results_with_fe = self.run_scenario(with_feature_engineering=True)
        
        # Combine results
        all_results = results_without_fe + results_with_fe
        results_df = self.evaluator.create_comparison_table(all_results)
        
        # Save results
        save_results_csv(results_df, 'classification_results', self.reports_dir)
        
        # Plot feature importance for both scenarios
        self.plot_feature_importance('without_fe')
        self.plot_feature_importance('with_fe')
        
        # Create comparison plot
        metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
        results_df['Model_Scenario'] = results_df['Model'] + ' (' + results_df['Scenario'] + ')'
        
        fig = self.plot_utils.plot_comparison_bars(
            results_df,
            'Model_Scenario',
            metrics_cols,
            title='Classification Model Comparison'
        )
        self.plot_utils.save_figure(fig, 'classification_comparison')
        
        # Check if target accuracy (85%) is met
        best_accuracy = results_df['Accuracy'].max()
        if best_accuracy >= 0.85:
            self.logger.info(f"Target accuracy (85%) ACHIEVED! Best: {best_accuracy:.4f}")
        else:
            self.logger.warning(f"Target accuracy (85%) not met. Best: {best_accuracy:.4f}")
        
        self.logger.info("Classification Pipeline completed!")
        
        return results_df


def run_classification_task(
    data_dir: str = 'data',
    results_dir: str = 'results'
) -> pd.DataFrame:
    """
    Convenience function to run the classification task.
    
    Args:
        data_dir: Directory containing the data files
        results_dir: Directory to save results
        
    Returns:
        DataFrame with results
    """
    logger = setup_logging()
    pipeline = ClassificationPipeline(data_dir, results_dir, logger)
    return pipeline.run()


if __name__ == '__main__':
    results = run_classification_task()
    print(results)
