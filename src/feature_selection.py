"""
Feature selection study module for Task 5.
Compares Chi-square, Mutual Information, and RFE methods.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.preprocessing import MinMaxScaler

from .preprocessing import DataPreprocessor
from .utils import ModelEvaluator, PlotUtils, save_results_csv, setup_logging


class FeatureSelectionStudy:
    """
    Feature selection comparison study.
    Compares Chi-square, Mutual Information, and RFE methods.
    """
    
    # K values to test
    K_VALUES = [5, 8, 10, 12]
    
    def __init__(
        self,
        data_dir: str = 'data',
        results_dir: str = 'results',
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the feature selection study.
        
        Args:
            data_dir: Directory containing the data files
            results_dir: Directory to save results
            logger: Optional logger instance
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.plots_dir = os.path.join(results_dir, 'plots')
        self.reports_dir = os.path.join(results_dir, 'reports')
        
        self.logger = logger or logging.getLogger('DataMining')
        self.preprocessor = DataPreprocessor(data_dir, logger)
        self.evaluator = ModelEvaluator(logger)
        self.plot_utils = PlotUtils(self.plots_dir, logger)
        
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        self.results = []
        self.selected_features = {}
        
    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Load and prepare data for feature selection.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names)
        """
        self.logger.info("Loading and preprocessing data...")
        
        X_train, X_test, y_train, y_test, feature_names = \
            self.preprocessor.preprocess_for_classification(with_feature_engineering=False)
        
        # Store for later use
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = np.array(feature_names)
        
        self.logger.info(f"Data shape: train={X_train.shape}, test={X_test.shape}")
        
        return X_train, X_test, y_train, y_test, feature_names
    
    def evaluate_with_selection(
        self,
        X_train_selected: np.ndarray,
        X_test_selected: np.ndarray,
        method_name: str,
        k: int
    ) -> Dict[str, Any]:
        """
        Train RandomForest on selected features and evaluate.
        
        Args:
            X_train_selected: Training data with selected features
            X_test_selected: Test data with selected features
            method_name: Name of the selection method
            k: Number of features selected
            
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        
        # Train RandomForest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_selected, self.y_train)
        
        train_time = time.time() - start_time
        
        # Predict and evaluate
        y_pred = rf.predict(X_test_selected)
        y_prob = rf.predict_proba(X_test_selected)[:, 1]
        
        metrics = self.evaluator.calculate_metrics(self.y_test, y_pred, y_prob)
        
        result = {
            'Method': method_name,
            'k': k,
            'Accuracy': metrics['Accuracy'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1': metrics['F1'],
            'ROC-AUC': metrics['ROC-AUC'],
            'Train_Time_s': round(train_time, 3)
        }
        
        return result
    
    def run_chi_square(self) -> List[Dict[str, Any]]:
        """
        Run Chi-square feature selection for all k values.
        
        Returns:
            List of results for each k
        """
        self.logger.info("Running Chi-square feature selection...")
        
        results = []
        self.selected_features['Chi-square'] = {}
        
        # Make features non-negative for chi2 (use MinMaxScaler)
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        for k in self.K_VALUES:
            if k > X_train_scaled.shape[1]:
                self.logger.warning(f"k={k} exceeds number of features, skipping")
                continue
                
            selector = SelectKBest(chi2, k=k)
            X_train_selected = selector.fit_transform(X_train_scaled, self.y_train)
            X_test_selected = selector.transform(X_test_scaled)
            
            # Get selected feature names
            selected_mask = selector.get_support()
            selected_names = self.feature_names[selected_mask]
            self.selected_features['Chi-square'][k] = set(selected_names)
            
            result = self.evaluate_with_selection(
                X_train_selected, X_test_selected, 'Chi-square', k
            )
            results.append(result)
            
            self.logger.info(f"Chi-square k={k}: Accuracy={result['Accuracy']:.4f}")
        
        return results
    
    def run_mutual_information(self) -> List[Dict[str, Any]]:
        """
        Run Mutual Information feature selection for all k values.
        
        Returns:
            List of results for each k
        """
        self.logger.info("Running Mutual Information feature selection...")
        
        results = []
        self.selected_features['Mutual_Information'] = {}
        
        for k in self.K_VALUES:
            if k > self.X_train.shape[1]:
                self.logger.warning(f"k={k} exceeds number of features, skipping")
                continue
                
            selector = SelectKBest(mutual_info_classif, k=k)
            X_train_selected = selector.fit_transform(self.X_train, self.y_train)
            X_test_selected = selector.transform(self.X_test)
            
            # Get selected feature names
            selected_mask = selector.get_support()
            selected_names = self.feature_names[selected_mask]
            self.selected_features['Mutual_Information'][k] = set(selected_names)
            
            result = self.evaluate_with_selection(
                X_train_selected, X_test_selected, 'Mutual_Information', k
            )
            results.append(result)
            
            self.logger.info(f"Mutual Information k={k}: Accuracy={result['Accuracy']:.4f}")
        
        return results
    
    def run_rfe(self) -> List[Dict[str, Any]]:
        """
        Run RFE feature selection for all k values.
        
        Returns:
            List of results for each k
        """
        self.logger.info("Running RFE feature selection...")
        
        results = []
        self.selected_features['RFE'] = {}
        
        for k in self.K_VALUES:
            if k > self.X_train.shape[1]:
                self.logger.warning(f"k={k} exceeds number of features, skipping")
                continue
            
            # Use smaller RF for RFE efficiency
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rfe = RFE(rf, n_features_to_select=k, step=0.1)
            
            start_time = time.time()
            X_train_selected = rfe.fit_transform(self.X_train, self.y_train)
            fit_time = time.time() - start_time
            
            X_test_selected = rfe.transform(self.X_test)
            
            # Get selected feature names
            selected_mask = rfe.support_
            selected_names = self.feature_names[selected_mask]
            self.selected_features['RFE'][k] = set(selected_names)
            
            # Evaluate
            start_time = time.time()
            rf_eval = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf_eval.fit(X_train_selected, self.y_train)
            train_time = time.time() - start_time + fit_time
            
            y_pred = rf_eval.predict(X_test_selected)
            y_prob = rf_eval.predict_proba(X_test_selected)[:, 1]
            
            metrics = self.evaluator.calculate_metrics(self.y_test, y_pred, y_prob)
            
            result = {
                'Method': 'RFE',
                'k': k,
                'Accuracy': metrics['Accuracy'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1': metrics['F1'],
                'ROC-AUC': metrics['ROC-AUC'],
                'Train_Time_s': round(train_time, 3)
            }
            results.append(result)
            
            self.logger.info(f"RFE k={k}: Accuracy={result['Accuracy']:.4f}")
        
        return results
    
    def find_common_features(self, k: int = 10) -> Set[str]:
        """
        Find features selected by all methods at a specific k.
        
        Args:
            k: Number of features
            
        Returns:
            Set of common feature names
        """
        common = None
        
        for method, k_dict in self.selected_features.items():
            if k in k_dict:
                if common is None:
                    common = k_dict[k]
                else:
                    common = common.intersection(k_dict[k])
        
        if common is None:
            common = set()
        
        self.logger.info(f"\nCommon features selected by all methods (k={k}):")
        for feature in sorted(common):
            self.logger.info(f"  - {feature}")
        
        return common
    
    def plot_accuracy_comparison(self, results_df: pd.DataFrame) -> None:
        """
        Create line plot comparing accuracy vs k for all methods.
        
        Args:
            results_df: DataFrame with all results
        """
        self.logger.info("Creating accuracy comparison plot...")
        
        # Prepare data for plotting
        y_series = {}
        for method in results_df['Method'].unique():
            method_data = results_df[results_df['Method'] == method].sort_values('k')
            y_series[method] = method_data['Accuracy'].tolist()
        
        k_values = sorted(results_df['k'].unique())
        
        fig = self.plot_utils.plot_line_comparison(
            k_values,
            y_series,
            x_label='Number of Features (k)',
            y_label='Accuracy',
            title='Feature Selection: Accuracy vs Number of Features'
        )
        
        self.plot_utils.save_figure(fig, 'feature_selection_accuracy_comparison')
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete feature selection study.
        
        Returns:
            Dictionary with all results
        """
        self.logger.info("Starting Feature Selection Study (Task 5)")
        
        # Load data
        self.load_and_prepare_data()
        
        # Run all methods
        chi2_results = self.run_chi_square()
        mi_results = self.run_mutual_information()
        rfe_results = self.run_rfe()
        
        # Combine results
        all_results = chi2_results + mi_results + rfe_results
        results_df = pd.DataFrame(all_results).round(4)
        
        # Save results
        save_results_csv(results_df, 'feature_selection_comparison', self.reports_dir)
        
        # Plot comparison
        self.plot_accuracy_comparison(results_df)
        
        # Find common features at k=10
        common_features = self.find_common_features(k=10)
        
        # Save common features
        common_df = pd.DataFrame({
            'Common_Features_k10': list(common_features)
        })
        save_results_csv(common_df, 'common_features_k10', self.reports_dir)
        
        # Save all selected features per method at k=10
        features_k10 = {}
        for method, k_dict in self.selected_features.items():
            if 10 in k_dict:
                features_k10[method] = list(k_dict[10])
        
        # Create comparison DataFrame
        max_features = max(len(v) for v in features_k10.values()) if features_k10 else 0
        features_comparison = {}
        for method, features in features_k10.items():
            padded = features + [''] * (max_features - len(features))
            features_comparison[method] = padded
        
        features_df = pd.DataFrame(features_comparison)
        save_results_csv(features_df, 'selected_features_k10', self.reports_dir)
        
        self.logger.info("Feature Selection Study completed!")
        
        return {
            'results': results_df,
            'common_features_k10': common_features,
            'selected_features': self.selected_features
        }


def run_feature_selection_task(
    data_dir: str = 'data',
    results_dir: str = 'results'
) -> Dict[str, Any]:
    """
    Convenience function to run the feature selection task.
    
    Args:
        data_dir: Directory containing the data files
        results_dir: Directory to save results
        
    Returns:
        Dictionary with results
    """
    logger = setup_logging()
    study = FeatureSelectionStudy(data_dir, results_dir, logger)
    return study.run()


if __name__ == '__main__':
    results = run_feature_selection_task()
    print(results['results'])
