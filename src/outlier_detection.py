"""
Outlier detection module for Task 2.
Analyzes outliers in high-income data using IsolationForest and LocalOutlierFactor.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch

from .preprocessing import DataPreprocessor
from .utils import PlotUtils, save_results_csv, setup_logging


class OutlierAnalyzer:
    """
    Outlier detection analyzer for high-income individuals.
    Uses IsolationForest and LocalOutlierFactor methods.
    """
    
    # Features to use for outlier detection
    FEATURES = ['age', 'hours-per-week', 'capital-gain']
    
    def __init__(
        self,
        data_dir: str = 'data',
        results_dir: str = 'results',
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the outlier analyzer.
        
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
        self.plot_utils = PlotUtils(self.plots_dir, logger)
        
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.data = None
        self.outlier_results = {}
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Load and prepare high-income data for outlier detection.
        
        Returns:
            DataFrame with high-income individuals
        """
        self.data = self.preprocessor.get_high_income_data()
        self.logger.info(f"Loaded {len(self.data)} high-income instances")
        
        # Scale features for outlier detection
        self.data_scaled = self.scaler.fit_transform(self.data[self.FEATURES])
        
        return self.data
    
    def detect_outliers_isolation_forest(
        self,
        contamination: float = 0.1,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Detect outliers using Isolation Forest.
        
        Args:
            contamination: Expected proportion of outliers
            random_state: Random seed
            
        Returns:
            Array of predictions (-1 for outliers, 1 for inliers)
        """
        self.logger.info("Detecting outliers with Isolation Forest...")
        
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        
        predictions = iso_forest.fit_predict(self.data_scaled)
        
        n_outliers = (predictions == -1).sum()
        self.logger.info(f"Isolation Forest detected {n_outliers} outliers ({n_outliers/len(predictions)*100:.1f}%)")
        
        self.outlier_results['IsolationForest'] = predictions
        
        return predictions
    
    def detect_outliers_lof(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.1
    ) -> np.ndarray:
        """
        Detect outliers using Local Outlier Factor.
        
        Args:
            n_neighbors: Number of neighbors for LOF
            contamination: Expected proportion of outliers
            
        Returns:
            Array of predictions (-1 for outliers, 1 for inliers)
        """
        self.logger.info("Detecting outliers with Local Outlier Factor...")
        
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            n_jobs=-1
        )
        
        predictions = lof.fit_predict(self.data_scaled)
        
        n_outliers = (predictions == -1).sum()
        self.logger.info(f"LOF detected {n_outliers} outliers ({n_outliers/len(predictions)*100:.1f}%)")
        
        self.outlier_results['LOF'] = predictions
        
        return predictions
    
    def plot_outliers(self) -> None:
        """
        Create scatter plots showing outliers detected by both methods.
        """
        self.logger.info("Creating outlier visualization plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Colors: -1 (outlier) = red, 1 (inlier) = blue
        colors = {-1: 'red', 1: 'blue'}
        
        for idx, (method_name, predictions) in enumerate(self.outlier_results.items()):
            ax = axes[idx]
            
            # Create color array
            color_array = [colors[p] for p in predictions]
            
            ax.scatter(
                self.data['age'],
                self.data['hours-per-week'],
                c=color_array,
                alpha=0.5,
                s=50
            )
            
            ax.set_xlabel('Age')
            ax.set_ylabel('Hours per Week')
            ax.set_title(f'{method_name}: Age vs Hours per Week\n(Red = Outlier, Blue = Normal)')
            
            # Add legend
            legend_elements = [
                Patch(facecolor='blue', alpha=0.5, label='Normal'),
                Patch(facecolor='red', alpha=0.5, label='Outlier')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        self.plot_utils.save_figure(fig, 'outlier_detection_scatter')
        
        # Additional 3D plot with capital-gain
        self._plot_3d_outliers()
    
    def _plot_3d_outliers(self) -> None:
        """
        Create 3D scatter plot showing outliers with all three features.
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(16, 6))
        
        for idx, (method_name, predictions) in enumerate(self.outlier_results.items()):
            ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
            
            # Separate inliers and outliers
            inlier_mask = predictions == 1
            outlier_mask = predictions == -1
            
            # Plot inliers
            ax.scatter(
                self.data.loc[inlier_mask, 'age'],
                self.data.loc[inlier_mask, 'hours-per-week'],
                self.data.loc[inlier_mask, 'capital-gain'],
                c='blue', alpha=0.3, s=20, label='Normal'
            )
            
            # Plot outliers
            ax.scatter(
                self.data.loc[outlier_mask, 'age'],
                self.data.loc[outlier_mask, 'hours-per-week'],
                self.data.loc[outlier_mask, 'capital-gain'],
                c='red', alpha=0.7, s=50, label='Outlier'
            )
            
            ax.set_xlabel('Age')
            ax.set_ylabel('Hours per Week')
            ax.set_zlabel('Capital Gain')
            ax.set_title(f'{method_name}: 3D Outlier View')
            ax.legend()
        
        plt.tight_layout()
        self.plot_utils.save_figure(fig, 'outlier_detection_3d')
    
    def calculate_statistics(self) -> pd.DataFrame:
        """
        Calculate statistics for outliers vs normal instances.
        
        Returns:
            DataFrame with statistics
        """
        stats_list = []
        
        for method_name, predictions in self.outlier_results.items():
            outlier_mask = predictions == -1
            normal_mask = predictions == 1
            
            # Outlier statistics
            outlier_stats = {
                'Method': method_name,
                'Group': 'Outliers',
                'Count': outlier_mask.sum(),
                'Age_Mean': self.data.loc[outlier_mask, 'age'].mean(),
                'Age_Std': self.data.loc[outlier_mask, 'age'].std(),
                'Hours_Mean': self.data.loc[outlier_mask, 'hours-per-week'].mean(),
                'Hours_Std': self.data.loc[outlier_mask, 'hours-per-week'].std(),
                'CapitalGain_Mean': self.data.loc[outlier_mask, 'capital-gain'].mean(),
                'CapitalGain_Std': self.data.loc[outlier_mask, 'capital-gain'].std()
            }
            stats_list.append(outlier_stats)
            
            # Normal statistics
            normal_stats = {
                'Method': method_name,
                'Group': 'Normal',
                'Count': normal_mask.sum(),
                'Age_Mean': self.data.loc[normal_mask, 'age'].mean(),
                'Age_Std': self.data.loc[normal_mask, 'age'].std(),
                'Hours_Mean': self.data.loc[normal_mask, 'hours-per-week'].mean(),
                'Hours_Std': self.data.loc[normal_mask, 'hours-per-week'].std(),
                'CapitalGain_Mean': self.data.loc[normal_mask, 'capital-gain'].mean(),
                'CapitalGain_Std': self.data.loc[normal_mask, 'capital-gain'].std()
            }
            stats_list.append(normal_stats)
        
        stats_df = pd.DataFrame(stats_list).round(2)
        self.logger.info("\nOutlier Statistics:")
        self.logger.info(f"\n{stats_df.to_string()}")
        
        return stats_df
    
    def find_extreme_anomalies(self, top_n: int = 3) -> pd.DataFrame:
        """
        Find the most extreme anomalies based on unusual patterns.
        
        Extreme: young age, high hours, high capital gain
        
        Args:
            top_n: Number of top anomalies to return
            
        Returns:
            DataFrame with top anomalies
        """
        self.logger.info(f"Finding top {top_n} extreme anomalies...")
        
        # Use Isolation Forest predictions for this
        if 'IsolationForest' not in self.outlier_results:
            self.detect_outliers_isolation_forest()
        
        predictions = self.outlier_results['IsolationForest']
        outlier_mask = predictions == -1
        outliers = self.data[outlier_mask].copy()
        
        if len(outliers) == 0:
            self.logger.warning("No outliers found!")
            return pd.DataFrame()
        
        # Calculate anomaly score: young age + high hours + high capital gain
        # Lower age is more unusual for high income, higher hours and capital gain are unusual
        outliers['age_normalized'] = 1 - (outliers['age'] - outliers['age'].min()) / (outliers['age'].max() - outliers['age'].min() + 1)
        outliers['hours_normalized'] = (outliers['hours-per-week'] - outliers['hours-per-week'].min()) / (outliers['hours-per-week'].max() - outliers['hours-per-week'].min() + 1)
        outliers['gain_normalized'] = (outliers['capital-gain'] - outliers['capital-gain'].min()) / (outliers['capital-gain'].max() - outliers['capital-gain'].min() + 1)
        
        outliers['anomaly_score'] = outliers['age_normalized'] + outliers['hours_normalized'] + outliers['gain_normalized']
        
        # Get top anomalies
        top_anomalies = outliers.nlargest(top_n, 'anomaly_score')[
            ['age', 'hours-per-week', 'capital-gain', 'occupation', 'education', 'anomaly_score']
        ]
        
        self.logger.info(f"\nTop {top_n} Extreme Anomalies:")
        self.logger.info(f"\n{top_anomalies.to_string()}")
        
        return top_anomalies
    
    def run(self) -> Dict[str, pd.DataFrame]:
        """
        Run the complete outlier detection analysis.
        
        Returns:
            Dictionary with results DataFrames
        """
        self.logger.info("Starting Outlier Detection Analysis (Task 2)")
        
        # Load data
        self.load_and_prepare_data()
        
        # Detect outliers with both methods
        self.detect_outliers_isolation_forest()
        self.detect_outliers_lof()
        
        # Create visualizations
        self.plot_outliers()
        
        # Calculate statistics
        stats_df = self.calculate_statistics()
        save_results_csv(stats_df, 'outlier_statistics', self.reports_dir)
        
        # Find extreme anomalies
        extreme_anomalies = self.find_extreme_anomalies(top_n=3)
        save_results_csv(extreme_anomalies, 'extreme_anomalies', self.reports_dir)
        
        self.logger.info("Outlier Detection Analysis completed!")
        
        return {
            'statistics': stats_df,
            'extreme_anomalies': extreme_anomalies
        }


def run_outlier_detection_task(
    data_dir: str = 'data',
    results_dir: str = 'results'
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to run the outlier detection task.
    
    Args:
        data_dir: Directory containing the data files
        results_dir: Directory to save results
        
    Returns:
        Dictionary with results
    """
    logger = setup_logging()
    analyzer = OutlierAnalyzer(data_dir, results_dir, logger)
    return analyzer.run()


if __name__ == '__main__':
    results = run_outlier_detection_task()
    print(results['statistics'])
