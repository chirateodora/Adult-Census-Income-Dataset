"""
Utility functions for the Data Mining project.
Common evaluation metrics, plotting utilities, and logging setup.
"""

import logging
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, 
    classification_report
)


def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return the project logger.
    
    Args:
        log_level: Logging level (default INFO)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger('DataMining')
    return logger


class ModelEvaluator:
    """
    Class for evaluating machine learning models.
    Calculates standard metrics: Accuracy, Precision, Recall, F1, ROC-AUC.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the evaluator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger('DataMining')
    
    def calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities for positive class (for ROC-AUC)
            
        Returns:
            Dictionary with metric names and values
        """
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'F1': f1_score(y_true, y_pred, average='binary', zero_division=0)
        }
        
        if y_prob is not None:
            try:
                metrics['ROC-AUC'] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics['ROC-AUC'] = np.nan
        else:
            metrics['ROC-AUC'] = np.nan
            
        return metrics
    
    def create_comparison_table(
        self, 
        results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Create a comparison table from multiple model results.
        
        Args:
            results: List of dictionaries with model name and metrics
            
        Returns:
            DataFrame with comparison table
        """
        df = pd.DataFrame(results)
        return df.round(4)
    
    def log_metrics(self, model_name: str, metrics: Dict[str, float]) -> None:
        """
        Log metrics for a model.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics
        """
        self.logger.info(f"Metrics for {model_name}:")
        for name, value in metrics.items():
            self.logger.info(f"  {name}: {value:.4f}")


class PlotUtils:
    """
    Utility class for creating and saving plots.
    """
    
    def __init__(
        self, 
        output_dir: str = 'results/plots',
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize plot utilities.
        
        Args:
            output_dir: Directory to save plots
            logger: Optional logger instance
        """
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger('DataMining')
        os.makedirs(output_dir, exist_ok=True)
        
        # Set default style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def save_figure(self, fig: plt.Figure, filename: str, dpi: int = 150) -> str:
        """
        Save a matplotlib figure to file.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename (without extension)
            dpi: Resolution
            
        Returns:
            Full path to saved file
        """
        filepath = os.path.join(self.output_dir, f"{filename}.png")
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        self.logger.info(f"Saved plot: {filepath}")
        return filepath
    
    def plot_feature_importance(
        self, 
        feature_names: List[str], 
        importances: np.ndarray,
        title: str = "Feature Importance",
        top_n: int = 10
    ) -> plt.Figure:
        """
        Create a horizontal bar plot for feature importance.
        
        Args:
            feature_names: Names of features
            importances: Importance values
            title: Plot title
            top_n: Number of top features to show
            
        Returns:
            Matplotlib figure
        """
        # Sort by importance
        indices = np.argsort(importances)[-top_n:]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            range(len(indices)), 
            importances[indices], 
            color=sns.color_palette("viridis", len(indices))
        )
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title(title)
        
        return fig
    
    def plot_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        labels: List[str] = None,
        title: str = "Confusion Matrix"
    ) -> plt.Figure:
        """
        Create a confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels or ['0', '1'],
            yticklabels=labels or ['0', '1'],
            ax=ax
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title)
        
        return fig
    
    def plot_comparison_bars(
        self, 
        data: pd.DataFrame,
        x_col: str,
        y_cols: List[str],
        title: str = "Model Comparison"
    ) -> plt.Figure:
        """
        Create grouped bar chart for model comparison.
        
        Args:
            data: DataFrame with comparison data
            x_col: Column for x-axis categories
            y_cols: Columns to plot as bars
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        x = np.arange(len(data))
        width = 0.8 / len(y_cols)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, col in enumerate(y_cols):
            offset = (i - len(y_cols)/2 + 0.5) * width
            bars = ax.bar(x + offset, data[col], width, label=col)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(data[x_col], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        return fig
    
    def plot_elbow(
        self, 
        k_values: List[int], 
        inertias: List[float],
        title: str = "Elbow Method"
    ) -> plt.Figure:
        """
        Create an elbow plot for K-Means.
        
        Args:
            k_values: List of k values
            inertias: List of inertia values
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Inertia')
        ax.set_title(title)
        ax.set_xticks(k_values)
        
        return fig
    
    def plot_silhouette(
        self, 
        k_values: List[int], 
        scores: List[float],
        title: str = "Silhouette Scores"
    ) -> plt.Figure:
        """
        Create a silhouette score plot.
        
        Args:
            k_values: List of k values
            scores: List of silhouette scores
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(k_values, scores, color=sns.color_palette("viridis", len(k_values)))
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Silhouette Score')
        ax.set_title(title)
        ax.set_xticks(k_values)
        
        return fig
    
    def plot_scatter_2d(
        self, 
        x: np.ndarray, 
        y: np.ndarray,
        labels: np.ndarray = None,
        x_label: str = "X",
        y_label: str = "Y",
        title: str = "Scatter Plot"
    ) -> plt.Figure:
        """
        Create a 2D scatter plot.
        
        Args:
            x: X coordinates
            y: Y coordinates
            labels: Color labels for points
            x_label: X axis label
            y_label: Y axis label
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if labels is not None:
            scatter = ax.scatter(x, y, c=labels, cmap='viridis', alpha=0.6, s=50)
            plt.colorbar(scatter, ax=ax, label='Cluster/Label')
        else:
            ax.scatter(x, y, alpha=0.6, s=50)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        
        return fig
    
    def plot_line_comparison(
        self, 
        x_values: List[Any],
        y_series: Dict[str, List[float]],
        x_label: str = "X",
        y_label: str = "Y",
        title: str = "Comparison"
    ) -> plt.Figure:
        """
        Create a line plot comparing multiple series.
        
        Args:
            x_values: X axis values
            y_series: Dictionary of series name -> values
            x_label: X axis label
            y_label: Y axis label
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for name, values in y_series.items():
            ax.plot(x_values, values, 'o-', linewidth=2, markersize=8, label=name)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


def save_results_csv(
    data: pd.DataFrame, 
    filename: str, 
    output_dir: str = 'results/reports'
) -> str:
    """
    Save results to CSV file.
    
    Args:
        data: DataFrame to save
        filename: Output filename (without extension)
        output_dir: Output directory
        
    Returns:
        Full path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{filename}.csv")
    data.to_csv(filepath, index=False)
    logging.getLogger('DataMining').info(f"Saved results: {filepath}")
    return filepath
