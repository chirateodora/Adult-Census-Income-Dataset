"""
Clustering module for Task 3.
Implements K-Means and DBSCAN clustering with validation and semantic analysis.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from .preprocessing import DataPreprocessor
from .utils import PlotUtils, save_results_csv, setup_logging


class ClusteringAnalyzer:
    """
    Clustering analyzer using K-Means and DBSCAN.
    Includes validation metrics and semantic analysis.
    """
    
    def __init__(
        self,
        data_dir: str = 'data',
        results_dir: str = 'results',
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the clustering analyzer.
        
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
        
        self.pca = PCA(n_components=5)
        self.scaler = StandardScaler()
        self.data_pca = None
        self.data_2d = None
        self.original_data = None
        self.cluster_results = {}
        
    def load_and_prepare_data(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load and prepare data for clustering.
        
        Returns:
            Tuple of (PCA-reduced data, original DataFrame)
        """
        self.logger.info("Loading and preprocessing data for clustering...")
        
        X_processed, original_df = self.preprocessor.preprocess_for_clustering()
        self.original_data = original_df
        
        # Apply PCA for dimensionality reduction
        self.logger.info("Applying PCA with 5 components...")
        X_scaled = self.scaler.fit_transform(X_processed)
        self.data_pca = self.pca.fit_transform(X_scaled)
        
        # Also create 2D version for visualization
        pca_2d = PCA(n_components=2)
        self.data_2d = pca_2d.fit_transform(X_scaled)
        
        explained_variance = sum(self.pca.explained_variance_ratio_)
        self.logger.info(f"PCA explained variance (5 components): {explained_variance:.2%}")
        
        return self.data_pca, original_df
    
    def run_kmeans_elbow(
        self,
        k_range: range = range(2, 11)
    ) -> Tuple[List[int], List[float]]:
        """
        Run K-Means for elbow method analysis.
        
        Args:
            k_range: Range of k values to test
            
        Returns:
            Tuple of (k values, inertias)
        """
        self.logger.info("Running K-Means elbow method...")
        
        k_values = list(k_range)
        inertias = []
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.data_pca)
            inertias.append(kmeans.inertia_)
            self.logger.info(f"k={k}: inertia={kmeans.inertia_:.2f}")
        
        # Plot elbow
        fig = self.plot_utils.plot_elbow(k_values, inertias, "K-Means Elbow Method")
        self.plot_utils.save_figure(fig, 'kmeans_elbow')
        
        return k_values, inertias
    
    def run_kmeans_silhouette(
        self,
        k_range: range = range(3, 6)
    ) -> Tuple[List[int], List[float]]:
        """
        Run K-Means with silhouette score analysis.
        
        Args:
            k_range: Range of k values to test
            
        Returns:
            Tuple of (k values, silhouette scores)
        """
        self.logger.info("Running K-Means silhouette analysis...")
        
        k_values = list(k_range)
        silhouette_scores = []
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.data_pca)
            score = silhouette_score(self.data_pca, labels)
            silhouette_scores.append(score)
            self.logger.info(f"k={k}: silhouette={score:.4f}")
        
        # Plot silhouette scores
        fig = self.plot_utils.plot_silhouette(k_values, silhouette_scores, "K-Means Silhouette Scores")
        self.plot_utils.save_figure(fig, 'kmeans_silhouette')
        
        # Find best k
        best_k_idx = np.argmax(silhouette_scores)
        best_k = k_values[best_k_idx]
        self.logger.info(f"Best k by silhouette: {best_k} (score: {silhouette_scores[best_k_idx]:.4f})")
        
        return k_values, silhouette_scores
    
    def run_kmeans(self, n_clusters: int = 4) -> np.ndarray:
        """
        Run K-Means clustering with the specified number of clusters.
        
        Args:
            n_clusters: Number of clusters
            
        Returns:
            Cluster labels
        """
        self.logger.info(f"Running K-Means with {n_clusters} clusters...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.data_pca)
        
        self.cluster_results['KMeans'] = labels
        self.logger.info(f"K-Means cluster sizes: {np.bincount(labels)}")
        
        return labels
    
    def run_dbscan(
        self,
        eps_values: List[float] = [0.3, 0.5, 0.7],
        min_samples_values: List[int] = [5, 10]
    ) -> Dict[str, np.ndarray]:
        """
        Run DBSCAN with different parameters.
        
        Args:
            eps_values: List of eps values to test
            min_samples_values: List of min_samples values to test
            
        Returns:
            Dictionary of parameter combination -> labels
        """
        self.logger.info("Running DBSCAN with multiple parameter combinations...")
        
        results = []
        best_score = -1
        best_labels = None
        best_params = None
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(self.data_pca)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = (labels == -1).sum()
                
                # Calculate silhouette score if we have valid clusters
                if n_clusters >= 2 and n_clusters < len(labels) - 1:
                    # Filter out noise points for silhouette
                    mask = labels != -1
                    if mask.sum() > n_clusters:
                        score = silhouette_score(self.data_pca[mask], labels[mask])
                    else:
                        score = -1
                else:
                    score = -1
                
                results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'silhouette': score
                })
                
                self.logger.info(f"DBSCAN(eps={eps}, min_samples={min_samples}): "
                               f"clusters={n_clusters}, noise={n_noise}, silhouette={score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_params = (eps, min_samples)
        
        # Save DBSCAN comparison results
        dbscan_df = pd.DataFrame(results)
        save_results_csv(dbscan_df, 'dbscan_comparison', self.reports_dir)
        
        # Store best result
        if best_labels is not None:
            self.cluster_results['DBSCAN'] = best_labels
            self.logger.info(f"Best DBSCAN params: eps={best_params[0]}, min_samples={best_params[1]}")
        
        return results
    
    def plot_clusters(self) -> None:
        """
        Create 2D scatter plots of clusters for both methods.
        """
        self.logger.info("Creating cluster visualization plots...")
        
        n_methods = len(self.cluster_results)
        fig, axes = plt.subplots(1, n_methods, figsize=(8*n_methods, 6))
        
        if n_methods == 1:
            axes = [axes]
        
        for idx, (method_name, labels) in enumerate(self.cluster_results.items()):
            ax = axes[idx]
            
            # Create scatter with different colors for each cluster
            unique_labels = np.unique(labels)
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
            
            for cluster_id, color in zip(unique_labels, colors):
                mask = labels == cluster_id
                label_text = 'Noise' if cluster_id == -1 else f'Cluster {cluster_id}'
                ax.scatter(
                    self.data_2d[mask, 0],
                    self.data_2d[mask, 1],
                    c=[color],
                    alpha=0.5 if cluster_id == -1 else 0.7,
                    s=30,
                    label=label_text
                )
            
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title(f'{method_name} Clustering (2D PCA View)')
            ax.legend(loc='best', fontsize=8)
        
        plt.tight_layout()
        self.plot_utils.save_figure(fig, 'clustering_comparison')
    
    def analyze_clusters_semantically(self, method: str = 'KMeans') -> pd.DataFrame:
        """
        Perform semantic analysis of clusters.
        
        Calculates mean age, education-num, hours-per-week and top 3 occupations
        for each cluster.
        
        Args:
            method: Clustering method to analyze ('KMeans' or 'DBSCAN')
            
        Returns:
            DataFrame with cluster analysis
        """
        if method not in self.cluster_results:
            self.logger.warning(f"No results for {method}")
            return pd.DataFrame()
        
        labels = self.cluster_results[method]
        data = self.original_data.copy()
        data['cluster'] = labels
        
        analysis_results = []
        
        for cluster_id in sorted(data['cluster'].unique()):
            cluster_data = data[data['cluster'] == cluster_id]
            
            # Basic statistics
            stats = {
                'Cluster': cluster_id if cluster_id != -1 else 'Noise',
                'Size': len(cluster_data),
                'Age_Mean': cluster_data['age'].mean(),
                'Education_Num_Mean': cluster_data['education-num'].mean(),
                'Hours_Mean': cluster_data['hours-per-week'].mean(),
            }
            
            # Top 3 occupations
            top_occupations = cluster_data['occupation'].value_counts().head(3)
            for i, (occ, count) in enumerate(top_occupations.items()):
                stats[f'Top{i+1}_Occupation'] = f"{occ} ({count/len(cluster_data)*100:.1f}%)"
            
            analysis_results.append(stats)
        
        analysis_df = pd.DataFrame(analysis_results)
        
        self.logger.info(f"\n{method} Cluster Semantic Analysis:")
        self.logger.info(f"\n{analysis_df.to_string()}")
        
        return analysis_df
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete clustering analysis.
        
        Returns:
            Dictionary with all results
        """
        self.logger.info("Starting Clustering Analysis (Task 3)")
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # K-Means analysis
        k_values, inertias = self.run_kmeans_elbow()
        k_values_sil, silhouettes = self.run_kmeans_silhouette()
        
        # Select best k from silhouette and run final K-Means
        best_k = k_values_sil[np.argmax(silhouettes)]
        self.run_kmeans(n_clusters=best_k)
        
        # DBSCAN analysis
        self.run_dbscan()
        
        # Visualize clusters
        self.plot_clusters()
        
        # Semantic analysis
        kmeans_analysis = self.analyze_clusters_semantically('KMeans')
        save_results_csv(kmeans_analysis, 'kmeans_cluster_analysis', self.reports_dir)
        
        if 'DBSCAN' in self.cluster_results:
            dbscan_analysis = self.analyze_clusters_semantically('DBSCAN')
            save_results_csv(dbscan_analysis, 'dbscan_cluster_analysis', self.reports_dir)
        else:
            dbscan_analysis = pd.DataFrame()
        
        self.logger.info("Clustering Analysis completed!")
        
        return {
            'elbow': {'k_values': k_values, 'inertias': inertias},
            'silhouette': {'k_values': k_values_sil, 'scores': silhouettes},
            'kmeans_analysis': kmeans_analysis,
            'dbscan_analysis': dbscan_analysis
        }


def run_clustering_task(
    data_dir: str = 'data',
    results_dir: str = 'results'
) -> Dict[str, Any]:
    """
    Convenience function to run the clustering task.
    
    Args:
        data_dir: Directory containing the data files
        results_dir: Directory to save results
        
    Returns:
        Dictionary with results
    """
    logger = setup_logging()
    analyzer = ClusteringAnalyzer(data_dir, results_dir, logger)
    return analyzer.run()


if __name__ == '__main__':
    results = run_clustering_task()
    print(results['kmeans_analysis'])
