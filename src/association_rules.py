"""
Association rules mining module for Task 4.
Implements Apriori and FP-Growth with discretization and visualization.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

from .preprocessing import DataPreprocessor
from .utils import PlotUtils, save_results_csv, setup_logging


class AssociationRulesMiner:
    """
    Association rules mining using Apriori and FP-Growth.
    Includes discretization, rule extraction, and network visualization.
    """
    
    def __init__(
        self,
        data_dir: str = 'data',
        results_dir: str = 'results',
        min_support: float = 0.01,
        min_confidence: float = 0.5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the association rules miner.
        
        Args:
            data_dir: Directory containing the data files
            results_dir: Directory to save results
            min_support: Minimum support threshold
            min_confidence: Minimum confidence threshold
            logger: Optional logger instance
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.plots_dir = os.path.join(results_dir, 'plots')
        self.reports_dir = os.path.join(results_dir, 'reports')
        
        self.min_support = min_support
        self.min_confidence = min_confidence
        
        self.logger = logger or logging.getLogger('DataMining')
        self.preprocessor = DataPreprocessor(data_dir, logger)
        self.plot_utils = PlotUtils(self.plots_dir, logger)
        
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        self.transactions = None
        self.frequent_itemsets = {}
        self.rules = {}
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Load data and discretize for association rules.
        
        Returns:
            DataFrame with discretized data
        """
        self.logger.info("Loading and discretizing data for association rules...")
        
        combined_df = self.preprocessor.load_combined_data()
        combined_df = self.preprocessor.handle_missing_values(combined_df)
        
        # Discretize the data
        discretized_df = self.preprocessor.discretize_for_association_rules(combined_df)
        
        self.logger.info(f"Discretized {len(discretized_df)} instances")
        
        return discretized_df
    
    def create_transactions(self, df: pd.DataFrame) -> List[List[str]]:
        """
        Convert DataFrame to list of transactions.
        
        Args:
            df: Discretized DataFrame
            
        Returns:
            List of transactions (each transaction is a list of items)
        """
        # Select columns for transactions
        cols_to_use = [
            'age', 'hours-per-week', 'capital-gain', 'education-num',
            'workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'native-country', 'income'
        ]
        
        # Filter to available columns
        cols_to_use = [c for c in cols_to_use if c in df.columns]
        
        transactions = df[cols_to_use].values.tolist()
        self.transactions = transactions
        
        self.logger.info(f"Created {len(transactions)} transactions with {len(cols_to_use)} items each")
        
        return transactions
    
    def encode_transactions(self) -> pd.DataFrame:
        """
        Encode transactions into one-hot format for mlxtend.
        
        Returns:
            One-hot encoded DataFrame
        """
        te = TransactionEncoder()
        te_array = te.fit_transform(self.transactions)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)
        
        self.logger.info(f"Encoded transactions shape: {df_encoded.shape}")
        
        return df_encoded
    
    def mine_with_apriori(self, df_encoded: pd.DataFrame) -> pd.DataFrame:
        """
        Mine frequent itemsets using Apriori algorithm.
        
        Args:
            df_encoded: One-hot encoded DataFrame
            
        Returns:
            DataFrame with frequent itemsets
        """
        self.logger.info(f"Mining with Apriori (min_support={self.min_support})...")
        
        frequent_itemsets = apriori(
            df_encoded,
            min_support=self.min_support,
            use_colnames=True
        )
        
        self.frequent_itemsets['Apriori'] = frequent_itemsets
        self.logger.info(f"Apriori found {len(frequent_itemsets)} frequent itemsets")
        
        return frequent_itemsets
    
    def mine_with_fpgrowth(self, df_encoded: pd.DataFrame) -> pd.DataFrame:
        """
        Mine frequent itemsets using FP-Growth algorithm.
        
        Args:
            df_encoded: One-hot encoded DataFrame
            
        Returns:
            DataFrame with frequent itemsets
        """
        self.logger.info(f"Mining with FP-Growth (min_support={self.min_support})...")
        
        frequent_itemsets = fpgrowth(
            df_encoded,
            min_support=self.min_support,
            use_colnames=True
        )
        
        self.frequent_itemsets['FP-Growth'] = frequent_itemsets
        self.logger.info(f"FP-Growth found {len(frequent_itemsets)} frequent itemsets")
        
        return frequent_itemsets
    
    def generate_rules(self, method: str = 'Apriori') -> pd.DataFrame:
        """
        Generate association rules from frequent itemsets.
        
        Args:
            method: Mining method to use ('Apriori' or 'FP-Growth')
            
        Returns:
            DataFrame with association rules
        """
        if method not in self.frequent_itemsets:
            self.logger.warning(f"No frequent itemsets for {method}")
            return pd.DataFrame()
        
        self.logger.info(f"Generating rules with min_confidence={self.min_confidence}...")
        
        frequent_itemsets = self.frequent_itemsets[method]
        
        if len(frequent_itemsets) == 0:
            self.logger.warning("No frequent itemsets found!")
            return pd.DataFrame()
        
        rules = association_rules(
            frequent_itemsets,
            metric='confidence',
            min_threshold=self.min_confidence
        )
        
        self.rules[method] = rules
        self.logger.info(f"Generated {len(rules)} rules with {method}")
        
        return rules
    
    def get_top_rules_for_income(
        self,
        income_class: str = '>50K',
        top_n: int = 10,
        method: str = 'Apriori'
    ) -> pd.DataFrame:
        """
        Get top rules with income as consequent, sorted by lift.
        
        Args:
            income_class: Target income class ('>50K' or '<=50K')
            top_n: Number of top rules to return
            method: Mining method to use
            
        Returns:
            DataFrame with top rules
        """
        if method not in self.rules:
            self.logger.warning(f"No rules for {method}")
            return pd.DataFrame()
        
        rules = self.rules[method].copy()
        
        # Filter rules where consequent contains income
        income_item = f"income={income_class}"
        
        # Convert frozensets to strings for filtering
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        
        # Filter for income in consequent
        income_rules = rules[rules['consequents_str'].str.contains(income_item)]
        
        # Sort by lift
        top_rules = income_rules.nlargest(top_n, 'lift')
        
        # Select relevant columns
        result = top_rules[[
            'antecedents_str', 'consequents_str', 
            'support', 'confidence', 'lift'
        ]].rename(columns={
            'antecedents_str': 'antecedents',
            'consequents_str': 'consequents'
        })
        
        self.logger.info(f"\nTop {top_n} rules for {income_item}:")
        self.logger.info(f"\n{result.to_string()}")
        
        return result
    
    def plot_rules_network(
        self,
        rules_df: pd.DataFrame,
        top_n: int = 5,
        title: str = "Association Rules Network"
    ) -> None:
        """
        Create a network graph visualization of association rules.
        
        Args:
            rules_df: DataFrame with rules
            top_n: Number of top rules to visualize
            title: Plot title
        """
        if len(rules_df) == 0:
            self.logger.warning("No rules to plot!")
            return
        
        # Take top n rules
        rules_to_plot = rules_df.head(top_n)
        
        # Create graph
        G = nx.DiGraph()
        
        for idx, row in rules_to_plot.iterrows():
            antecedent = row['antecedents']
            consequent = row['consequents']
            lift = row['lift']
            
            # Add nodes
            G.add_node(antecedent, node_type='antecedent')
            G.add_node(consequent, node_type='consequent')
            
            # Add edge
            G.add_edge(antecedent, consequent, weight=lift)
        
        # Create layout
        fig, ax = plt.subplots(figsize=(14, 10))
        
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw nodes
        antecedent_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'antecedent']
        consequent_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'consequent']
        
        nx.draw_networkx_nodes(G, pos, nodelist=antecedent_nodes, 
                               node_color='lightblue', node_size=2000, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=consequent_nodes,
                               node_color='lightcoral', node_size=2500, ax=ax)
        
        # Draw edges
        edges = G.edges(data=True)
        weights = [e[2]['weight'] for e in edges]
        max_weight = max(weights) if weights else 1
        normalized_weights = [2 + 3 * w / max_weight for w in weights]
        
        nx.draw_networkx_edges(G, pos, width=normalized_weights, 
                               alpha=0.7, edge_color='gray',
                               arrows=True, arrowsize=20, ax=ax)
        
        # Draw labels (wrap long labels)
        labels = {}
        for node in G.nodes():
            if len(node) > 30:
                labels[node] = node[:27] + '...'
            else:
                labels[node] = node
        
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
        
        # Add edge labels (lift values)
        edge_labels = {(u, v): f"lift={d['weight']:.2f}" for u, v, d in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                      font_size=7, ax=ax)
        
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        filename = title.lower().replace(' ', '_').replace('>', 'gt').replace('<', 'lt')
        self.plot_utils.save_figure(fig, f'network_{filename}')
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete association rules mining analysis.
        
        Returns:
            Dictionary with all results
        """
        self.logger.info("Starting Association Rules Mining (Task 4)")
        
        # Load and prepare data
        discretized_df = self.load_and_prepare_data()
        
        # Create transactions
        self.create_transactions(discretized_df)
        
        # Encode transactions
        df_encoded = self.encode_transactions()
        
        # Mine with both methods
        self.mine_with_apriori(df_encoded)
        self.mine_with_fpgrowth(df_encoded)
        
        # Generate rules
        rules_apriori = self.generate_rules('Apriori')
        rules_fpgrowth = self.generate_rules('FP-Growth')
        
        # Get top rules for each income class
        results = {}
        
        for method in ['Apriori', 'FP-Growth']:
            if method in self.rules and len(self.rules[method]) > 0:
                # Top rules for >50K
                top_high = self.get_top_rules_for_income('>50K', top_n=10, method=method)
                save_results_csv(top_high, f'{method.lower()}_rules_high_income', self.reports_dir)
                results[f'{method}_high_income'] = top_high
                
                # Top rules for <=50K
                top_low = self.get_top_rules_for_income('<=50K', top_n=10, method=method)
                save_results_csv(top_low, f'{method.lower()}_rules_low_income', self.reports_dir)
                results[f'{method}_low_income'] = top_low
                
                # Network visualization for top 5 high income rules
                if len(top_high) > 0:
                    self.plot_rules_network(
                        top_high, top_n=5,
                        title=f'{method} Rules for Income >50K'
                    )
        
        self.logger.info("Association Rules Mining completed!")
        
        return results


def run_association_rules_task(
    data_dir: str = 'data',
    results_dir: str = 'results'
) -> Dict[str, Any]:
    """
    Convenience function to run the association rules task.
    
    Args:
        data_dir: Directory containing the data files
        results_dir: Directory to save results
        
    Returns:
        Dictionary with results
    """
    logger = setup_logging()
    miner = AssociationRulesMiner(data_dir, results_dir, logger=logger)
    return miner.run()


if __name__ == '__main__':
    results = run_association_rules_task()
    for key, df in results.items():
        print(f"\n{key}:")
        print(df.head())
