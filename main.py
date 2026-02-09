#!/usr/bin/env python3
"""
Main orchestration script for the Data Mining project.
Runs all 5 tasks on the UCI Adult dataset.

Usage:
    python main.py              # Run all tasks
    python main.py --task 1     # Run specific task (1-5)
    python main.py --task 1,3   # Run specific tasks
"""

import argparse
import logging
import os
import sys
from typing import List

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_logging
from src.classification import ClassificationPipeline
from src.outlier_detection import OutlierAnalyzer
from src.clustering import ClusteringAnalyzer
from src.association_rules import AssociationRulesMiner
from src.feature_selection import FeatureSelectionStudy
from src.naive_bayes import NaiveBayesBaseline
from src.knn_task import KNNStudy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Project directories
DATA_DIR = 'data'
RESULTS_DIR = 'results'


def check_data_files(logger: logging.Logger) -> bool:
    """
    Check if the required data files exist.
    
    Args:
        logger: Logger instance
        
    Returns:
        True if all files exist, False otherwise
    """
    train_path = os.path.join(DATA_DIR, 'adult.data')
    test_path = os.path.join(DATA_DIR, 'adult.test')
    
    if not os.path.exists(train_path):
        logger.error(f"Training data not found: {train_path}")
        logger.error("Please download adult.data from https://archive.ics.uci.edu/dataset/2/adult")
        return False
    
    if not os.path.exists(test_path):
        logger.error(f"Test data not found: {test_path}")
        logger.error("Please download adult.test from https://archive.ics.uci.edu/dataset/2/adult")
        return False
    
    logger.info("Data files found successfully")
    return True


def run_task_1(logger: logging.Logger) -> None:
    """
    Task 1: Classification Multi-Algoritm + Feature Engineering Comparison.
    """
    logger.info("\n" + "="*80)
    logger.info("TASK 1: Classification Multi-Algoritm + Feature Engineering")
    logger.info("="*80)
    
    pipeline = ClassificationPipeline(DATA_DIR, RESULTS_DIR, logger)
    results = pipeline.run()
    
    logger.info("\nTask 1 Results Summary:")
    logger.info(f"\n{results.to_string()}")


def run_task_2(logger: logging.Logger) -> None:
    """
    Task 2: Outlier Analysis.
    """
    logger.info("\n" + "="*80)
    logger.info("TASK 2: Outlier Analysis")
    logger.info("="*80)
    
    analyzer = OutlierAnalyzer(DATA_DIR, RESULTS_DIR, logger)
    results = analyzer.run()
    
    logger.info("\nTask 2 completed - check results/plots/ for visualizations")


def run_task_3(logger: logging.Logger) -> None:
    """
    Task 3: Clustering.
    """
    logger.info("\n" + "="*80)
    logger.info("TASK 3: Clustering (K-Means & DBSCAN)")
    logger.info("="*80)
    
    analyzer = ClusteringAnalyzer(DATA_DIR, RESULTS_DIR, logger)
    results = analyzer.run()
    
    logger.info("\nTask 3 completed - check results/reports/ for cluster analysis")


def run_task_4(logger: logging.Logger) -> None:
    """
    Task 4: Association Rules.
    """
    logger.info("\n" + "="*80)
    logger.info("TASK 4: Association Rules Mining")
    logger.info("="*80)
    
    miner = AssociationRulesMiner(DATA_DIR, RESULTS_DIR, logger=logger)
    results = miner.run()
    
    logger.info("\nTask 4 completed - check results/reports/ for rules")


def run_task_5(logger: logging.Logger) -> None:
    """
    Task 5: Feature Selection Study.
    """
    logger.info("\n" + "="*80)
    logger.info("TASK 5: Feature Selection Study")
    logger.info("="*80)
    
    study = FeatureSelectionStudy(DATA_DIR, RESULTS_DIR, logger)
    results = study.run()
    
    logger.info("\nTask 5 Results Summary:")
    logger.info(f"\n{results['results'].to_string()}")
    logger.info(f"\nCommon features at k=10: {results['common_features_k10']}")

def run_task_6(logger: logging.Logger) -> None:
    """
    Task 6: Naive Bayes Baseline (GaussianNB).
    """
    logger.info("\n" + "="*80)
    logger.info("TASK 6: Naive Bayes Baseline (GaussianNB)")
    logger.info("="*80)

    nb = NaiveBayesBaseline(DATA_DIR, RESULTS_DIR, logger)
    results = nb.run()

    logger.info("\nTask 6 Results Summary:")
    logger.info(f"\n{results.to_string(index=False)}")

def run_task_7(logger: logging.Logger) -> None:
    """
    Task 7: k-Nearest Neighbors (KNN) Classifier.
    """
    logger.info("\n" + "="*80)
    logger.info("TASK 7: KNN Classifier (k-Nearest Neighbors)")
    logger.info("="*80)

    knn = KNNStudy(DATA_DIR, RESULTS_DIR, logger)
    results = knn.run()

    logger.info("\nTask 7 Results Summary:")
    logger.info(f"\n{results.to_string(index=False)}")


def parse_tasks(task_str: str) -> List[int]:
    """
    Parse task string into list of task numbers.
    
    Args:
        task_str: Comma-separated task numbers (e.g., "1,3,5")
        
    Returns:
        List of task numbers
    """
    tasks = []
    for t in task_str.split(','):
        t = t.strip()
        if t.isdigit():
            task_num = int(t)
            if 1 <= task_num <= 7:
                tasks.append(task_num)
    return sorted(set(tasks))


def main():
    """
    Main entry point for the Data Mining project.
    """
    parser = argparse.ArgumentParser(
        description='Data Mining Project - UCI Adult Dataset Analysis'
    )
    parser.add_argument(
        '--task', '-t',
        type=str,
        default='all',
        help='Task(s) to run: "all" or comma-separated numbers (1-6), e.g., "1,3"'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("="*80)
    logger.info("DATA MINING PROJECT - UCI Adult Dataset")
    logger.info("="*80)
    
    # Check data files
    if not check_data_files(logger):
        logger.error("Exiting due to missing data files")
        sys.exit(1)
    
    # Create output directories
    os.makedirs(os.path.join(RESULTS_DIR, 'models'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'reports'), exist_ok=True)
    
    # Define task functions
    task_functions = {
        1: run_task_1,
        2: run_task_2,
        3: run_task_3,
        4: run_task_4,
        5: run_task_5,
        6: run_task_6,
        7: run_task_7
    }
    
    # Determine which tasks to run
    if args.task.lower() == 'all':
        tasks_to_run = [1, 2, 3, 4, 5, 6, 7]
    else:
        tasks_to_run = parse_tasks(args.task)
        if not tasks_to_run:
            logger.error(f"Invalid task specification: {args.task}")
            sys.exit(1)
    
    logger.info(f"Tasks to run: {tasks_to_run}")
    
    # Run tasks
    for task_num in tasks_to_run:
        try:
            task_functions[task_num](logger)
        except Exception as e:
            logger.error(f"Error in Task {task_num}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    logger.info("\n" + "="*80)
    logger.info("ALL TASKS COMPLETED")
    logger.info("="*80)
    logger.info("\nResults saved to:")
    logger.info(f"  - Models: {RESULTS_DIR}/models/")
    logger.info(f"  - Plots: {RESULTS_DIR}/plots/")
    logger.info(f"  - Reports: {RESULTS_DIR}/reports/")


if __name__ == '__main__':
    main()
