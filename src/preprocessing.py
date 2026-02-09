"""
Data preprocessing module for the Adult dataset.
Handles loading, cleaning, encoding, and feature engineering.
"""

import logging
import os
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class DataPreprocessor:
    """
    Preprocessor class for the UCI Adult dataset.
    Handles data loading, missing values, encoding, and feature engineering.
    """
    
    # Column names for the Adult dataset
    COLUMN_NAMES = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    # Categorical columns
    CATEGORICAL_COLS = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country'
    ]
    
    # Numeric columns
    NUMERIC_COLS = [
        'age', 'fnlwgt', 'education-num', 'capital-gain',
        'capital-loss', 'hours-per-week'
    ]
    
    def __init__(
        self, 
        data_dir: str = 'data',
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the preprocessor.
        
        Args:
            data_dir: Directory containing the data files
            logger: Optional logger instance
        """
        self.data_dir = data_dir
        self.logger = logger or logging.getLogger('DataMining')
        self.scaler = StandardScaler()
        self.encoder = None
        self.label_encoder = LabelEncoder()
        self._is_fitted = False
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the Adult dataset from data directory.
        
        Returns:
            Tuple of (train_data, test_data) DataFrames
            
        Raises:
            FileNotFoundError: If data files are not found
        """
        train_path = os.path.join(self.data_dir, 'adult.data')
        test_path = os.path.join(self.data_dir, 'adult.test')
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test data not found: {test_path}")
        
        self.logger.info(f"Loading training data from {train_path}")
        train_df = pd.read_csv(
            train_path, 
            names=self.COLUMN_NAMES,
            skipinitialspace=True,
            na_values='?'
        )
        
        self.logger.info(f"Loading test data from {test_path}")
        # Test file has a different format - skip first line and handle trailing dot
        test_df = pd.read_csv(
            test_path,
            names=self.COLUMN_NAMES,
            skipinitialspace=True,
            skiprows=1,  # Skip the first line with "|1x3 Cross validator"
            na_values='?'
        )
        
        # Clean the income column in test set (remove trailing period)
        test_df['income'] = test_df['income'].str.rstrip('.')
        
        self.logger.info(f"Loaded {len(train_df)} training and {len(test_df)} test instances")
        
        return train_df, test_df
    
    def load_combined_data(self) -> pd.DataFrame:
        """
        Load and combine train and test data into a single DataFrame.
        
        Returns:
            Combined DataFrame
        """
        train_df, test_df = self.load_data()
        combined = pd.concat([train_df, test_df], ignore_index=True)
        self.logger.info(f"Combined dataset: {len(combined)} instances")
        return combined
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        strategy: str = 'mode'
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values ('mode', 'drop')
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        missing_count = df.isnull().sum().sum()
        self.logger.info(f"Found {missing_count} missing values")
        
        if strategy == 'drop':
            df = df.dropna()
            self.logger.info(f"Dropped rows with missing values. Remaining: {len(df)}")
        elif strategy == 'mode':
            for col in df.columns:
                if df[col].isnull().any():
                    mode_value = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_value)
                    self.logger.info(f"Filled {col} missing values with mode: {mode_value}")
        
        return df
    
    def encode_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Encode the target variable (income).
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features DataFrame, encoded target array)
        """
        df = df.copy()
        
        # Binary encoding: >50K = 1, <=50K = 0
        y = (df['income'].str.strip() == '>50K').astype(int).values
        X = df.drop('income', axis=1)
        
        self.logger.info(f"Target distribution: {np.bincount(y)}")
        
        return X, y
    
    def create_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features.
        
        Features created:
        - gain_loss_ratio: capital-gain / (capital-loss + 1)
        - is_high_hours: 1 if hours-per-week > 40, else 0
        - education_efficiency: education-num / age
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new features
        """
        df = df.copy()
        
        # Gain/Loss ratio
        df['gain_loss_ratio'] = df['capital-gain'] / (df['capital-loss'] + 1)
        
        # High hours indicator (>40 hours per week)
        df['is_high_hours'] = (df['hours-per-week'] > 40).astype(int)
        
        # Education efficiency (education level relative to age)
        df['education_efficiency'] = df['education-num'] / df['age']
        
        self.logger.info("Created 3 engineered features: gain_loss_ratio, is_high_hours, education_efficiency")
        
        return df
    
    def get_preprocessor_pipeline(
        self, 
        with_feature_engineering: bool = False
    ) -> ColumnTransformer:
        """
        Create a preprocessing pipeline for the data.
        
        Args:
            with_feature_engineering: Whether to include engineered features
            
        Returns:
            ColumnTransformer for preprocessing
        """
        numeric_features = self.NUMERIC_COLS.copy()
        categorical_features = self.CATEGORICAL_COLS.copy()
        
        if with_feature_engineering:
            # Add engineered features to numeric list
            numeric_features.extend(['gain_loss_ratio', 'is_high_hours', 'education_efficiency'])
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return preprocessor
    
    def preprocess_for_classification(
        self, 
        with_feature_engineering: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Full preprocessing pipeline for classification tasks.
        
        Args:
            with_feature_engineering: Whether to include engineered features
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names)
        """
        train_df, test_df = self.load_data()
        
        # Handle missing values
        train_df = self.handle_missing_values(train_df)
        test_df = self.handle_missing_values(test_df)
        
        # Feature engineering if requested
        if with_feature_engineering:
            train_df = self.create_feature_engineering(train_df)
            test_df = self.create_feature_engineering(test_df)
        
        # Separate features and target
        X_train, y_train = self.encode_target(train_df)
        X_test, y_test = self.encode_target(test_df)
        
        # Create and fit preprocessor
        preprocessor = self.get_preprocessor_pipeline(with_feature_engineering)
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Get feature names
        feature_names = self._get_feature_names(preprocessor, with_feature_engineering)
        
        self.logger.info(f"Preprocessed data shape: train={X_train_processed.shape}, test={X_test_processed.shape}")
        
        return X_train_processed, X_test_processed, y_train, y_test, feature_names
    
    def _get_feature_names(
        self, 
        preprocessor: ColumnTransformer,
        with_feature_engineering: bool
    ) -> List[str]:
        """
        Extract feature names from the preprocessor.
        
        Args:
            preprocessor: Fitted ColumnTransformer
            with_feature_engineering: Whether engineered features are included
            
        Returns:
            List of feature names
        """
        try:
            return list(preprocessor.get_feature_names_out())
        except Exception:
            # Fallback for older sklearn versions
            numeric_features = self.NUMERIC_COLS.copy()
            if with_feature_engineering:
                numeric_features.extend(['gain_loss_ratio', 'is_high_hours', 'education_efficiency'])
            
            # Get categorical feature names from fitted encoder
            cat_transformer = preprocessor.named_transformers_['cat']
            cat_features = list(cat_transformer.named_steps['onehot'].get_feature_names_out(self.CATEGORICAL_COLS))
            
            return numeric_features + cat_features
    
    def preprocess_for_clustering(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Preprocess data for clustering (without target).
        
        Returns:
            Tuple of (processed features array, original DataFrame)
        """
        combined_df = self.load_combined_data()
        combined_df = self.handle_missing_values(combined_df)
        
        # Drop target for clustering
        X = combined_df.drop('income', axis=1)
        
        # Create preprocessor and transform
        preprocessor = self.get_preprocessor_pipeline(with_feature_engineering=False)
        X_processed = preprocessor.fit_transform(X)
        
        self.logger.info(f"Preprocessed clustering data shape: {X_processed.shape}")
        
        return X_processed, combined_df
    
    def get_high_income_data(self) -> pd.DataFrame:
        """
        Get data filtered for income >50K (for outlier detection).
        
        Returns:
            DataFrame with high income instances
        """
        combined_df = self.load_combined_data()
        combined_df = self.handle_missing_values(combined_df)
        
        # Filter for income >50K
        high_income = combined_df[combined_df['income'].str.strip() == '>50K'].copy()
        self.logger.info(f"High income instances: {len(high_income)}")
        
        return high_income
    
    def discretize_for_association_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Discretize numeric columns for association rule mining.
        
        Discretization bins:
        - age: [0-25, 26-35, 36-45, 46-55, 56-100]
        - hours-per-week: [0-30, 31-40, 41-50, 51-100]
        - capital-gain: [0, 1-5000, 5001+]
        - education-num: keep original values
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with discretized columns in string format
        """
        df = df.copy()
        
        # Age bins
        df['age'] = pd.cut(
            df['age'], 
            bins=[0, 25, 35, 45, 55, 100],
            labels=['0-25', '26-35', '36-45', '46-55', '56-100']
        ).astype(str)
        df['age'] = 'age=' + df['age']
        
        # Hours per week bins
        df['hours-per-week'] = pd.cut(
            df['hours-per-week'],
            bins=[0, 30, 40, 50, 100],
            labels=['0-30', '31-40', '41-50', '51-100']
        ).astype(str)
        df['hours-per-week'] = 'hours=' + df['hours-per-week']
        
        # Capital gain bins
        df['capital-gain'] = pd.cut(
            df['capital-gain'],
            bins=[-1, 0, 5000, float('inf')],
            labels=['0', '1-5000', '5001+']
        ).astype(str)
        df['capital-gain'] = 'capital-gain=' + df['capital-gain']
        
        # Education num - keep as string
        df['education-num'] = 'education-num=' + df['education-num'].astype(str)
        
        # Convert all categorical columns to string format
        for col in self.CATEGORICAL_COLS:
            df[col] = col + '=' + df[col].astype(str)
        
        # Income
        df['income'] = 'income=' + df['income'].str.strip()
        
        self.logger.info("Discretized data for association rules")
        
        return df
