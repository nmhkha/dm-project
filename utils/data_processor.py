import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Optional

class DataProcessor:
    """Handles data loading, validation, and basic processing operations."""
    
    def __init__(self):
        self.expected_columns = [
            'UDI', 'Product ID', 'Type', 'Air temperature [K]', 
            'Process temperature [K]', 'Rotational speed [rpm]', 
            'Torque [Nm]', 'Tool wear [min]', 'Target'
        ]
        
    def load_csv(self, file) -> pd.DataFrame:
        """Load CSV file and perform basic validation."""
        try:
            df = pd.read_csv(file)
            return df
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")
            return None
    
    def validate_data_structure(self, df: pd.DataFrame) -> Dict:
        """Validate if the dataset has expected structure for predictive maintenance."""
        validation_results = {
            'is_valid': True,
            'messages': [],
            'warnings': [],
            'column_mapping': {}
        }
        
        # Check if we have minimum required columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if len(numeric_cols) < 3:
            validation_results['warnings'].append("Dataset has fewer than 3 numeric columns")
        
        if len(categorical_cols) < 1:
            validation_results['warnings'].append("Dataset has no categorical columns")
            
        # Check for potential target column
        potential_targets = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['target', 'failure', 'label', 'class']):
                potential_targets.append(col)
        
        if potential_targets:
            validation_results['messages'].append(f"Potential target columns found: {potential_targets}")
        
        # Check for ID columns
        potential_ids = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['id', 'udi', 'identifier']):
                potential_ids.append(col)
                
        if potential_ids:
            validation_results['messages'].append(f"Potential ID columns found: {potential_ids}")
        
        return validation_results
    
    def get_data_overview(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive data overview."""
        overview = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        }
        
        # Add descriptive statistics for numeric columns
        if overview['numeric_columns']:
            overview['numeric_stats'] = df[overview['numeric_columns']].describe().to_dict()
        
        # Add value counts for categorical columns (top 10)
        overview['categorical_stats'] = {}
        for col in overview['categorical_columns']:
            value_counts = df[col].value_counts().head(10)
            overview['categorical_stats'][col] = value_counts.to_dict()
        
        return overview
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> Dict:
        """Detect outliers in numeric columns."""
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_mask = z_scores > 3
            
            outliers[col] = {
                'count': outlier_mask.sum(),
                'percentage': (outlier_mask.sum() / len(df)) * 100,
                'indices': df[outlier_mask].index.tolist()
            }
        
        return outliers
    
    def get_data_quality_score(self, df: pd.DataFrame) -> Dict:
        """Calculate overall data quality score."""
        scores = {}
        
        # Missing values score (0-100)
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        scores['missing_values'] = max(0, 100 - (missing_ratio * 100))
        
        # Duplicates score
        duplicate_ratio = df.duplicated().sum() / len(df)
        scores['duplicates'] = max(0, 100 - (duplicate_ratio * 100))
        
        # Outliers score (using IQR method)
        outliers = self.detect_outliers(df)
        total_outliers = sum([info['count'] for info in outliers.values()])
        outlier_ratio = total_outliers / len(df) if len(df) > 0 else 0
        scores['outliers'] = max(0, 100 - (outlier_ratio * 10))  # Less penalty for outliers
        
        # Overall score
        scores['overall'] = np.mean(list(scores.values()))
        
        return scores
    
    def compare_datasets(self, original: pd.DataFrame, modified: pd.DataFrame) -> Dict:
        """Compare two datasets and return differences."""
        comparison = {
            'shape_change': {
                'original': original.shape,
                'modified': modified.shape,
                'rows_diff': modified.shape[0] - original.shape[0],
                'cols_diff': modified.shape[1] - original.shape[1]
            },
            'missing_values_change': {},
            'quality_scores': {
                'original': self.get_data_quality_score(original),
                'modified': self.get_data_quality_score(modified)
            }
        }
        
        # Compare missing values
        original_missing = original.isnull().sum()
        modified_missing = modified.isnull().sum()
        
        for col in original.columns:
            if col in modified.columns:
                comparison['missing_values_change'][col] = {
                    'original': original_missing[col],
                    'modified': modified_missing[col],
                    'change': modified_missing[col] - original_missing[col]
                }
        
        return comparison
