import pandas as pd
import numpy as np
import random
import string
from typing import Dict, List, Optional

class DirtyDataGenerator:
    """Generates controlled 'dirty' data for practicing data cleaning techniques."""
    
    def __init__(self):
        self.categorical_errors = {
            'typos': ['tpye', 'typ', 'tyep'],
            'case_variations': ['type', 'Type', 'TYPE', 'tYpE'],
            'extra_spaces': [' type', 'type ', ' type '],
            'special_chars': ['type!', 'type?', 'type#'],
            'misspellings': ['tipe', 'tupe', 'typo']
        }
    
    def introduce_missing_values(self, df: pd.DataFrame, missing_percentage: float = 0.1, 
                                columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Introduce missing values to specified columns or random columns."""
        df_dirty = df.copy()
        
        if columns is None:
            # Select random columns (exclude potential ID columns)
            exclude_patterns = ['id', 'udi', 'index']
            eligible_cols = [col for col in df.columns 
                           if not any(pattern in col.lower() for pattern in exclude_patterns)]
            if eligible_cols:
                columns = np.random.choice(eligible_cols, 
                                         size=min(3, len(eligible_cols)), 
                                         replace=False).tolist()
            else:
                columns = df.columns[:3].tolist()
        
        for col in columns:
            if col in df.columns:
                n_missing = int(len(df) * missing_percentage)
                missing_indices = np.random.choice(df.index, size=n_missing, replace=False)
                df_dirty.loc[missing_indices, col] = np.nan
        
        return df_dirty
    
    def introduce_outliers(self, df: pd.DataFrame, outlier_percentage: float = 0.05,
                          columns: Optional[List[str]] = None, method: str = 'extreme') -> pd.DataFrame:
        """Introduce outliers to numeric columns."""
        df_dirty = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in df.columns and df[col].dtype in [np.number, 'int64', 'float64']:
                n_outliers = int(len(df) * outlier_percentage)
                outlier_indices = np.random.choice(df.index, size=n_outliers, replace=False)
                
                col_mean = df[col].mean()
                col_std = df[col].std()
                
                if method == 'extreme':
                    # Generate extreme values (beyond 3 standard deviations)
                    outlier_values = np.random.choice(
                        [col_mean + 4*col_std, col_mean - 4*col_std, 
                         col_mean + 5*col_std, col_mean - 5*col_std],
                        size=n_outliers
                    )
                elif method == 'random':
                    # Generate random extreme values
                    outlier_values = np.random.uniform(
                        col_mean - 6*col_std, col_mean + 6*col_std, n_outliers
                    )
                
                df_dirty.loc[outlier_indices, col] = outlier_values
        
        return df_dirty
    
    def introduce_categorical_errors(self, df: pd.DataFrame, error_percentage: float = 0.1,
                                   columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Introduce various errors in categorical columns."""
        df_dirty = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in columns:
            if col in df.columns and df[col].dtype == 'object':
                n_errors = int(len(df) * error_percentage)
                error_indices = np.random.choice(df.index, size=n_errors, replace=False)
                
                unique_values = df[col].dropna().unique()
                
                for idx in error_indices:
                    if pd.notna(df_dirty.loc[idx, col]):
                        original_value = str(df_dirty.loc[idx, col])
                        error_type = np.random.choice(['typo', 'case', 'space', 'special', 'invalid'])
                        
                        if error_type == 'typo':
                            # Introduce typos
                            if len(original_value) > 2:
                                pos = np.random.randint(0, len(original_value))
                                new_char = np.random.choice(list(string.ascii_lowercase))
                                corrupted = original_value[:pos] + new_char + original_value[pos+1:]
                                df_dirty.loc[idx, col] = corrupted
                        
                        elif error_type == 'case':
                            # Random case variations
                            df_dirty.loc[idx, col] = ''.join(
                                char.upper() if np.random.random() > 0.5 else char.lower() 
                                for char in original_value
                            )
                        
                        elif error_type == 'space':
                            # Add random spaces
                            spaces = np.random.choice([' ', '  ', '   '])
                            position = np.random.choice(['before', 'after', 'both'])
                            if position == 'before':
                                df_dirty.loc[idx, col] = spaces + original_value
                            elif position == 'after':
                                df_dirty.loc[idx, col] = original_value + spaces
                            else:
                                df_dirty.loc[idx, col] = spaces + original_value + spaces
                        
                        elif error_type == 'special':
                            # Add special characters
                            special_char = np.random.choice(['!', '?', '#', '@', '$'])
                            df_dirty.loc[idx, col] = original_value + special_char
                        
                        elif error_type == 'invalid':
                            # Completely invalid values
                            invalid_values = ['NULL', 'N/A', 'UNKNOWN', 'ERROR', '???', '---']
                            df_dirty.loc[idx, col] = np.random.choice(invalid_values)
        
        return df_dirty
    
    def generate_dirty_data(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Generate dirty data based on configuration."""
        df_dirty = df.copy()
        
        # Apply missing values
        if config.get('missing_values', {}).get('enabled', False):
            missing_config = config['missing_values']
            df_dirty = self.introduce_missing_values(
                df_dirty,
                missing_percentage=missing_config.get('percentage', 0.1),
                columns=missing_config.get('columns', None)
            )
        
        # Apply outliers
        if config.get('outliers', {}).get('enabled', False):
            outlier_config = config['outliers']
            df_dirty = self.introduce_outliers(
                df_dirty,
                outlier_percentage=outlier_config.get('percentage', 0.05),
                columns=outlier_config.get('columns', None),
                method=outlier_config.get('method', 'extreme')
            )
        
        # Apply categorical errors
        if config.get('categorical_errors', {}).get('enabled', False):
            cat_config = config['categorical_errors']
            df_dirty = self.introduce_categorical_errors(
                df_dirty,
                error_percentage=cat_config.get('percentage', 0.1),
                columns=cat_config.get('columns', None)
            )
        
        return df_dirty
    
    def get_corruption_summary(self, original: pd.DataFrame, dirty: pd.DataFrame) -> Dict:
        """Generate summary of applied corruptions."""
        summary = {
            'missing_values': {
                'original_count': original.isnull().sum().sum(),
                'dirty_count': dirty.isnull().sum().sum(),
                'added': dirty.isnull().sum().sum() - original.isnull().sum().sum()
            },
            'data_changes': {},
            'shape_preserved': original.shape == dirty.shape
        }
        
        # Analyze changes per column
        for col in original.columns:
            if col in dirty.columns:
                original_nulls = original[col].isnull().sum()
                dirty_nulls = dirty[col].isnull().sum()
                
                # Compare non-null values
                if original[col].dtype == 'object':
                    original_values = set(original[col].dropna().astype(str))
                    dirty_values = set(dirty[col].dropna().astype(str))
                    new_values = dirty_values - original_values
                else:
                    # For numeric, check if values changed significantly
                    mask = original[col].notna() & dirty[col].notna()
                    changed_values = (~np.isclose(original[col][mask], dirty[col][mask], equal_nan=True)).sum()
                    new_values = f"{changed_values} modified values"
                
                summary['data_changes'][col] = {
                    'missing_added': dirty_nulls - original_nulls,
                    'new_values': new_values if isinstance(new_values, str) else len(new_values),
                    'data_type': str(original[col].dtype)
                }
        
        return summary
