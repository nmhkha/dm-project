import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import IsolationForest
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any

class DataCleaner:
    """Provides various data cleaning and preprocessing techniques."""
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.imputers = {}
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean', 
                            columns: Optional[List[str]] = None, custom_value: Any = None) -> pd.DataFrame:
        """Handle missing values using various strategies."""
        df_cleaned = df.copy()
        
        if columns is None:
            columns = df.columns[df.isnull().any()].tolist()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if df[col].isnull().sum() == 0:
                continue
            
            if strategy == 'drop_rows':
                df_cleaned = df_cleaned.dropna(subset=[col])
            
            elif strategy == 'drop_columns':
                df_cleaned = df_cleaned.drop(columns=[col])
            
            elif strategy == 'custom':
                df_cleaned[col] = df_cleaned[col].fillna(custom_value)
            
            elif df[col].dtype in ['object']:
                # Categorical columns
                if strategy == 'mode':
                    mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                    df_cleaned[col] = df_cleaned[col].fillna(mode_value)
                elif strategy == 'constant':
                    df_cleaned[col] = df_cleaned[col].fillna('Missing')
                elif strategy == 'forward_fill':
                    df_cleaned[col] = df_cleaned[col].fillna(method='ffill')
                elif strategy == 'backward_fill':
                    df_cleaned[col] = df_cleaned[col].fillna(method='bfill')
            
            else:
                # Numeric columns
                if strategy == 'mean':
                    df_cleaned[col] = df_cleaned[col].fillna(df[col].mean())
                elif strategy == 'median':
                    df_cleaned[col] = df_cleaned[col].fillna(df[col].median())
                elif strategy == 'mode':
                    mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 0
                    df_cleaned[col] = df_cleaned[col].fillna(mode_value)
                elif strategy == 'interpolate':
                    df_cleaned[col] = df_cleaned[col].interpolate()
                elif strategy == 'knn':
                    # Use KNN imputation for numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 1:
                        imputer = KNNImputer(n_neighbors=5)
                        df_cleaned[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        return df_cleaned
    
    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr', 
                       columns: Optional[List[str]] = None, threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers using various methods."""
        df_cleaned = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_indices = set()
        
        for col in columns:
            if col not in df.columns or df[col].dtype not in [np.number, 'int64', 'float64']:
                continue
            
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers = df[col].dropna().iloc[z_scores > threshold].index
            
            elif method == 'isolation_forest':
                isolation_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = isolation_forest.fit_predict(df[[col]].dropna())
                outliers = df[col].dropna().iloc[outlier_labels == -1].index
            
            outlier_indices.update(outliers)
        
        # Remove outliers
        df_cleaned = df_cleaned.drop(outlier_indices)
        
        return df_cleaned
    
    def standardize_categorical_data(self, df: pd.DataFrame, 
                                   columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Standardize categorical data by cleaning common issues."""
        df_cleaned = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Convert to string to handle mixed types
            df_cleaned[col] = df_cleaned[col].astype(str)
            
            # Remove leading/trailing whitespace
            df_cleaned[col] = df_cleaned[col].str.strip()
            
            # Remove extra spaces
            df_cleaned[col] = df_cleaned[col].str.replace(r'\s+', ' ', regex=True)
            
            # Convert to lowercase for consistency (optional)
            # df_cleaned[col] = df_cleaned[col].str.lower()
            
            # Replace common null representations
            null_representations = ['null', 'NULL', 'n/a', 'N/A', 'na', 'NA', 
                                  'none', 'NONE', 'unknown', 'UNKNOWN', '']
            df_cleaned[col] = df_cleaned[col].replace(null_representations, np.nan)
            
            # Remove special characters (optional, depends on data)
            # df_cleaned[col] = df_cleaned[col].str.replace(r'[^\w\s]', '', regex=True)
        
        return df_cleaned
    
    def encode_categorical_variables(self, df: pd.DataFrame, method: str = 'label',
                                   columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Encode categorical variables using various methods."""
        df_encoded = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == 'label':
                le = LabelEncoder()
                # Handle NaN values
                mask = df_encoded[col].notna()
                df_encoded.loc[mask, col] = le.fit_transform(df_encoded.loc[mask, col])
                self.encoders[col] = le
            
            elif method == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(df_encoded[col], prefix=col, dummy_na=True)
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
            
            elif method == 'frequency':
                # Frequency encoding
                frequency_map = df_encoded[col].value_counts().to_dict()
                df_encoded[col] = df_encoded[col].map(frequency_map)
        
        return df_encoded
    
    def scale_numeric_features(self, df: pd.DataFrame, method: str = 'standard',
                             columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Scale numeric features using various methods."""
        df_scaled = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == 'standard':
                scaler = StandardScaler()
                df_scaled[col] = scaler.fit_transform(df_scaled[[col]])
                self.scalers[col] = scaler
            
            elif method == 'minmax':
                scaler = MinMaxScaler()
                df_scaled[col] = scaler.fit_transform(df_scaled[[col]])
                self.scalers[col] = scaler
            
            elif method == 'robust':
                # Robust scaling (median and IQR)
                median = df_scaled[col].median()
                q75 = df_scaled[col].quantile(0.75)
                q25 = df_scaled[col].quantile(0.25)
                iqr = q75 - q25
                df_scaled[col] = (df_scaled[col] - median) / iqr
        
        return df_scaled
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None,
                         keep: str = 'first') -> pd.DataFrame:
        """Remove duplicate rows."""
        return df.drop_duplicates(subset=subset, keep=keep)
    
    def detect_and_fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically detect and fix data types."""
        df_fixed = df.copy()
        
        for col in df.columns:
            # Try to convert to numeric if possible
            if df[col].dtype == 'object':
                try:
                    # Remove any non-numeric characters except decimal points and minus signs
                    cleaned_series = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                    numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                    
                    # If more than 80% of values can be converted to numeric, convert the column
                    non_null_count = numeric_series.notna().sum()
                    total_count = len(df[col].dropna())
                    
                    if total_count > 0 and (non_null_count / total_count) > 0.8:
                        df_fixed[col] = numeric_series
                except:
                    pass
            
            # Try to convert to datetime if the column name suggests it's a date
            if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp']):
                try:
                    df_fixed[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        
        return df_fixed
    
    def get_cleaning_summary(self, original: pd.DataFrame, cleaned: pd.DataFrame) -> Dict:
        """Generate summary of cleaning operations performed."""
        summary = {
            'rows_removed': original.shape[0] - cleaned.shape[0],
            'columns_removed': original.shape[1] - cleaned.shape[1],
            'missing_values_handled': {
                'before': original.isnull().sum().sum(),
                'after': cleaned.isnull().sum().sum(),
                'reduction': original.isnull().sum().sum() - cleaned.isnull().sum().sum()
            },
            'data_types_changed': {},
            'new_columns_added': list(set(cleaned.columns) - set(original.columns))
        }
        
        # Check for data type changes
        for col in original.columns:
            if col in cleaned.columns:
                if str(original[col].dtype) != str(cleaned[col].dtype):
                    summary['data_types_changed'][col] = {
                        'from': str(original[col].dtype),
                        'to': str(cleaned[col].dtype)
                    }
        
        return summary
