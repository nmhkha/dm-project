import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    def __init__(self):
        self.cleaning_log = []
    
    def clean_text(self, val, remove_special_chars=True, remove_extra_spaces=True, standardize_case=False):
        """Clean text data by removing special characters and extra spaces"""
        if pd.isna(val):
            return val
        
        val = str(val).strip()
        
        if remove_special_chars:
            val = re.sub(r"[!@#$%^&*?]+", "", val)
        
        if remove_extra_spaces:
            val = re.sub(r"\s+", " ", val)
        
        if standardize_case:
            val = val.upper()
        
        return val
    
    def clean_price(self, val):
        """Extract numeric values from price/currency fields"""
        if pd.isna(val):
            return val
        val = re.sub(r"[^0-9.]", "", str(val))
        try:
            return float(val) if val else None
        except:
            return None
    
    def clean_date(self, val):
        """Standardize date formats"""
        if pd.isna(val):
            return val
        
        # Try different date formats
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(str(val), fmt).strftime("%d/%m/%Y")
            except:
                pass
        return None
    
    def fix_categorical_inconsistencies(self, series):
        """Fix common categorical data inconsistencies"""
        if series.dtype != 'object':
            return series
        
        # Create a mapping for common inconsistencies
        value_counts = series.value_counts()
        mapping = {}
        
        # Group similar values
        unique_values = series.dropna().unique()
        for val in unique_values:
            val_clean = str(val).strip().upper()
            
            # Common mappings
            if val_clean in ['H', 'HIGH']:
                mapping[val] = 'H'
            elif val_clean in ['M', 'MED', 'MEDIUM']:
                mapping[val] = 'M'
            elif val_clean in ['L', 'LOW', 'L ']:  # Note the space after L
                mapping[val] = 'L'
            elif val_clean in ['??', 'UNKNOWN', 'NULL', '']:
                mapping[val] = np.nan
        
        # Apply mapping
        if mapping:
            series = series.replace(mapping)
            self.cleaning_log.append(f"Fixed categorical inconsistencies: {mapping}")
        
        return series
    
    def detect_outliers(self, df, columns, method='IQR', threshold=0.05):
        """Detect outliers using various methods"""
        outlier_indices = set()
        
        if method == 'IQR':
            for col in columns:
                if df[col].dtype in ['int64', 'float64']:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
                    outlier_indices.update(col_outliers)
        
        elif method == 'Z-Score':
            for col in columns:
                if df[col].dtype in ['int64', 'float64']:
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    col_outliers = df[z_scores > 3].index
                    outlier_indices.update(col_outliers)
        
        elif method == 'Isolation Forest':
            # Use only numeric columns for Isolation Forest
            numeric_data = df[columns].select_dtypes(include=[np.number])
            if not numeric_data.empty:
                # Handle missing values
                numeric_data_clean = numeric_data.dropna()
                if len(numeric_data_clean) > 0:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(numeric_data_clean)
                    
                    iso_forest = IsolationForest(contamination=threshold, random_state=42)
                    outlier_labels = iso_forest.fit_predict(scaled_data)
                    
                    outlier_indices = set(numeric_data_clean.index[outlier_labels == -1])
        
        return list(outlier_indices)
    
    def handle_missing_data(self, df, strategy='mean'):
        """Handle missing data using various strategies"""
        df_cleaned = df.copy()
        
        if strategy == 'drop':
            df_cleaned = df_cleaned.dropna()
            self.cleaning_log.append(f"Dropped {len(df) - len(df_cleaned)} rows with missing values")
        
        elif strategy in ['mean', 'median', 'mode']:
            numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
            categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
            
            # Handle numeric columns
            if not numeric_columns.empty:
                if strategy == 'mean':
                    imputer = SimpleImputer(strategy='mean')
                elif strategy == 'median':
                    imputer = SimpleImputer(strategy='median')
                else:  # mode for numeric (most frequent)
                    imputer = SimpleImputer(strategy='most_frequent')
                
                df_cleaned[numeric_columns] = imputer.fit_transform(df_cleaned[numeric_columns])
            
            # Handle categorical columns with mode
            if not categorical_columns.empty:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df_cleaned[categorical_columns] = cat_imputer.fit_transform(df_cleaned[categorical_columns].astype(str))
            
            self.cleaning_log.append(f"Imputed missing values using {strategy} strategy")
        
        elif strategy == 'forward_fill':
            df_cleaned = df_cleaned.ffill()
            self.cleaning_log.append("Applied forward fill for missing values")
        
        elif strategy == 'backward_fill':
            df_cleaned = df_cleaned.bfill()
            self.cleaning_log.append("Applied backward fill for missing values")
        
        return df_cleaned
    
    def clean_data(self, df, remove_special_chars=True, remove_extra_spaces=True, 
                   standardize_case=False, fix_categorical=True, clean_numeric=True,
                   missing_strategy='mean', detect_outliers=True, outlier_method='IQR',
                   outlier_threshold=0.05):
        """Main cleaning function that applies all cleaning steps"""
        
        self.cleaning_log = []  # Reset log
        df_cleaned = df.copy()
        
        # Text cleaning for object columns
        text_columns = df_cleaned.select_dtypes(include=['object']).columns
        for col in text_columns:
            if not any(keyword in col.lower() for keyword in ['id', 'product']):  # Skip ID columns
                df_cleaned[col] = df_cleaned[col].apply(
                    lambda x: self.clean_text(x, remove_special_chars, remove_extra_spaces, standardize_case)
                )
        
        self.cleaning_log.append(f"Applied text cleaning to {len(text_columns)} text columns")
        
        # Fix categorical inconsistencies
        if fix_categorical:
            categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if not any(keyword in col.lower() for keyword in ['id', 'product']):
                    df_cleaned[col] = self.fix_categorical_inconsistencies(df_cleaned[col])
        
        # Clean numeric data (remove obvious errors)
        if clean_numeric:
            numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                # Remove negative values where they don't make sense
                if any(keyword in col.lower() for keyword in ['temperature', 'speed', 'wear', 'torque']):
                    negative_count = (df_cleaned[col] < 0).sum()
                    if negative_count > 0:
                        df_cleaned.loc[df_cleaned[col] < 0, col] = np.nan
                        self.cleaning_log.append(f"Removed {negative_count} negative values from {col}")
        
        # Detect and flag outliers
        if detect_outliers and outlier_method:
            numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
            # Remove binary/categorical numeric columns
            numeric_columns = [col for col in numeric_columns if df_cleaned[col].nunique() > 2]
            
            if numeric_columns:
                outlier_indices = self.detect_outliers(
                    df_cleaned, numeric_columns, outlier_method, outlier_threshold
                )
                
                if outlier_indices:
                    # Add outlier flag column instead of removing
                    df_cleaned['outlier_flag'] = False
                    df_cleaned.loc[outlier_indices, 'outlier_flag'] = True
                    self.cleaning_log.append(f"Flagged {len(outlier_indices)} outliers using {outlier_method}")
        
        # Handle missing data
        df_cleaned = self.handle_missing_data(df_cleaned, missing_strategy)
        
        # Remove duplicates
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        removed_duplicates = initial_rows - len(df_cleaned)
        if removed_duplicates > 0:
            self.cleaning_log.append(f"Removed {removed_duplicates} duplicate rows")
        
        return df_cleaned
    
    def get_cleaning_report(self):
        """Return a report of all cleaning operations performed"""
        return self.cleaning_log
