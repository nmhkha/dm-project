import pandas as pd
import numpy as np
import base64
from io import StringIO, BytesIO
import streamlit as st

def create_download_link(df, filename="data.csv", link_text="Download CSV"):
    """Create a download link for a pandas DataFrame"""
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    b64 = base64.b64encode(csv_data.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def format_number(num, decimal_places=2):
    """Format numbers for display"""
    if pd.isna(num):
        return "N/A"
    if isinstance(num, (int, float)):
        if abs(num) >= 1000000:
            return f"{num/1000000:.{decimal_places}f}M"
        elif abs(num) >= 1000:
            return f"{num/1000:.{decimal_places}f}K"
        else:
            return f"{num:.{decimal_places}f}"
    return str(num)

def safe_divide(numerator, denominator, default=0):
    """Safely divide two numbers, avoiding division by zero"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default

def detect_column_types(df):
    """Detect and categorize column types"""
    column_types = {
        'numeric': [],
        'categorical': [],
        'datetime': [],
        'id': [],
        'binary': []
    }
    
    for col in df.columns:
        # Check if it's an ID column
        if any(keyword in col.lower() for keyword in ['id', 'udi', 'product']):
            column_types['id'].append(col)
        # Check if it's binary (0/1 or True/False)
        elif df[col].nunique() == 2 and set(df[col].dropna().unique()).issubset({0, 1, True, False}):
            column_types['binary'].append(col)
        # Check if it's numeric
        elif df[col].dtype in ['int64', 'float64']:
            column_types['numeric'].append(col)
        # Check if it's datetime
        elif df[col].dtype == 'datetime64[ns]':
            column_types['datetime'].append(col)
        # Otherwise it's categorical
        else:
            column_types['categorical'].append(col)
    
    return column_types

def calculate_missing_patterns(df):
    """Calculate missing data patterns"""
    missing_df = df.isnull()
    
    # Calculate missing patterns
    patterns = missing_df.value_counts()
    
    # Calculate co-occurrence of missing values
    missing_correlation = missing_df.corr()
    
    return {
        'patterns': patterns,
        'correlation': missing_correlation,
        'total_complete_rows': (~missing_df.any(axis=1)).sum(),
        'rows_with_any_missing': missing_df.any(axis=1).sum()
    }

def suggest_cleaning_actions(df):
    """Suggest appropriate cleaning actions based on data analysis"""
    suggestions = []
    
    # Missing data suggestions
    missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    if missing_percentage > 50:
        suggestions.append("Consider using a different dataset - over 50% missing data")
    elif missing_percentage > 20:
        suggestions.append("High missing data - consider advanced imputation techniques")
    elif missing_percentage > 5:
        suggestions.append("Moderate missing data - standard imputation should work")
    
    # Duplicate suggestions
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        suggestions.append(f"Remove {duplicates} duplicate rows")
    
    # Categorical data suggestions
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.5:
            suggestions.append(f"Consider encoding high-cardinality column: {col}")
        
        # Check for inconsistent categorical values
        values = df[col].dropna().unique()
        similar_values = []
        for val in values:
            val_lower = str(val).lower().strip()
            for other_val in values:
                other_lower = str(other_val).lower().strip()
                if val != other_val and val_lower == other_lower:
                    similar_values.append((val, other_val))
        
        if similar_values:
            suggestions.append(f"Standardize categorical values in {col}: {similar_values[:3]}")
    
    # Outlier suggestions
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if df[col].notna().sum() > 0:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            
            if len(outliers) > len(df) * 0.05:  # More than 5% outliers
                suggestions.append(f"High outlier rate in {col} - consider outlier treatment")
    
    return suggestions

def validate_data_quality(df):
    """Validate data quality and return quality score"""
    score = 100
    issues = []
    
    # Completeness score (30% weight)
    missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    completeness_score = max(0, 100 - missing_percentage)
    score = score * 0.7 + completeness_score * 0.3
    
    if missing_percentage > 20:
        issues.append(f"High missing data rate: {missing_percentage:.1f}%")
    
    # Consistency score (20% weight)
    consistency_issues = 0
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        # Check for inconsistent formatting
        values = df[col].dropna().astype(str)
        inconsistent = 0
        for val in values.unique():
            variations = values[values.str.lower().str.strip() == val.lower().strip()]
            if len(variations.unique()) > 1:
                inconsistent += 1
        
        if inconsistent > 0:
            consistency_issues += 1
    
    if consistency_issues > 0:
        consistency_score = max(0, 100 - (consistency_issues / len(categorical_cols)) * 100)
        score = score * 0.8 + consistency_score * 0.2
        issues.append(f"Formatting inconsistencies in {consistency_issues} categorical columns")
    
    # Validity score (20% weight)
    validity_issues = 0
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numeric_cols:
        # Check for impossible values (e.g., negative temperatures in Kelvin)
        if 'temperature' in col.lower() and 'k' in col.lower():
            if (df[col] < 0).any():
                validity_issues += 1
                issues.append(f"Negative Kelvin temperatures found in {col}")
        
        # Check for extreme outliers
        if df[col].notna().sum() > 0:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            extreme_outliers = (z_scores > 5).sum()
            if extreme_outliers > 0:
                validity_issues += 1
    
    if validity_issues > 0:
        validity_score = max(0, 100 - (validity_issues / len(numeric_cols)) * 100)
        score = score * 0.8 + validity_score * 0.2
    
    return {
        'overall_score': round(score, 2),
        'issues': issues,
        'completeness_percentage': round(100 - missing_percentage, 2),
        'consistency_issues': consistency_issues,
        'validity_issues': validity_issues
    }
