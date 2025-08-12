"""
Improved EDA Analyzer based on the uploaded notebooks
Phiên bản cải tiến từ EDA_1755016864579.ipynb
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class ImprovedEDAAnalyzer:
    def __init__(self):
        self.numeric_columns = [
            'Air temperature [K]',
            'Process temperature [K]', 
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]'
        ]
        self.failure_columns = [
            'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
        ]
    
    def basic_dataset_info(self, df):
        """Thông tin cơ bản về dataset"""
        info = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        }
        
        # Phân tích missing values
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        info['missing_percentage'] = missing_percentage.to_dict()
        
        return info
    
    def analyze_numeric_columns(self, df):
        """Phân tích các cột số theo notebook EDA"""
        results = {}
        
        # Lấy các cột numeric có trong data
        available_numeric = [col for col in self.numeric_columns if col in df.columns]
        
        for col in available_numeric:
            if df[col].notna().sum() > 0:  # Chỉ phân tích nếu có dữ liệu
                col_data = df[col].dropna()
                
                results[col] = {
                    'count': len(col_data),
                    'mean': col_data.mean(),
                    'std': col_data.std(), 
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'median': col_data.median(),
                    'skewness': col_data.skew(),
                    'kurtosis': col_data.kurtosis(),
                    'missing_count': df[col].isnull().sum(),
                    'missing_percentage': (df[col].isnull().sum() / len(df)) * 100
                }
                
                # Outlier detection bằng IQR
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75) 
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                results[col]['outlier_count'] = len(outliers)
                results[col]['outlier_percentage'] = (len(outliers) / len(col_data)) * 100
        
        return results
    
    def analyze_type_column(self, df):
        """Phân tích cột Type như trong notebook"""
        if 'Type' not in df.columns:
            return None
        
        type_analysis = {
            'unique_values': df['Type'].unique().tolist(),
            'value_counts': df['Type'].value_counts().to_dict(),
            'missing_count': df['Type'].isnull().sum()
        }
        
        # Phát hiện các giá trị không consistent (từ notebook)
        inconsistent_values = []
        for val in df['Type'].unique():
            if pd.notna(val):
                val_str = str(val).strip()
                if val_str not in ['L', 'M', 'H']:
                    inconsistent_values.append(val)
        
        type_analysis['inconsistent_values'] = inconsistent_values
        
        return type_analysis
    
    def analyze_machine_failures(self, df):
        """Phân tích machine failure theo notebook"""
        failure_analysis = {}
        
        # Lấy các cột failure có trong data
        available_failure_cols = [col for col in self.failure_columns if col in df.columns]
        
        for col in available_failure_cols:
            failure_analysis[col] = {
                'value_counts': df[col].value_counts().to_dict(),
                'failure_rate': df[col].mean() if df[col].dtype in ['int64', 'float64'] else None,
                'total_failures': df[col].sum() if df[col].dtype in ['int64', 'float64'] else None
            }
        
        # Tính tổng quan failure rate
        if 'Machine failure' in df.columns:
            total_failures = df['Machine failure'].sum()
            failure_rate = df['Machine failure'].mean()
            
            failure_analysis['summary'] = {
                'total_samples': len(df),
                'total_failures': int(total_failures),
                'failure_rate': round(failure_rate, 4),
                'failure_percentage': round(failure_rate * 100, 2)
            }
        
        return failure_analysis
    
    def correlation_analysis(self, df):
        """Phân tích correlation giữa các biến numeric"""
        # Lấy các cột numeric có trong data
        available_numeric = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
        
        if len(available_numeric) < 2:
            return None
        
        # Tính correlation matrix
        corr_matrix = df[available_numeric].corr()
        
        # Tìm correlation pairs mạnh nhất
        correlation_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if not pd.isna(corr_val):
                    correlation_pairs.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j], 
                        'correlation': corr_val
                    })
        
        # Sắp xếp theo độ mạnh correlation
        correlation_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'top_correlations': correlation_pairs[:10],
            'strong_correlations': [p for p in correlation_pairs if abs(p['correlation']) > 0.7]
        }
    
    def create_distribution_plots_data(self, df):
        """Tạo dữ liệu cho biểu đồ phân phối như trong notebook"""
        plot_data = {}
        
        # Lấy các cột numeric có sẵn
        available_numeric = [col for col in self.numeric_columns if col in df.columns]
        
        for col in available_numeric:
            if df[col].notna().sum() > 0:
                col_data = df[col].dropna()
                
                plot_data[col] = {
                    'data': col_data.tolist(),
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max()
                }
                
                # Kiểm tra normality
                if len(col_data) > 8:  # cần ít nhất 8 điểm cho test
                    try:
                        stat, p_value = stats.normaltest(col_data)
                        plot_data[col]['normality_test'] = {
                            'statistic': stat,
                            'p_value': p_value,
                            'is_normal': p_value > 0.05
                        }
                    except:
                        plot_data[col]['normality_test'] = None
        
        return plot_data
    
    def detect_data_quality_issues(self, df):
        """Phát hiện các vấn đề chất lượng dữ liệu"""
        issues = []
        
        # 1. Missing data issues
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_percentage > 10:
            issues.append(f"High missing data rate: {missing_percentage:.1f}%")
        
        # 2. Type column issues
        if 'Type' in df.columns:
            type_issues = []
            for val in df['Type'].unique():
                if pd.notna(val) and str(val).strip() not in ['L', 'M', 'H']:
                    type_issues.append(str(val))
            
            if type_issues:
                issues.append(f"Inconsistent Type values found: {type_issues}")
        
        # 3. Extreme outliers
        numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
        high_outlier_cols = []
        
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                col_data = df[col].dropna()
                if len(col_data) > 1:
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = col_data[(col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)]
                    
                    if len(outliers) / len(col_data) > 0.1:  # >10% outliers
                        high_outlier_cols.append(col)
        
        if high_outlier_cols:
            issues.append(f"High outlier rate in columns: {high_outlier_cols}")
        
        return issues
    
    def full_eda_report(self, df):
        """Báo cáo EDA đầy đủ theo notebook"""
        report = {}
        
        # 1. Thông tin cơ bản
        report['basic_info'] = self.basic_dataset_info(df)
        
        # 2. Phân tích numeric columns
        report['numeric_analysis'] = self.analyze_numeric_columns(df)
        
        # 3. Phân tích Type column
        report['type_analysis'] = self.analyze_type_column(df)
        
        # 4. Phân tích machine failures
        report['failure_analysis'] = self.analyze_machine_failures(df)
        
        # 5. Correlation analysis
        report['correlation_analysis'] = self.correlation_analysis(df)
        
        # 6. Data quality issues
        report['data_quality_issues'] = self.detect_data_quality_issues(df)
        
        # 7. Dữ liệu cho visualization
        report['plot_data'] = self.create_distribution_plots_data(df)
        
        return report

# Helper function
def analyze_maintenance_data(df):
    """
    Hàm tiện ích để phân tích dữ liệu predictive maintenance
    """
    analyzer = ImprovedEDAAnalyzer()
    report = analyzer.full_eda_report(df)
    return report