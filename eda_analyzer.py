import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class EDAAnalyzer:
    def __init__(self):
        self.analysis_results = {}
    
    def basic_info(self, df):
        """Generate basic information about the dataset"""
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        }
        return info
    
    def statistical_summary(self, df):
        """Generate statistical summary for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if numeric_cols.empty:
            return None
        
        summary = df[numeric_cols].describe()
        
        # Add additional statistics
        additional_stats = pd.DataFrame(index=numeric_cols)
        additional_stats['skewness'] = df[numeric_cols].skew()
        additional_stats['kurtosis'] = df[numeric_cols].kurtosis()
        additional_stats['missing_count'] = df[numeric_cols].isnull().sum()
        additional_stats['missing_percentage'] = (df[numeric_cols].isnull().sum() / len(df)) * 100
        
        return {
            'basic_stats': summary,
            'additional_stats': additional_stats
        }
    
    def categorical_analysis(self, df):
        """Analyze categorical columns"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if categorical_cols.empty:
            return None
        
        analysis = {}
        for col in categorical_cols:
            analysis[col] = {
                'unique_count': df[col].nunique(),
                'unique_values': df[col].unique().tolist(),
                'value_counts': df[col].value_counts().to_dict(),
                'missing_count': df[col].isnull().sum(),
                'missing_percentage': (df[col].isnull().sum() / len(df)) * 100
            }
        
        return analysis
    
    def correlation_analysis(self, df):
        """Perform correlation analysis"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return None
        
        correlation_matrix = df[numeric_cols].corr()
        
        # Find strongest correlations
        correlation_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                correlation_pairs.append({
                    'variable1': correlation_matrix.columns[i],
                    'variable2': correlation_matrix.columns[j],
                    'correlation': correlation_matrix.iloc[i, j]
                })
        
        # Sort by absolute correlation
        correlation_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            'correlation_matrix': correlation_matrix,
            'top_correlations': correlation_pairs[:10]
        }
    
    def outlier_analysis(self, df, method='IQR'):
        """Detect outliers in numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_summary = {}
        
        for col in numeric_cols:
            if df[col].notna().sum() < 2:  # Skip if too few non-null values
                continue
                
            if method == 'IQR':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
                outlier_summary[col] = {
                    'method': 'IQR',
                    'outlier_count': len(outliers),
                    'outlier_percentage': (len(outliers) / len(df)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'outlier_indices': outliers.index.tolist()
                }
            
            elif method == 'Z-Score':
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outlier_threshold = 3
                outliers_mask = z_scores > outlier_threshold
                
                outlier_summary[col] = {
                    'method': 'Z-Score',
                    'outlier_count': outliers_mask.sum(),
                    'outlier_percentage': (outliers_mask.sum() / len(df[col].dropna())) * 100,
                    'threshold': outlier_threshold
                }
        
        return outlier_summary
    
    def distribution_analysis(self, df):
        """Analyze distributions of numeric variables"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        distribution_summary = {}
        
        for col in numeric_cols:
            if df[col].notna().sum() < 2:
                continue
            
            # Normality test
            stat, p_value = stats.normaltest(df[col].dropna())
            is_normal = p_value > 0.05
            
            distribution_summary[col] = {
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis(),
                'normality_test_stat': stat,
                'normality_test_p_value': p_value,
                'is_normal': is_normal,
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
        
        return distribution_summary
    
    def feature_importance_for_target(self, df, target_col):
        """Calculate feature importance for a target variable"""
        if target_col not in df.columns:
            return None
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        importance_scores = {}
        
        # Calculate correlation with target for numeric features
        for col in numeric_cols:
            if df[col].notna().sum() > 0 and df[target_col].notna().sum() > 0:
                # Remove rows where either column is NaN
                clean_data = df[[col, target_col]].dropna()
                if len(clean_data) > 1:
                    correlation = clean_data[col].corr(clean_data[target_col])
                    if not pd.isna(correlation):
                        importance_scores[col] = abs(correlation)
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def data_quality_assessment(self, df):
        """Comprehensive data quality assessment"""
        quality_metrics = {}
        
        # Completeness
        completeness = (1 - df.isnull().sum() / len(df)) * 100
        quality_metrics['completeness'] = completeness.to_dict()
        
        # Uniqueness (for categorical columns)
        categorical_cols = df.select_dtypes(include=['object']).columns
        uniqueness = {}
        for col in categorical_cols:
            total_values = df[col].notna().sum()
            unique_values = df[col].nunique()
            uniqueness[col] = (unique_values / total_values) * 100 if total_values > 0 else 0
        quality_metrics['uniqueness'] = uniqueness
        
        # Data type consistency
        quality_metrics['data_types'] = df.dtypes.value_counts().to_dict()
        
        # Overall quality score (weighted average of completeness)
        overall_completeness = completeness.mean()
        quality_metrics['overall_quality_score'] = overall_completeness
        
        return quality_metrics
    
    def generate_insights(self, df):
        """Generate automated insights from the data"""
        insights = []
        
        # Dataset size insights
        if len(df) > 10000:
            insights.append(f"Large dataset detected with {len(df):,} rows - consider sampling for faster analysis")
        elif len(df) < 100:
            insights.append(f"Small dataset with only {len(df)} rows - results may not be statistically significant")
        
        # Missing data insights
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_percentage > 20:
            insights.append(f"High missing data rate ({missing_percentage:.1f}%) - data imputation recommended")
        elif missing_percentage > 5:
            insights.append(f"Moderate missing data rate ({missing_percentage:.1f}%) - check data collection process")
        
        # Correlation insights
        correlation_analysis = self.correlation_analysis(df)
        if correlation_analysis:
            top_corr = correlation_analysis['top_correlations'][0]
            if abs(top_corr['correlation']) > 0.8:
                insights.append(f"Strong correlation detected between {top_corr['variable1']} and {top_corr['variable2']} ({top_corr['correlation']:.3f})")
        
        # Outlier insights
        outlier_analysis = self.outlier_analysis(df)
        high_outlier_cols = [col for col, data in outlier_analysis.items() if data['outlier_percentage'] > 5]
        if high_outlier_cols:
            insights.append(f"Columns with high outlier rates (>5%): {', '.join(high_outlier_cols)}")
        
        # Categorical insights
        categorical_analysis = self.categorical_analysis(df)
        if categorical_analysis:
            high_cardinality_cols = [col for col, data in categorical_analysis.items() if data['unique_count'] > len(df) * 0.5]
            if high_cardinality_cols:
                insights.append(f"High cardinality categorical columns detected: {', '.join(high_cardinality_cols)} - consider encoding strategies")
        
        return insights
    
    def full_analysis(self, df, target_col=None):
        """Perform comprehensive EDA analysis"""
        results = {}
        
        results['basic_info'] = self.basic_info(df)
        results['statistical_summary'] = self.statistical_summary(df)
        results['categorical_analysis'] = self.categorical_analysis(df)
        results['correlation_analysis'] = self.correlation_analysis(df)
        results['outlier_analysis'] = self.outlier_analysis(df)
        results['distribution_analysis'] = self.distribution_analysis(df)
        results['data_quality'] = self.data_quality_assessment(df)
        results['insights'] = self.generate_insights(df)
        
        if target_col and target_col in df.columns:
            results['feature_importance'] = self.feature_importance_for_target(df, target_col)
        
        self.analysis_results = results
        return results
