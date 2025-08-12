import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
import re
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

from data_cleaner import DataCleaner
from eda_analyzer import EDAAnalyzer
from utils import create_download_link

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance Data Analyzer",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e3440;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🔧 Predictive Maintenance Data Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📤 Data Upload & Cleaning", "📊 Exploratory Data Analysis", "🎯 Machine Failure Analysis", "📈 Advanced Analytics"]
    )
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    if 'cleaner' not in st.session_state:
        st.session_state.cleaner = DataCleaner()
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = EDAAnalyzer()
    
    # Route to different pages
    if page == "📤 Data Upload & Cleaning":
        data_upload_and_cleaning()
    elif page == "📊 Exploratory Data Analysis":
        exploratory_data_analysis()
    elif page == "🎯 Machine Failure Analysis":
        machine_failure_analysis()
    elif page == "📈 Advanced Analytics":
        advanced_analytics()

def data_upload_and_cleaning():
    st.markdown('<h2 class="section-header">Data Upload & Cleaning</h2>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your predictive maintenance dataset in CSV format"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Show data preview
            st.markdown('<h3 class="section-header">Data Preview</h3>', unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data quality assessment
            st.markdown('<h3 class="section-header">Data Quality Assessment</h3>', unsafe_allow_html=True)
            
            # Missing data visualization
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if not missing_data.empty:
                fig = px.bar(
                    x=missing_data.index,
                    y=missing_data.values,
                    title="Missing Values by Column",
                    labels={'x': 'Columns', 'y': 'Missing Count'}
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("✅ No missing values detected!")
            
            # Data cleaning configuration
            st.markdown('<h3 class="section-header">Cleaning Configuration</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Text Cleaning Options")
                remove_special_chars = st.checkbox("Remove special characters (!@#$%^&*?)", True)
                remove_extra_spaces = st.checkbox("Remove extra whitespace", True)
                standardize_case = st.checkbox("Standardize text case", False)
                
                st.subheader("Categorical Data Cleaning")
                fix_categorical = st.checkbox("Fix categorical inconsistencies", True)
                if fix_categorical:
                    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
                    if categorical_columns:
                        selected_cat_cols = st.multiselect(
                            "Select categorical columns to clean:",
                            categorical_columns,
                            default=categorical_columns
                        )
            
            with col2:
                st.subheader("Numeric Data Cleaning")
                clean_numeric = st.checkbox("Clean numeric columns", True)
                
                st.subheader("Missing Data Handling")
                missing_strategy = st.selectbox(
                    "Missing data strategy:",
                    ["mean", "median", "mode", "drop", "forward_fill", "backward_fill"]
                )
                
                st.subheader("Outlier Detection")
                detect_outliers = st.checkbox("Detect and flag outliers", True)
                if detect_outliers:
                    outlier_method = st.selectbox(
                        "Outlier detection method:",
                        ["IQR", "Z-Score", "Isolation Forest"]
                    )
                    outlier_threshold = st.slider("Sensitivity", 0.01, 0.1, 0.05)
                else:
                    outlier_method = None
                    outlier_threshold = None
            
            # Clean data button
            if st.button("🧹 Clean Data", type="primary"):
                with st.spinner("Cleaning data..."):
                    cleaned_df = st.session_state.cleaner.clean_data(
                        df.copy(),
                        remove_special_chars=remove_special_chars,
                        remove_extra_spaces=remove_extra_spaces,
                        standardize_case=standardize_case,
                        fix_categorical=fix_categorical,
                        clean_numeric=clean_numeric,
                        missing_strategy=missing_strategy,
                        detect_outliers=detect_outliers,
                        outlier_method=outlier_method if detect_outliers else None,
                        outlier_threshold=outlier_threshold if detect_outliers else None
                    )
                    
                    st.session_state.cleaned_data = cleaned_df
                    
                    st.success("✅ Data cleaning completed!")
                    
                    # Show cleaning results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Original Shape", f"{df.shape[0]} x {df.shape[1]}")
                        st.metric("Original Missing Values", df.isnull().sum().sum())
                    with col2:
                        st.metric("Cleaned Shape", f"{cleaned_df.shape[0]} x {cleaned_df.shape[1]}")
                        st.metric("Remaining Missing Values", cleaned_df.isnull().sum().sum())
                    
                    # Show cleaned data preview
                    st.markdown('<h3 class="section-header">Cleaned Data Preview</h3>', unsafe_allow_html=True)
                    st.dataframe(cleaned_df.head(10), use_container_width=True)
                    
                    # Download link
                    csv_buffer = StringIO()
                    cleaned_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Download Cleaned Data",
                        data=csv_data,
                        file_name="cleaned_data.csv",
                        mime="text/csv",
                        type="secondary"
                    )
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

def exploratory_data_analysis():
    st.markdown('<h2 class="section-header">Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload & Cleaning' page.")
        return
    
    # Choose which dataset to analyze
    data_choice = st.radio(
        "Choose dataset to analyze:",
        ["Original Data", "Cleaned Data"] if st.session_state.cleaned_data is not None else ["Original Data"]
    )
    
    df = st.session_state.cleaned_data if data_choice == "Cleaned Data" and st.session_state.cleaned_data is not None else st.session_state.data
    
    # Dataset overview
    st.markdown('<h3 class="section-header">Dataset Overview</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Numeric Columns", df.select_dtypes(include=[np.number]).shape[1])
    with col4:
        st.metric("Categorical Columns", df.select_dtypes(include=['object']).shape[1])
    
    # Column analysis
    st.markdown('<h3 class="section-header">Column Analysis</h3>', unsafe_allow_html=True)
    
    # Select columns for analysis
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    analysis_type = st.selectbox(
        "Choose analysis type:",
        ["Numeric Analysis", "Categorical Analysis", "Correlation Analysis", "Distribution Analysis"]
    )
    
    if analysis_type == "Numeric Analysis" and numeric_columns:
        selected_numeric = st.multiselect(
            "Select numeric columns:",
            numeric_columns,
            default=numeric_columns[:5] if len(numeric_columns) > 5 else numeric_columns
        )
        
        if selected_numeric:
            # Statistical summary
            st.subheader("Statistical Summary")
            st.dataframe(df[selected_numeric].describe(), use_container_width=True)
            
            # Box plots
            st.subheader("Distribution Plots")
            for col in selected_numeric:
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=[f'{col} - Box Plot', f'{col} - Histogram'],
                    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Box plot
                fig.add_trace(
                    go.Box(y=df[col], name=col, showlegend=False),
                    row=1, col=1
                )
                
                # Histogram
                fig.add_trace(
                    go.Histogram(x=df[col], name=col, showlegend=False, nbinsx=30),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, title_text=f"Analysis of {col}")
                st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Categorical Analysis" and categorical_columns:
        selected_categorical = st.multiselect(
            "Select categorical columns:",
            categorical_columns,
            default=categorical_columns[:3] if len(categorical_columns) > 3 else categorical_columns
        )
        
        if selected_categorical:
            for col in selected_categorical:
                st.subheader(f"Analysis of {col}")
                
                value_counts = df[col].value_counts()
                
                # Bar chart
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Distribution of {col}",
                    labels={'x': col, 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show value counts table
                st.dataframe(value_counts.reset_index().rename(columns={'index': col, col: 'Count'}))
    
    elif analysis_type == "Correlation Analysis" and len(numeric_columns) > 1:
        st.subheader("Correlation Matrix")
        
        corr_matrix = df[numeric_columns].corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Correlation Matrix",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Strongest correlations
        st.subheader("Strongest Correlations")
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df['Correlation_abs'] = corr_df['Correlation'].abs()
        corr_df = corr_df.sort_values('Correlation_abs', ascending=False).drop('Correlation_abs', axis=1)
        st.dataframe(corr_df.head(10), use_container_width=True)
    
    elif analysis_type == "Distribution Analysis" and numeric_columns:
        selected_cols = st.multiselect(
            "Select columns for distribution comparison:",
            numeric_columns,
            default=numeric_columns[:3] if len(numeric_columns) > 3 else numeric_columns
        )
        
        if selected_cols and len(selected_cols) > 1:
            # Pair plot equivalent using scatter plots
            st.subheader("Pairwise Relationships")
            
            for i, col1 in enumerate(selected_cols):
                for col2 in selected_cols[i+1:]:
                    fig = px.scatter(
                        df, x=col1, y=col2,
                        title=f"{col1} vs {col2}",
                        trendline="ols"
                    )
                    st.plotly_chart(fig, use_container_width=True)

def machine_failure_analysis():
    st.markdown('<h2 class="section-header">Machine Failure Analysis</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload & Cleaning' page.")
        return
    
    # Choose which dataset to analyze
    data_choice = st.radio(
        "Choose dataset to analyze:",
        ["Original Data", "Cleaned Data"] if st.session_state.cleaned_data is not None else ["Original Data"]
    )
    
    df = st.session_state.cleaned_data if data_choice == "Cleaned Data" and st.session_state.cleaned_data is not None else st.session_state.data
    
    # Look for failure-related columns
    failure_columns = [col for col in df.columns if 'failure' in col.lower() or 'fail' in col.lower()]
    target_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['twf', 'hdf', 'pwf', 'osf', 'rnf', 'machine', 'failure'])]
    
    if not failure_columns and not target_columns:
        st.warning("⚠️ No failure-related columns found in the dataset.")
        return
    
    # Failure overview
    st.markdown('<h3 class="section-header">Failure Overview</h3>', unsafe_allow_html=True)
    
    if 'Machine failure' in df.columns:
        failure_col = 'Machine failure'
    elif failure_columns:
        failure_col = failure_columns[0]
    else:
        failure_col = st.selectbox("Select the main failure column:", target_columns)
    
    if failure_col in df.columns:
        failure_rate = df[failure_col].mean()
        total_failures = df[failure_col].sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Total Failures", int(total_failures))
        with col3:
            st.metric("Failure Rate", f"{failure_rate:.2%}")
        
        # Failure distribution
        st.subheader("Failure Distribution")
        failure_counts = df[failure_col].value_counts()
        
        fig = px.pie(
            values=failure_counts.values,
            names=['No Failure', 'Failure'] if len(failure_counts) == 2 else failure_counts.index,
            title="Machine Failure Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Failure type analysis
        failure_type_columns = [col for col in df.columns if col in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']]
        
        if failure_type_columns:
            st.subheader("Failure Type Analysis")
            
            failure_types = {}
            for col in failure_type_columns:
                failure_types[col] = df[col].sum()
            
            if any(failure_types.values()):
                fig = px.bar(
                    x=list(failure_types.keys()),
                    y=list(failure_types.values()),
                    title="Failure Types Distribution",
                    labels={'x': 'Failure Type', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Failure type definitions
                st.info("""
                **Failure Type Definitions:**
                - **TWF**: Tool Wear Failure
                - **HDF**: Heat Dissipation Failure
                - **PWF**: Power Failure
                - **OSF**: Overstrain Failure
                - **RNF**: Random Failure
                """)
        
        # Feature importance for failure prediction
        st.subheader("Feature Analysis for Failure Prediction")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if failure_col in numeric_columns:
            numeric_columns.remove(failure_col)
        
        # Remove other failure columns from features
        for col in failure_type_columns:
            if col in numeric_columns:
                numeric_columns.remove(col)
        
        if numeric_columns:
            # Calculate correlation with failure
            correlations = {}
            for col in numeric_columns:
                if df[col].notna().sum() > 0:
                    corr = df[col].corr(df[failure_col])
                    if not pd.isna(corr):
                        correlations[col] = abs(corr)
            
            if correlations:
                # Sort by absolute correlation
                sorted_corr = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))
                
                fig = px.bar(
                    x=list(sorted_corr.values()),
                    y=list(sorted_corr.keys()),
                    orientation='h',
                    title="Feature Correlation with Machine Failure (Absolute Values)",
                    labels={'x': 'Absolute Correlation', 'y': 'Features'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show top correlated features
                st.subheader("Top Features Correlated with Failure")
                top_features = list(sorted_corr.keys())[:5]
                
                for feature in top_features:
                    st.subheader(f"Analysis: {feature}")
                    
                    # Box plot by failure status
                    fig = px.box(
                        df, x=failure_col, y=feature,
                        title=f"{feature} Distribution by Failure Status"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics by failure status
                    stats_by_failure = df.groupby(failure_col)[feature].describe()
                    st.dataframe(stats_by_failure, use_container_width=True)

def advanced_analytics():
    st.markdown('<h2 class="section-header">Advanced Analytics</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload & Cleaning' page.")
        return
    
    # Choose which dataset to analyze
    data_choice = st.radio(
        "Choose dataset to analyze:",
        ["Original Data", "Cleaned Data"] if st.session_state.cleaned_data is not None else ["Original Data"]
    )
    
    df = st.session_state.cleaned_data if data_choice == "Cleaned Data" and st.session_state.cleaned_data is not None else st.session_state.data
    
    # Advanced analytics options
    analytics_type = st.selectbox(
        "Choose advanced analytics:",
        ["Outlier Detection", "Data Quality Metrics", "Feature Engineering", "Time Series Analysis"]
    )
    
    if analytics_type == "Outlier Detection":
        st.subheader("Outlier Detection")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_columns:
            selected_columns = st.multiselect(
                "Select columns for outlier detection:",
                numeric_columns,
                default=numeric_columns[:3] if len(numeric_columns) > 3 else numeric_columns
            )
            
            outlier_method = st.selectbox(
                "Select outlier detection method:",
                ["IQR Method", "Z-Score", "Isolation Forest"]
            )
            
            if selected_columns:
                if outlier_method == "IQR Method":
                    st.subheader("IQR Method Results")
                    
                    for col in selected_columns:
                        if df[col].notna().sum() > 0:
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            
                            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                            
                            st.write(f"**{col}:**")
                            st.write(f"- Outliers detected: {len(outliers)}")
                            st.write(f"- Lower bound: {lower_bound:.2f}")
                            st.write(f"- Upper bound: {upper_bound:.2f}")
                            
                            # Visualization
                            fig = px.box(df, y=col, title=f"Box Plot: {col}")
                            st.plotly_chart(fig, use_container_width=True)
                
                elif outlier_method == "Isolation Forest":
                    st.subheader("Isolation Forest Results")
                    
                    # Prepare data
                    data_for_outliers = df[selected_columns].dropna()
                    
                    if len(data_for_outliers) > 0:
                        # Standardize data
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(data_for_outliers)
                        
                        # Apply Isolation Forest
                        contamination = st.slider("Contamination rate", 0.01, 0.2, 0.1)
                        iso_forest = IsolationForest(contamination=contamination, random_state=42)
                        outlier_labels = iso_forest.fit_predict(scaled_data)
                        
                        # Results
                        n_outliers = sum(outlier_labels == -1)
                        st.write(f"**Outliers detected: {n_outliers} ({n_outliers/len(data_for_outliers):.2%})**")
                        
                        # Add outlier labels to dataframe for visualization
                        data_for_outliers = data_for_outliers.copy()
                        data_for_outliers['Outlier'] = outlier_labels == -1
                        
                        # Scatter plot for first two selected columns
                        if len(selected_columns) >= 2:
                            fig = px.scatter(
                                data_for_outliers,
                                x=selected_columns[0],
                                y=selected_columns[1],
                                color='Outlier',
                                title=f"Outlier Detection: {selected_columns[0]} vs {selected_columns[1]}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
    
    elif analytics_type == "Data Quality Metrics":
        st.subheader("Data Quality Metrics")
        
        # Completeness
        st.write("### Completeness")
        completeness = (1 - df.isnull().sum() / len(df)) * 100
        completeness_df = pd.DataFrame({
            'Column': completeness.index,
            'Completeness (%)': completeness.values
        }).sort_values('Completeness (%)', ascending=False)
        
        fig = px.bar(
            completeness_df,
            x='Completeness (%)',
            y='Column',
            orientation='h',
            title="Data Completeness by Column"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Data types
        st.write("### Data Types")
        dtype_summary = df.dtypes.value_counts()
        fig = px.pie(
            values=dtype_summary.values,
            names=dtype_summary.index,
            title="Data Types Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Memory usage
        st.write("### Memory Usage")
        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum() / (1024 * 1024)  # Convert to MB
        
        st.metric("Total Memory Usage", f"{total_memory:.2f} MB")
        
        memory_df = pd.DataFrame({
            'Column': memory_usage.index,
            'Memory (KB)': memory_usage.values / 1024
        }).sort_values('Memory (KB)', ascending=False)
        
        fig = px.bar(
            memory_df.head(10),
            x='Memory (KB)',
            y='Column',
            orientation='h',
            title="Memory Usage by Column (Top 10)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif analytics_type == "Feature Engineering":
        st.subheader("Feature Engineering")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) >= 2:
            st.write("### Create New Features")
            
            # Feature creation options
            feature_type = st.selectbox(
                "Select feature type:",
                ["Ratio Features", "Polynomial Features", "Binning", "Statistical Features"]
            )
            
            if feature_type == "Ratio Features":
                col1 = st.selectbox("Numerator column:", numeric_columns, key="ratio_num")
                col2 = st.selectbox("Denominator column:", [c for c in numeric_columns if c != col1], key="ratio_den")
                
                if col1 and col2:
                    feature_name = st.text_input("Feature name:", f"{col1}_to_{col2}_ratio")
                    
                    if st.button("Create Ratio Feature"):
                        # Avoid division by zero
                        new_feature = df[col1] / (df[col2] + 1e-8)
                        
                        st.write(f"**Created feature: {feature_name}**")
                        st.write(f"Statistics:")
                        st.dataframe(pd.DataFrame(new_feature.describe()).T)
                        
                        # Visualization
                        fig = px.histogram(new_feature.dropna(), title=f"Distribution of {feature_name}")
                        st.plotly_chart(fig, use_container_width=True)
            
            elif feature_type == "Binning":
                selected_col = st.selectbox("Select column to bin:", numeric_columns)
                n_bins = st.slider("Number of bins:", 3, 10, 5)
                
                if st.button("Create Binned Feature"):
                    binned_feature, bin_edges = pd.cut(df[selected_col], bins=n_bins, labels=False, retbins=True)
                    
                    st.write(f"**Binned {selected_col} into {n_bins} categories**")
                    
                    # Show bin distribution
                    bin_counts = pd.Series(binned_feature).value_counts().sort_index()
                    fig = px.bar(
                        x=bin_counts.index,
                        y=bin_counts.values,
                        title=f"Distribution of Binned {selected_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
