"""
Streamlit App cải tiến cho Predictive Maintenance Data Analysis
Tích hợp từ các notebook đã upload
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
import base64
import warnings
warnings.filterwarnings('ignore')

# Import improved modules
from improved_cleaner import ImprovedDataCleaner, clean_maintenance_data
from improved_eda import ImprovedEDAAnalyzer, analyze_maintenance_data

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance Data Analyzer",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_download_link(df, filename="cleaned_data.csv"):
    """Tạo link download cho DataFrame"""
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    csv_data = csv_buffer.getvalue()
    
    b64 = base64.b64encode(csv_data.encode('utf-8-sig')).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-link">📥 Tải xuống {filename}</a>'
    return href

def main():
    st.markdown('<h1 class="main-header">🔧 Predictive Maintenance Data Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Điều hướng")
    page = st.sidebar.selectbox(
        "Chọn trang:",
        ["📤 Tải lên & Làm sạch dữ liệu", "📊 Phân tích dữ liệu khám phá", "🎯 Phân tích Machine Failure", "📈 Phân tích nâng cao"]
    )
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    
    # Route to different pages
    if page == "📤 Tải lên & Làm sạch dữ liệu":
        upload_and_cleaning_page()
    elif page == "📊 Phân tích dữ liệu khám phá":
        eda_page()
    elif page == "🎯 Phân tích Machine Failure":
        failure_analysis_page()
    elif page == "📈 Phân tích nâng cao":
        advanced_analysis_page()

def upload_and_cleaning_page():
    st.markdown('<h2 class="section-header">Tải lên & Làm sạch dữ liệu</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    **Các loại file được hỗ trợ:**
    - **dirty_sample_500.csv**: Dữ liệu bẩn mẫu 500 dòng (cho EDA)
    - **dirty_sample_500_worse_plus.csv**: Dữ liệu bẩn nặng (test làm sạch)
    - **ai4i2020.csv**: Dataset gốc Predictive Maintenance
    - **sample_dirty_data.csv**: File mẫu để test ngay
    
    *Hoặc tải file CSV khác tương tự để phân tích*
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Chọn file CSV",
        type="csv",
        help="Tải lên dữ liệu predictive maintenance của bạn"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            
            st.success(f"✅ Tải file thành công! Kích thước: {df.shape[0]} hàng x {df.shape[1]} cột")
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Số hàng", df.shape[0])
            with col2:
                st.metric("Số cột", df.shape[1])
            with col3:
                st.metric("Giá trị thiếu", df.isnull().sum().sum())
            
            # Show data preview
            st.markdown('<h3 class="section-header">Xem trước dữ liệu</h3>', unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data info
            st.markdown('<h3 class="section-header">Thông tin dữ liệu</h3>', unsafe_allow_html=True)
            
            # Missing data chart
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if not missing_data.empty:
                fig = px.bar(
                    x=missing_data.index,
                    y=missing_data.values,
                    title="Số lượng giá trị thiếu theo cột",
                    labels={'x': 'Cột', 'y': 'Số lượng thiếu'}
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("✅ Không có giá trị thiếu nào!")
            
            # Clean data button
            if st.button("🧹 Làm sạch dữ liệu", type="primary"):
                with st.spinner("Đang làm sạch dữ liệu..."):
                    cleaned_df, cleaning_summary = clean_maintenance_data(df)
                    st.session_state.cleaned_data = cleaned_df
                    
                    st.success("✅ Làm sạch dữ liệu hoàn thành!")
                    
                    # Show results comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Dữ liệu gốc")
                        st.metric("Hàng", df.shape[0])
                        st.metric("Cột", df.shape[1])
                        st.metric("Giá trị thiếu", df.isnull().sum().sum())
                        st.metric("Trùng lặp", df.duplicated().sum())
                    
                    with col2:
                        st.subheader("Dữ liệu sau làm sạch")
                        st.metric("Hàng", cleaned_df.shape[0])
                        st.metric("Cột", cleaned_df.shape[1])
                        st.metric("Giá trị thiếu", cleaned_df.isnull().sum().sum())
                        st.metric("Trùng lặp", cleaned_df.duplicated().sum())
                    
                    # Show cleaning log
                    st.subheader("Các bước làm sạch đã thực hiện:")
                    for step in cleaning_summary:
                        st.write(f"✓ {step}")
                    
                    # Download cleaned data
                    st.markdown(
                        create_download_link(cleaned_df, "cleaned_data.csv"),
                        unsafe_allow_html=True
                    )
                    
                    # Show cleaned data preview
                    st.markdown('<h3 class="section-header">Dữ liệu sau khi làm sạch</h3>', unsafe_allow_html=True)
                    st.dataframe(cleaned_df.head(10), use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Lỗi khi tải file: {str(e)}")

def eda_page():
    st.markdown('<h2 class="section-header">Phân tích dữ liệu khám phá (EDA)</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Vui lòng tải lên dữ liệu trước ở trang 'Tải lên & Làm sạch dữ liệu'.")
        return
    
    # Choose dataset
    data_options = ["Dữ liệu gốc"]
    if st.session_state.cleaned_data is not None:
        data_options.append("Dữ liệu đã làm sạch")
    
    data_choice = st.radio("Chọn dữ liệu để phân tích:", data_options)
    
    df = st.session_state.cleaned_data if data_choice == "Dữ liệu đã làm sạch" else st.session_state.data
    
    # Perform EDA analysis
    with st.spinner("Đang phân tích dữ liệu..."):
        eda_report = analyze_maintenance_data(df)
    
    # Display basic info
    st.markdown('<h3 class="section-header">Thông tin tổng quan</h3>', unsafe_allow_html=True)
    
    basic_info = eda_report['basic_info']
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tổng số hàng", basic_info['shape'][0])
    with col2:
        st.metric("Tổng số cột", basic_info['shape'][1])
    with col3:
        st.metric("Bộ nhớ (MB)", basic_info['memory_usage_mb'])
    with col4:
        missing_total = sum(basic_info['missing_values'].values())
        st.metric("Tổng giá trị thiếu", missing_total)
    
    # Data quality issues
    if eda_report['data_quality_issues']:
        st.markdown('<h3 class="section-header">Vấn đề chất lượng dữ liệu</h3>', unsafe_allow_html=True)
        for issue in eda_report['data_quality_issues']:
            st.warning(f"⚠️ {issue}")
    
    # Numeric analysis
    if eda_report['numeric_analysis']:
        st.markdown('<h3 class="section-header">Phân tích các cột số</h3>', unsafe_allow_html=True)
        
        numeric_df = pd.DataFrame(eda_report['numeric_analysis']).T
        st.dataframe(numeric_df, use_container_width=True)
        
        # Distribution plots
        st.subheader("Phân phối dữ liệu")
        plot_data = eda_report['plot_data']
        
        for col_name, col_data in plot_data.items():
            if col_data['data']:
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=[f'{col_name} - Box Plot', f'{col_name} - Histogram']
                )
                
                # Box plot
                fig.add_trace(
                    go.Box(y=col_data['data'], name=col_name, showlegend=False),
                    row=1, col=1
                )
                
                # Histogram
                fig.add_trace(
                    go.Histogram(x=col_data['data'], name=col_name, showlegend=False, nbinsx=30),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, title_text=f"Phân tích {col_name}")
                st.plotly_chart(fig, use_container_width=True)
    
    # Type column analysis
    if eda_report['type_analysis']:
        st.markdown('<h3 class="section-header">Phân tích cột Type</h3>', unsafe_allow_html=True)
        
        type_analysis = eda_report['type_analysis']
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Các giá trị unique:**")
            st.write(type_analysis['unique_values'])
            
            if type_analysis['inconsistent_values']:
                st.warning(f"⚠️ Giá trị không nhất quán: {type_analysis['inconsistent_values']}")
        
        with col2:
            st.write("**Số lượng theo từng loại:**")
            items_list = list(type_analysis['value_counts'].items())
            value_counts_df = pd.DataFrame(items_list, columns=['Type', 'Số lượng'])
            st.dataframe(value_counts_df)
            
            # Type distribution chart
            fig = px.pie(
                values=list(type_analysis['value_counts'].values()),
                names=list(type_analysis['value_counts'].keys()),
                title="Phân phối Type"
            )
            st.plotly_chart(fig)
    
    # Correlation analysis
    if eda_report['correlation_analysis']:
        st.markdown('<h3 class="section-header">Phân tích tương quan</h3>', unsafe_allow_html=True)
        
        corr_analysis = eda_report['correlation_analysis']
        
        # Show top correlations
        if corr_analysis['top_correlations']:
            st.subheader("Top 10 tương quan mạnh nhất")
            corr_df = pd.DataFrame(corr_analysis['top_correlations'])
            st.dataframe(corr_df, use_container_width=True)
        
        # Strong correlations
        if corr_analysis['strong_correlations']:
            st.subheader("Tương quan mạnh (|r| > 0.7)")
            strong_corr_df = pd.DataFrame(corr_analysis['strong_correlations'])
            st.dataframe(strong_corr_df, use_container_width=True)

def failure_analysis_page():
    st.markdown('<h2 class="section-header">Phân tích Machine Failure</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Vui lòng tải lên dữ liệu trước.")
        return
    
    # Choose dataset
    data_options = ["Dữ liệu gốc"]
    if st.session_state.cleaned_data is not None:
        data_options.append("Dữ liệu đã làm sạch")
    
    data_choice = st.radio("Chọn dữ liệu để phân tích:", data_options)
    df = st.session_state.cleaned_data if data_choice == "Dữ liệu đã làm sạch" else st.session_state.data
    
    # Get failure analysis
    eda_report = analyze_maintenance_data(df)
    failure_analysis = eda_report['failure_analysis']
    
    if not failure_analysis:
        st.warning("⚠️ Không tìm thấy cột failure trong dữ liệu.")
        return
    
    # Show failure summary
    if 'summary' in failure_analysis:
        summary = failure_analysis['summary']
        
        st.markdown('<h3 class="section-header">Tổng quan Machine Failure</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tổng mẫu", summary['total_samples'])
        with col2:
            st.metric("Tổng failure", summary['total_failures'])
        with col3:
            st.metric("Tỉ lệ failure", f"{summary['failure_percentage']:.2f}%")
        with col4:
            normal_samples = summary['total_samples'] - summary['total_failures']
            st.metric("Mẫu bình thường", normal_samples)
        
        # Failure distribution pie chart
        fig = px.pie(
            values=[summary['total_failures'], normal_samples],
            names=['Machine Failure', 'Normal Operation'],
            title="Phân phối Machine Failure vs Normal"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Individual failure type analysis
    st.markdown('<h3 class="section-header">Phân tích từng loại failure</h3>', unsafe_allow_html=True)
    
    failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    available_types = [f for f in failure_types if f in failure_analysis]
    
    if available_types:
        # Create subplot for failure types
        cols_per_row = 3
        rows = (len(available_types) + cols_per_row - 1) // cols_per_row
        
        fig = make_subplots(
            rows=rows, cols=cols_per_row,
            subplot_titles=available_types,
            specs=[[{"type": "pie"}] * cols_per_row for _ in range(rows)]
        )
        
        for i, failure_type in enumerate(available_types):
            row = i // cols_per_row + 1
            col = i % cols_per_row + 1
            
            if failure_type in failure_analysis:
                failure_data = failure_analysis[failure_type]
                if 'value_counts' in failure_data:
                    values = list(failure_data['value_counts'].values())
                    labels = [f"{failure_type}={k}" for k in failure_data['value_counts'].keys()]
                    
                    fig.add_trace(
                        go.Pie(values=values, labels=labels, name=failure_type),
                        row=row, col=col
                    )
        
        fig.update_layout(height=400 * rows, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Failure rates table
        st.subheader("Tỉ lệ failure theo từng loại")
        failure_rates = []
        for failure_type in available_types:
            if failure_type in failure_analysis:
                failure_data = failure_analysis[failure_type]
                if failure_data['failure_rate'] is not None:
                    failure_rates.append({
                        'Loại Failure': failure_type,
                        'Tỉ lệ': f"{failure_data['failure_rate']:.4f}",
                        'Tổng số failure': failure_data['total_failures']
                    })
        
        if failure_rates:
            rates_df = pd.DataFrame(failure_rates)
            st.dataframe(rates_df, use_container_width=True)

def advanced_analysis_page():
    st.markdown('<h2 class="section-header">Phân tích nâng cao</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Vui lòng tải lên dữ liệu trước.")
        return
    
    data_options = ["Dữ liệu gốc"]
    if st.session_state.cleaned_data is not None:
        data_options.append("Dữ liệu đã làm sạch")
    
    data_choice = st.radio("Chọn dữ liệu để phân tích:", data_options)
    df = st.session_state.cleaned_data if data_choice == "Dữ liệu đã làm sạch" else st.session_state.data
    
    analysis_type = st.selectbox(
        "Chọn loại phân tích:",
        ["Phân tích Outlier chi tiết", "Phân tích mối quan hệ biến", "Thống kê mô tả nâng cao"]
    )
    
    if analysis_type == "Phân tích Outlier chi tiết":
        st.subheader("Phân tích Outlier chi tiết")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove binary columns
        numeric_cols = [col for col in numeric_cols if df[col].nunique() > 2]
        
        if numeric_cols:
            selected_col = st.selectbox("Chọn cột để phân tích outlier:", numeric_cols)
            
            if selected_col:
                col_data = df[selected_col].dropna()
                
                # Calculate outlier bounds
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tổng số outlier", len(outliers))
                with col2:
                    st.metric("Tỉ lệ outlier", f"{len(outliers)/len(col_data)*100:.2f}%")
                with col3:
                    st.metric("Giá trị bình thường", len(col_data) - len(outliers))
                
                # Outlier visualization
                fig = go.Figure()
                
                # Box plot
                fig.add_trace(go.Box(y=col_data, name="Box Plot", boxpoints="outliers"))
                
                fig.update_layout(
                    title=f"Outlier Analysis - {selected_col}",
                    yaxis_title=selected_col
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show outlier values
                if len(outliers) > 0:
                    st.subheader("Giá trị outlier:")
                    outlier_df = pd.DataFrame({
                        'Index': outliers.index,
                        'Value': outliers.values
                    })
                    st.dataframe(outlier_df.head(20))
    
    elif analysis_type == "Phân tích mối quan hệ biến":
        st.subheader("Phân tích mối quan hệ giữa các biến")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            col1 = st.selectbox("Chọn biến X:", numeric_cols)
            col2 = st.selectbox("Chọn biến Y:", [c for c in numeric_cols if c != col1])
            
            if col1 and col2:
                # Scatter plot with correlation
                corr_coef = df[col1].corr(df[col2])
                
                fig = px.scatter(
                    df, x=col1, y=col2,
                    title=f"Mối quan hệ giữa {col1} và {col2}<br>Correlation: {corr_coef:.3f}",
                    trendline="ols"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional statistics
                st.subheader("Thống kê mối quan hệ:")
                st.write(f"**Hệ số tương quan Pearson:** {corr_coef:.4f}")
                
                if abs(corr_coef) > 0.7:
                    st.success("🔵 Tương quan mạnh")
                elif abs(corr_coef) > 0.5:
                    st.info("🔵 Tương quan vừa")
                elif abs(corr_coef) > 0.3:
                    st.warning("🟡 Tương quan yếu")
                else:
                    st.error("🔴 Không có tương quan đáng kể")
    
    elif analysis_type == "Thống kê mô tả nâng cao":
        st.subheader("Thống kê mô tả nâng cao")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_cols = st.multiselect(
                "Chọn các cột để phân tích:",
                numeric_cols,
                default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
            )
            
            if selected_cols:
                # Advanced statistics
                advanced_stats = []
                for col in selected_cols:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        stats_dict = {
                            'Column': col,
                            'Count': len(col_data),
                            'Mean': col_data.mean(),
                            'Median': col_data.median(),
                            'Std': col_data.std(),
                            'Skewness': col_data.skew(),
                            'Kurtosis': col_data.kurtosis(),
                            'Min': col_data.min(),
                            'Max': col_data.max(),
                            'Range': col_data.max() - col_data.min(),
                            'IQR': col_data.quantile(0.75) - col_data.quantile(0.25),
                            'CV': col_data.std() / col_data.mean() * 100 if col_data.mean() != 0 else 0
                        }
                        advanced_stats.append(stats_dict)
                
                if advanced_stats:
                    stats_df = pd.DataFrame(advanced_stats)
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Distribution comparison
                    st.subheader("So sánh phân phối")
                    
                    # Create normalized histograms
                    fig = go.Figure()
                    
                    for col in selected_cols:
                        col_data = df[col].dropna()
                        if len(col_data) > 0:
                            fig.add_trace(go.Histogram(
                                x=col_data,
                                name=col,
                                opacity=0.7,
                                histnorm='probability density'
                            ))
                    
                    fig.update_layout(
                        title="So sánh phân phối các biến",
                        xaxis_title="Giá trị",
                        yaxis_title="Mật độ xác suất",
                        barmode='overlay'
                    )
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()