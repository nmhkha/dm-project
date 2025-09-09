import streamlit as st
import pandas as pd
from utils.visualizer import Visualizer

st.set_page_config(page_title="Tổng Quan Dữ Liệu", page_icon="📊", layout="wide")

def main():
    st.title("📊 Tổng Quan Dữ Liệu & Phân Tích Khám Phá")
    
    visualizer = Visualizer()
    
    # Data selection
    data_options = []
    if st.session_state.original_data is not None:
        data_options.append("Dữ Liệu Gốc")
    if st.session_state.dirty_data is not None:
        data_options.append("Dữ Liệu Bẩn")
    if st.session_state.cleaned_data is not None:
        data_options.append("Dữ Liệu Đã Làm Sạch")
    
    if not data_options:
        st.warning("⚠️ Không có dữ liệu. Vui lòng tải dữ liệu trước.")
        st.stop()
    
    selected_data = st.selectbox("Chọn bộ dữ liệu để phân tích:", data_options)
    
    # Get the selected dataset
    if selected_data == "Dữ Liệu Gốc":
        df = st.session_state.original_data
    elif selected_data == "Dữ Liệu Bẩn":
        df = st.session_state.dirty_data
    else:
        df = st.session_state.cleaned_data
    
    if df is None:
        st.error("Dữ liệu được chọn không khả dụng.")
        st.stop()
    
    # Data overview
    st.subheader("📈 Tóm Tắt Bộ Dữ Liệu")
    
    overview = st.session_state.data_processor.get_data_overview(df)
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tổng Số Dòng", overview['shape'][0])
    with col2:
        st.metric("Tổng Số Cột", overview['shape'][1])
    with col3:
        st.metric("Giá Trị Thiếu", sum(overview['missing_values'].values()))
    with col4:
        st.metric("Sử Dụng Bộ Nhớ", f"{overview['memory_usage']:.2f} MB")
    
    # Data quality score
    quality_scores = st.session_state.data_processor.get_data_quality_score(df)
    
    st.subheader("🎯 Điểm Chất Lượng Dữ Liệu")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Giá Trị Thiếu", f"{quality_scores['missing_values']:.1f}/100")
    with col2:
        st.metric("Dữ Liệu Trùng Lặp", f"{quality_scores['duplicates']:.1f}/100")
    with col3:
        st.metric("Giá Trị Ngoại Lai", f"{quality_scores['outliers']:.1f}/100")
    with col4:
        st.metric("Điểm Tổng Thể", f"{quality_scores['overall']:.1f}/100")
    
    # Column information
    st.subheader("📋 Thông Tin Cột")
    
    col_info = pd.DataFrame({
        'Cột': overview['columns'],
        'Kiểu Dữ Liệu': [overview['dtypes'][col] for col in overview['columns']],
        'Số Thiếu': [overview['missing_values'][col] for col in overview['columns']],
        'Tỉ Lệ Thiếu %': [f"{overview['missing_percentage'][col]:.1f}%" for col in overview['columns']]
    })
    
    st.dataframe(col_info, use_container_width=True)
    
    # Data preview
    st.subheader("👀 Xem Trước Dữ Liệu")
    
    preview_options = st.radio(
        "Chọn loại xem trước:",
        ["10 dòng đầu", "10 dòng cuối", "10 dòng ngẫu nhiên", "Mẫu có giá trị thiếu"],
        horizontal=True
    )
    
    if preview_options == "10 dòng đầu":
        st.dataframe(df.head(10), use_container_width=True)
    elif preview_options == "10 dòng cuối":
        st.dataframe(df.tail(10), use_container_width=True)
    elif preview_options == "10 dòng ngẫu nhiên":
        st.dataframe(df.sample(min(10, len(df))), use_container_width=True)
    else:
        # Show rows with missing values
        missing_rows = df[df.isnull().any(axis=1)]
        if len(missing_rows) > 0:
            st.dataframe(missing_rows.head(10), use_container_width=True)
        else:
            st.info("Không tìm thấy dòng nào có giá trị thiếu.")
    
    # Visualizations
    st.subheader("📊 Trực Quan Hóa Dữ Liệu")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Giá Trị Thiếu", "Phân Phối", "Tương Quan", "Phân Loại"])
    
    with tab1:
        if df.isnull().sum().sum() > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(
                    visualizer.plot_missing_values_summary(df),
                    use_container_width=True
                )
            
            with col2:
                st.plotly_chart(
                    visualizer.plot_missing_values_heatmap(df),
                    use_container_width=True
                )
        else:
            st.success("🎉 Không phát hiện giá trị thiếu nào trong bộ dữ liệu này!")
    
    with tab2:
        numeric_cols = overview['numeric_columns']
        if numeric_cols:
            selected_cols = st.multiselect(
                "Chọn các cột để trực quan hóa:",
                numeric_cols,
                default=numeric_cols[:3]
            )
            
            if selected_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(
                        visualizer.plot_data_distribution(df, selected_cols),
                        use_container_width=True
                    )
                
                with col2:
                    st.plotly_chart(
                        visualizer.plot_outliers_boxplot(df, selected_cols),
                        use_container_width=True
                    )
        else:
            st.info("Không có cột số nào để phân tích phân phối.")
    
    with tab3:
        if len(overview['numeric_columns']) > 1:
            correlation_plot = visualizer.plot_correlation_heatmap(df)
            if correlation_plot:
                st.plotly_chart(correlation_plot, use_container_width=True)
            
            # Show correlation table
            corr_matrix = df[overview['numeric_columns']].corr()
            st.subheader("Ma Trận Tương Quan")
            st.dataframe(corr_matrix, use_container_width=True)
        else:
            st.info("Cần ít nhất 2 cột số để phân tích tương quan.")
    
    with tab4:
        categorical_cols = overview['categorical_columns']
        if categorical_cols:
            selected_cat_cols = st.multiselect(
                "Chọn các cột phân loại để trực quan hóa:",
                categorical_cols,
                default=categorical_cols[:2]
            )
            
            if selected_cat_cols:
                st.plotly_chart(
                    visualizer.plot_categorical_distribution(df, selected_cat_cols),
                    use_container_width=True
                )
                
                # Show value counts
                for col in selected_cat_cols:
                    with st.expander(f"Số lượng giá trị cho {col}"):
                        value_counts = df[col].value_counts()
                        st.dataframe(value_counts.reset_index())
        else:
            st.info("Không có cột phân loại nào để phân tích.")
    
    # Descriptive statistics
    st.subheader("📈 Thống Kê Mô Tả")
    
    stat_tabs = st.tabs(["Số", "Phân Loại"])
    
    with stat_tabs[0]:
        if overview['numeric_columns']:
            st.dataframe(df[overview['numeric_columns']].describe(), use_container_width=True)
        else:
            st.info("Không có cột số nào.")
    
    with stat_tabs[1]:
        if overview['categorical_columns']:
            cat_stats = df[overview['categorical_columns']].describe(include='all')
            st.dataframe(cat_stats, use_container_width=True)
        else:
            st.info("Không có cột phân loại nào.")
    
    # Outlier detection
    if overview['numeric_columns']:
        st.subheader("🔍 Phát Hiện Giá Trị Ngoại Lai")
        
        outlier_method = st.selectbox("Chọn phương pháp phát hiện ngoại lai:", ["IQR", "Z-Score"])
        outliers = st.session_state.data_processor.detect_outliers(
            df, method=outlier_method.lower()
        )
        
        outlier_df = pd.DataFrame([
            {
                'Cột': col,
                'Số Ngoại Lai': info['count'],
                'Tỉ Lệ Ngoại Lai': f"{info['percentage']:.2f}%"
            }
            for col, info in outliers.items()
        ])
        
        st.dataframe(outlier_df, use_container_width=True)

if __name__ == "__main__":
    main()
