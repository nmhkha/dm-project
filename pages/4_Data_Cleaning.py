import streamlit as st
import pandas as pd
from utils.data_cleaner import DataCleaner
from utils.visualizer import Visualizer

st.set_page_config(page_title="Làm Sạch Dữ Liệu", page_icon="🧹", layout="wide")

def main():
    st.title("🧹 Làm Sạch Dữ Liệu Tương Tác")
    st.markdown("Áp dụng các kỹ thuật làm sạch dữ liệu khác nhau để cải thiện chất lượng dữ liệu.")
    
    # Check if we have data to clean
    if st.session_state.dirty_data is None:
        st.warning("⚠️ Không có dữ liệu bẩn. Vui lòng tạo dữ liệu bẩn trước.")
        if st.button("🗂️ Đi Đến Tạo Dữ Liệu Bẩn"):
            st.switch_page("pages/3_Generate_Dirty_Data.py")
        st.stop()
    
    cleaner = DataCleaner()
    visualizer = Visualizer()
    
    # Initialize cleaned data if not exists
    if st.session_state.cleaned_data is None:
        st.session_state.cleaned_data = st.session_state.dirty_data.copy()
    
    df_dirty = st.session_state.dirty_data
    df_cleaned = st.session_state.cleaned_data
    
    # Show current status
    st.subheader("📊 Trạng Thái Làm Sạch Hiện Tại")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Giá Trị Thiếu Gốc", df_dirty.isnull().sum().sum())
    with col2:
        st.metric("Giá Trị Thiếu Hiện Tại", df_cleaned.isnull().sum().sum())
        improvement = df_dirty.isnull().sum().sum() - df_cleaned.isnull().sum().sum()
        st.metric("Cải Thiện", improvement, delta=improvement)
    with col3:
        dirty_quality = st.session_state.data_processor.get_data_quality_score(df_dirty)
        cleaned_quality = st.session_state.data_processor.get_data_quality_score(df_cleaned)
        st.metric("Điểm Chất Lượng", f"{cleaned_quality['overall']:.1f}/100", 
                 delta=f"{cleaned_quality['overall'] - dirty_quality['overall']:.1f}")
    
    # Cleaning operations
    st.subheader("🛠️ Các Thao Tác Làm Sạch Dữ Liệu")
    
    # Create tabs for different cleaning operations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Giá Trị Thiếu", "Giá Trị Ngoại Lai", "Dữ Liệu Phân Loại", "Chuẩn Hóa & Mã Hóa", "Nâng Cao"
    ])
    
    with tab1:
        st.markdown("### 🕳️ Xử Lý Giá Trị Thiếu")
        
        # Show missing values summary
        missing_summary = df_cleaned.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0].index.tolist()
        
        if missing_cols:
            # Select columns to handle
            selected_cols = st.multiselect(
                "Chọn các cột để xử lý giá trị thiếu:",
                missing_cols,
                default=missing_cols
            )
            
            if selected_cols:
                # Choose strategy
                strategy = st.selectbox(
                    "Chọn chiến lược xử lý giá trị thiếu:",
                    ["mean", "median", "mode", "forward_fill", "backward_fill", 
                     "interpolate", "knn", "constant", "custom", "drop_rows", "drop_columns"]
                )
                
                # Additional parameters for some strategies
                custom_value = None
                if strategy == "custom":
                    custom_value = st.text_input("Enter custom value:", value="Unknown")
                elif strategy == "constant":
                    custom_value = st.text_input("Enter constant value:", value="Missing")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🧹 Áp Dụng Xử Lý Giá Trị Thiếu", type="primary"):
                        try:
                            df_cleaned = cleaner.handle_missing_values(
                                df_cleaned, strategy=strategy, columns=selected_cols, 
                                custom_value=custom_value
                            )
                            st.session_state.cleaned_data = df_cleaned
                            st.success(f"✅ Applied {strategy} strategy to {len(selected_cols)} columns")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error applying strategy: {str(e)}")
                
                with col2:
                    if st.button("👁️ Xem Trước Thay Đổi"):
                        try:
                            preview_df = cleaner.handle_missing_values(
                                df_cleaned, strategy=strategy, columns=selected_cols,
                                custom_value=custom_value
                            )
                            
                            # Show before/after comparison
                            st.markdown("**Before:**")
                            st.write(f"Missing values: {df_cleaned[selected_cols].isnull().sum().sum()}")
                            st.markdown("**After:**")
                            st.write(f"Missing values: {preview_df[selected_cols].isnull().sum().sum()}")
                            
                        except Exception as e:
                            st.error(f"❌ Error previewing: {str(e)}")
            
            # Visualize missing values
            if len(missing_cols) > 0:
                st.markdown("### 📊 Missing Values Visualization")
                fig = visualizer.plot_missing_values_summary(df_cleaned)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("🎉 Không phát hiện giá trị thiếu nào trong bộ dữ liệu hiện tại!")
    
    with tab2:
        st.markdown("### 📈 Handle Outliers")
        
        numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_cols:
            # Outlier detection first
            outlier_method = st.selectbox(
                "Select outlier detection method:",
                ["iqr", "zscore", "isolation_forest"]
            )
            
            outliers = st.session_state.data_processor.detect_outliers(
                df_cleaned, method=outlier_method
            )
            
            # Show outlier summary
            outlier_data = []
            for col, info in outliers.items():
                if info['count'] > 0:
                    outlier_data.append({
                        'Column': col,
                        'Outliers': info['count'],
                        'Percentage': f"{info['percentage']:.2f}%"
                    })
            
            if outlier_data:
                outlier_df = pd.DataFrame(outlier_data)
                st.dataframe(outlier_df, use_container_width=True)
                
                # Select columns for outlier removal
                outlier_cols = [row['Column'] for row in outlier_data]
                selected_outlier_cols = st.multiselect(
                    "Select columns to remove outliers:",
                    outlier_cols,
                    default=outlier_cols
                )
                
                if selected_outlier_cols:
                    threshold = st.slider(
                        "Threshold (for IQR/Z-score methods):",
                        min_value=1.0,
                        max_value=3.0,
                        value=1.5,
                        step=0.1
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("🧹 Remove Outliers", type="primary"):
                            try:
                                df_cleaned = cleaner.remove_outliers(
                                    df_cleaned, method=outlier_method, 
                                    columns=selected_outlier_cols, threshold=threshold
                                )
                                st.session_state.cleaned_data = df_cleaned
                                st.success(f"✅ Removed outliers from {len(selected_outlier_cols)} columns")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error removing outliers: {str(e)}")
                    
                    with col2:
                        if st.button("👁️ Preview Outlier Removal"):
                            try:
                                preview_df = cleaner.remove_outliers(
                                    df_cleaned, method=outlier_method,
                                    columns=selected_outlier_cols, threshold=threshold
                                )
                                st.write(f"Rows before: {len(df_cleaned)}")
                                st.write(f"Rows after: {len(preview_df)}")
                                st.write(f"Rows removed: {len(df_cleaned) - len(preview_df)}")
                            except Exception as e:
                                st.error(f"❌ Error previewing: {str(e)}")
                
                # Visualize outliers
                st.markdown("### 📊 Outliers Visualization")
                fig = visualizer.plot_outliers_boxplot(df_cleaned, selected_outlier_cols if 'selected_outlier_cols' in locals() else outlier_cols[:3])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("🎉 No outliers detected in the current dataset!")
        else:
            st.info("No numeric columns available for outlier detection.")
    
    with tab3:
        st.markdown("### 🔤 Standardize Categorical Data")
        
        categorical_cols = df_cleaned.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            selected_cat_cols = st.multiselect(
                "Select categorical columns to standardize:",
                categorical_cols,
                default=categorical_cols
            )
            
            if selected_cat_cols:
                # Show current categorical data issues
                for col in selected_cat_cols:
                    with st.expander(f"📊 Analysis for {col}"):
                        unique_values = df_cleaned[col].value_counts().head(20)
                        st.write(f"Unique values: {df_cleaned[col].nunique()}")
                        st.write("Top values:")
                        st.dataframe(unique_values, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🧹 Standardize Categorical Data", type="primary"):
                        try:
                            df_cleaned = cleaner.standardize_categorical_data(
                                df_cleaned, columns=selected_cat_cols
                            )
                            st.session_state.cleaned_data = df_cleaned
                            st.success(f"✅ Standardized {len(selected_cat_cols)} categorical columns")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error standardizing: {str(e)}")
                
                with col2:
                    if st.button("👁️ Preview Standardization"):
                        try:
                            preview_df = cleaner.standardize_categorical_data(
                                df_cleaned, columns=selected_cat_cols
                            )
                            
                            for col in selected_cat_cols[:2]:  # Show first 2 columns
                                st.write(f"**{col}** - Unique values before: {df_cleaned[col].nunique()}, after: {preview_df[col].nunique()}")
                                
                        except Exception as e:
                            st.error(f"❌ Error previewing: {str(e)}")
        else:
            st.info("No categorical columns available for standardization.")
    
    with tab4:
        st.markdown("### ⚖️ Scaling and Encoding")
        
        # Scaling section
        st.markdown("#### Numeric Feature Scaling")
        numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_cols:
            selected_scale_cols = st.multiselect(
                "Select numeric columns to scale:",
                numeric_cols,
                help="Scaling normalizes the range of features"
            )
            
            if selected_scale_cols:
                scale_method = st.selectbox(
                    "Select scaling method:",
                    ["standard", "minmax", "robust"]
                )
                
                if st.button("⚖️ Apply Scaling", type="primary"):
                    try:
                        df_cleaned = cleaner.scale_numeric_features(
                            df_cleaned, method=scale_method, columns=selected_scale_cols
                        )
                        st.session_state.cleaned_data = df_cleaned
                        st.success(f"✅ Applied {scale_method} scaling to {len(selected_scale_cols)} columns")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error scaling: {str(e)}")
        
        # Encoding section
        st.markdown("#### Categorical Variable Encoding")
        categorical_cols = df_cleaned.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            selected_encode_cols = st.multiselect(
                "Select categorical columns to encode:",
                categorical_cols,
                help="Encoding converts categorical variables to numeric"
            )
            
            if selected_encode_cols:
                encode_method = st.selectbox(
                    "Select encoding method:",
                    ["label", "onehot", "frequency"]
                )
                
                if st.button("🔢 Apply Encoding", type="primary"):
                    try:
                        df_cleaned = cleaner.encode_categorical_variables(
                            df_cleaned, method=encode_method, columns=selected_encode_cols
                        )
                        st.session_state.cleaned_data = df_cleaned
                        st.success(f"✅ Applied {encode_method} encoding to {len(selected_encode_cols)} columns")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error encoding: {str(e)}")
    
    with tab5:
        st.markdown("### 🚀 Advanced Operations")
        
        # Remove duplicates
        st.markdown("#### Remove Duplicate Rows")
        duplicate_count = df_cleaned.duplicated().sum()
        st.write(f"Duplicate rows found: {duplicate_count}")
        
        if duplicate_count > 0:
            keep_option = st.selectbox("Keep which duplicate:", ["first", "last", "none"])
            
            if st.button("🗑️ Remove Duplicates", type="primary"):
                try:
                    df_cleaned = cleaner.remove_duplicates(df_cleaned, keep=keep_option)
                    st.session_state.cleaned_data = df_cleaned
                    st.success(f"✅ Removed {duplicate_count} duplicate rows")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error removing duplicates: {str(e)}")
        else:
            st.success("🎉 No duplicate rows found!")
        
        # Data type detection and fixing
        st.markdown("#### Automatic Data Type Detection")
        if st.button("🔍 Detect and Fix Data Types"):
            try:
                df_cleaned = cleaner.detect_and_fix_data_types(df_cleaned)
                st.session_state.cleaned_data = df_cleaned
                st.success("✅ Automatically detected and fixed data types")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error fixing data types: {str(e)}")
        
        # Reset to original dirty data
        st.markdown("#### Reset Operations")
        if st.button("🔄 Reset to Original Dirty Data", type="secondary"):
            st.session_state.cleaned_data = st.session_state.dirty_data.copy()
            st.success("✅ Reset to original dirty data")
            st.rerun()
    
    # Show cleaning summary
    st.markdown("---")
    st.subheader("📋 Cleaning Summary")
    
    # Generate cleaning summary
    summary = cleaner.get_cleaning_summary(df_dirty, df_cleaned)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows Removed", summary['rows_removed'])
    with col2:
        st.metric("Columns Changed", summary['columns_removed'])
    with col3:
        st.metric("Missing Values Handled", summary['missing_values_handled']['reduction'])
    with col4:
        st.metric("New Columns Added", len(summary['new_columns_added']))
    
    # Data type changes
    if summary['data_types_changed']:
        with st.expander("📊 Data Type Changes"):
            for col, change in summary['data_types_changed'].items():
                st.write(f"**{col}**: {change['from']} → {change['to']}")
    
    # Final comparison visualization
    st.subheader("📊 Before vs After Comparison")
    
    quality_dirty = st.session_state.data_processor.get_data_quality_score(df_dirty)
    quality_cleaned = st.session_state.data_processor.get_data_quality_score(df_cleaned)
    
    fig = visualizer.plot_data_quality_comparison(quality_dirty, quality_cleaned)
    st.plotly_chart(fig, use_container_width=True)
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("💾 Export Cleaned Data", use_container_width=True):
            st.switch_page("pages/5_Export_Data.py")
    
    with col2:
        if st.button("📊 View Data Overview", use_container_width=True):
            st.switch_page("pages/2_Data_Overview.py")
    
    with col3:
        if st.button("🔄 Start Over", use_container_width=True):
            st.session_state.cleaned_data = None
            st.session_state.dirty_data = None
            st.switch_page("pages/1_Data_Upload.py")

if __name__ == "__main__":
    main()
