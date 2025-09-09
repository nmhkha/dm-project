import streamlit as st
from utils.dirty_data_generator import DirtyDataGenerator
from utils.visualizer import Visualizer

st.set_page_config(page_title="Generate Dirty Data", page_icon="🗂️", layout="wide")

def main():
    st.title("🗂️ Generate Dirty Data for Practice")
    st.markdown("Create controlled 'dirty' datasets to practice data cleaning techniques.")
    
    if st.session_state.original_data is None:
        st.warning("⚠️ No original data available. Please upload clean data first.")
        st.stop()
    
    generator = DirtyDataGenerator()
    visualizer = Visualizer()
    
    df = st.session_state.original_data
    
    # Show original data info
    st.subheader("📊 Original Data Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Current Missing Values", df.isnull().sum().sum())
    
    # Configuration sections
    st.subheader("⚙️ Dirty Data Configuration")
    
    config = {}
    
    # Missing Values Configuration
    with st.expander("🕳️ Missing Values Configuration", expanded=True):
        missing_enabled = st.checkbox("Add missing values", value=True)
        config['missing_values'] = {'enabled': missing_enabled}
        
        if missing_enabled:
            col1, col2 = st.columns(2)
            
            with col1:
                missing_percentage = st.slider(
                    "Missing values percentage:",
                    min_value=0.01,
                    max_value=0.5,
                    value=0.1,
                    step=0.01,
                    format="%.2f"
                )
                config['missing_values']['percentage'] = missing_percentage
            
            with col2:
                available_cols = [col for col in df.columns 
                                if not any(pattern in col.lower() for pattern in ['id', 'udi'])]
                selected_missing_cols = st.multiselect(
                    "Select columns to add missing values:",
                    available_cols,
                    default=available_cols[:3],
                    help="Leave empty to randomly select columns"
                )
                config['missing_values']['columns'] = selected_missing_cols if selected_missing_cols else None
    
    # Outliers Configuration
    with st.expander("📈 Outliers Configuration"):
        outliers_enabled = st.checkbox("Add outliers", value=True)
        config['outliers'] = {'enabled': outliers_enabled}
        
        if outliers_enabled:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                outlier_percentage = st.slider(
                    "Outliers percentage:",
                    min_value=0.01,
                    max_value=0.2,
                    value=0.05,
                    step=0.01,
                    format="%.2f"
                )
                config['outliers']['percentage'] = outlier_percentage
            
            with col2:
                outlier_method = st.selectbox(
                    "Outlier generation method:",
                    ["extreme", "random"]
                )
                config['outliers']['method'] = outlier_method
            
            with col3:
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                selected_outlier_cols = st.multiselect(
                    "Select numeric columns:",
                    numeric_cols,
                    default=numeric_cols[:2] if numeric_cols else [],
                    help="Leave empty to use all numeric columns"
                )
                config['outliers']['columns'] = selected_outlier_cols if selected_outlier_cols else None
    
    # Categorical Errors Configuration
    with st.expander("🔤 Categorical Data Errors Configuration"):
        cat_errors_enabled = st.checkbox("Add categorical errors", value=True)
        config['categorical_errors'] = {'enabled': cat_errors_enabled}
        
        if cat_errors_enabled:
            col1, col2 = st.columns(2)
            
            with col1:
                cat_error_percentage = st.slider(
                    "Categorical errors percentage:",
                    min_value=0.01,
                    max_value=0.3,
                    value=0.1,
                    step=0.01,
                    format="%.2f"
                )
                config['categorical_errors']['percentage'] = cat_error_percentage
            
            with col2:
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                selected_cat_cols = st.multiselect(
                    "Select categorical columns:",
                    categorical_cols,
                    default=categorical_cols[:2] if categorical_cols else [],
                    help="Leave empty to use all categorical columns"
                )
                config['categorical_errors']['columns'] = selected_cat_cols if selected_cat_cols else None
            
            # Error types preview
            st.info("""
            **Error types that will be introduced:**
            - Typos and misspellings
            - Case variations (upper/lower case)
            - Extra spaces (leading/trailing)
            - Special characters
            - Invalid category values
            """)
    
    # Generate button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🗂️ Generate Dirty Data", type="primary", use_container_width=True):
            with st.spinner("Generating dirty data..."):
                try:
                    dirty_df = generator.generate_dirty_data(df, config)
                    st.session_state.dirty_data = dirty_df
                    
                    # Generate summary
                    summary = generator.get_corruption_summary(df, dirty_df)
                    
                    st.success("✅ Dirty data generated successfully!")
                    
                    # Show generation summary
                    st.subheader("📋 Generation Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Missing Values Added", summary['missing_values']['added'])
                    with col2:
                        st.metric("Shape Preserved", "✅" if summary['shape_preserved'] else "❌")
                    with col3:
                        st.metric("Columns Modified", len([col for col, info in summary['data_changes'].items() 
                                                         if info['missing_added'] > 0 or info['new_values'] > 0]))
                    
                    # Detailed changes per column
                    with st.expander("📊 Detailed Changes per Column"):
                        changes_data = []
                        for col, info in summary['data_changes'].items():
                            changes_data.append({
                                'Column': col,
                                'Missing Added': info['missing_added'],
                                'Values Modified': info['new_values'],
                                'Data Type': info['data_type']
                            })
                        
                        if changes_data:
                            import pandas as pd
                            changes_df = pd.DataFrame(changes_data)
                            st.dataframe(changes_df, use_container_width=True)
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error generating dirty data: {str(e)}")
    
    # Show comparison if dirty data exists
    if st.session_state.dirty_data is not None:
        st.markdown("---")
        st.subheader("🔍 Before vs After Comparison")
        
        dirty_df = st.session_state.dirty_data
        
        # Metrics comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Data**")
            st.metric("Missing Values", df.isnull().sum().sum())
            st.metric("Rows", df.shape[0])
            
            # Data quality scores
            original_quality = st.session_state.data_processor.get_data_quality_score(df)
            st.metric("Quality Score", f"{original_quality['overall']:.1f}/100")
        
        with col2:
            st.markdown("**Dirty Data**")
            st.metric("Missing Values", dirty_df.isnull().sum().sum())
            st.metric("Rows", dirty_df.shape[0])
            
            # Data quality scores
            dirty_quality = st.session_state.data_processor.get_data_quality_score(dirty_df)
            st.metric("Quality Score", f"{dirty_quality['overall']:.1f}/100")
        
        # Visual comparison
        comparison_tabs = st.tabs(["Missing Values", "Quality Scores", "Data Preview"])
        
        with comparison_tabs[0]:
            fig = visualizer.plot_before_after_comparison(df, dirty_df, 'missing_values')
            st.plotly_chart(fig, use_container_width=True)
        
        with comparison_tabs[1]:
            fig = visualizer.plot_data_quality_comparison(original_quality, dirty_quality)
            st.plotly_chart(fig, use_container_width=True)
        
        with comparison_tabs[2]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Data Sample**")
                st.dataframe(df.head(), use_container_width=True)
            
            with col2:
                st.markdown("**Dirty Data Sample**")
                st.dataframe(dirty_df.head(), use_container_width=True)
        
        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🗑️ Xóa Dữ Liệu Bẩn", use_container_width=True):
                st.session_state.dirty_data = None
                st.rerun()
        
        with col2:
            if st.button("🔄 Tạo Lại", use_container_width=True):
                st.rerun()
        
        with col3:
            if st.button("➡️ Tiến Đến Làm Sạch", use_container_width=True):
                st.switch_page("pages/4_Data_Cleaning.py")

if __name__ == "__main__":
    main()
