import streamlit as st
import pandas as pd
import io
from datetime import datetime

st.set_page_config(page_title="Xuất Dữ Liệu", page_icon="💾", layout="wide")

def main():
    st.title("💾 Xuất Dữ Liệu Đã Làm Sạch")
    st.markdown("Tải xuống các bộ dữ liệu đã làm sạch và báo cáo phân tích.")
    
    # Check available data
    available_data = {}
    if st.session_state.original_data is not None:
        available_data['Dữ Liệu Gốc'] = st.session_state.original_data
    if st.session_state.dirty_data is not None:
        available_data['Dữ Liệu Bẩn'] = st.session_state.dirty_data
    if st.session_state.cleaned_data is not None:
        available_data['Dữ Liệu Đã Làm Sạch'] = st.session_state.cleaned_data
    
    if not available_data:
        st.warning("⚠️ Không có dữ liệu để xuất. Vui lòng xử lý dữ liệu trước.")
        st.stop()
    
    # Export options
    st.subheader("📁 Các Bộ Dữ Liệu Khả Dụng")
    
    for name, df in available_data.items():
        with st.expander(f"📊 {name} ({df.shape[0]} rows × {df.shape[1]} cols)", expanded=True):
            
            # Show data preview
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(df.head(), use_container_width=True)
            
            with col2:
                # Data summary
                st.markdown("**Tóm tắt:**")
                st.write(f"Số dòng: {df.shape[0]}")
                st.write(f"Số cột: {df.shape[1]}")
                st.write(f"Giá trị thiếu: {df.isnull().sum().sum()}")
                st.write(f"Sử dụng bộ nhớ: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                # Data quality score
                quality_score = st.session_state.data_processor.get_data_quality_score(df)
                st.metric("Điểm Chất Lượng", f"{quality_score['overall']:.1f}/100")
            
            # Export options
            st.markdown("**Tùy Chọn Xuất:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV export
                csv_data = df.to_csv(index=False)
                filename = f"{name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                st.download_button(
                    label="📄 Tải CSV",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Excel export
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='Data', index=False)
                    
                    # Add summary sheet
                    summary_data = {
                        'Metric': ['Rows', 'Columns', 'Missing Values', 'Data Types', 'Memory (MB)'],
                        'Value': [
                            df.shape[0],
                            df.shape[1],
                            df.isnull().sum().sum(),
                            len(df.dtypes.unique()),
                            round(df.memory_usage(deep=True).sum() / 1024**2, 2)
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                excel_filename = f"{name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                
                st.download_button(
                    label="📊 Tải Excel",
                    data=buffer.getvalue(),
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with col3:
                # JSON export
                json_data = df.to_json(orient='records', indent=2)
                json_filename = f"{name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                st.download_button(
                    label="🔗 Tải JSON",
                    data=json_data,
                    file_name=json_filename,
                    mime="application/json",
                    use_container_width=True
                )
    
    # Generate comprehensive report
    st.markdown("---")
    st.subheader("📋 Báo Cáo Xử Lý Dữ Liệu")
    
    if st.button("📊 Tạo Báo Cáo Đầy Đủ", type="primary"):
        with st.spinner("Đang tạo báo cáo..."):
            # Create comprehensive report
            report_content = generate_comprehensive_report(available_data)
            
            # Display report
            st.markdown(report_content)
            
            # Download report
            report_filename = f"data_processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            
            st.download_button(
                label="📄 Tải Báo Cáo (Markdown)",
                data=report_content,
                file_name=report_filename,
                mime="text/markdown"
            )
    
    # Comparison analysis
    if len(available_data) > 1:
        st.markdown("---")
        st.subheader("🔍 Data Comparison Analysis")
        
        # Select datasets to compare
        dataset_names = list(available_data.keys())
        
        col1, col2 = st.columns(2)
        with col1:
            dataset1 = st.selectbox("Select first dataset:", dataset_names, index=0)
        with col2:
            dataset2 = st.selectbox("Select second dataset:", dataset_names, index=1 if len(dataset_names) > 1 else 0)
        
        if dataset1 != dataset2:
            comparison = st.session_state.data_processor.compare_datasets(
                available_data[dataset1], available_data[dataset2]
            )
            
            # Display comparison
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Row Difference", comparison['shape_change']['rows_diff'])
            with col2:
                st.metric("Column Difference", comparison['shape_change']['cols_diff'])
            with col3:
                quality1 = comparison['quality_scores']['original']['overall']
                quality2 = comparison['quality_scores']['modified']['overall']
                st.metric("Quality Improvement", f"{quality2 - quality1:.1f}", delta=f"{quality2 - quality1:.1f}")
            
            # Missing values comparison
            if comparison['missing_values_change']:
                st.markdown("**Missing Values Changes:**")
                missing_df = pd.DataFrame([
                    {
                        'Column': col,
                        f'{dataset1} Missing': change['original'],
                        f'{dataset2} Missing': change['modified'],
                        'Change': change['change']
                    }
                    for col, change in comparison['missing_values_change'].items()
                    if change['change'] != 0
                ])
                
                if not missing_df.empty:
                    st.dataframe(missing_df, use_container_width=True)
    
    # Export all data
    if len(available_data) > 1:
        st.markdown("---")
        st.subheader("📦 Export All Data")
        
        if st.button("📁 Download All Datasets (ZIP)", type="primary"):
            with st.spinner("Creating ZIP file..."):
                # Create ZIP file with all datasets
                zip_buffer = io.BytesIO()
                
                import zipfile
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for name, df in available_data.items():
                        # Add CSV
                        csv_data = df.to_csv(index=False)
                        csv_filename = f"{name.lower().replace(' ', '_')}.csv"
                        zip_file.writestr(csv_filename, csv_data)
                        
                        # Add summary
                        summary = generate_dataset_summary(name, df)
                        summary_filename = f"{name.lower().replace(' ', '_')}_summary.txt"
                        zip_file.writestr(summary_filename, summary)
                    
                    # Add comprehensive report
                    report_content = generate_comprehensive_report(available_data)
                    zip_file.writestr("comprehensive_report.md", report_content)
                
                zip_filename = f"data_cleaning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                
                st.download_button(
                    label="📦 Download ZIP Package",
                    data=zip_buffer.getvalue(),
                    file_name=zip_filename,
                    mime="application/zip"
                )
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔄 Process More Data", use_container_width=True):
            st.switch_page("pages/1_Data_Upload.py")
    
    with col2:
        if st.button("📊 View Overview", use_container_width=True):
            st.switch_page("pages/2_Data_Overview.py")
    
    with col3:
        if st.button("🧹 Clean More Data", use_container_width=True):
            st.switch_page("pages/4_Data_Cleaning.py")

def generate_dataset_summary(name: str, df: pd.DataFrame) -> str:
    """Generate a text summary for a dataset."""
    quality_score = st.session_state.data_processor.get_data_quality_score(df)
    
    summary = f"""
Dataset: {name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

BASIC INFORMATION:
- Rows: {df.shape[0]}
- Columns: {df.shape[1]}
- Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

DATA QUALITY:
- Overall Score: {quality_score['overall']:.1f}/100
- Missing Values Score: {quality_score['missing_values']:.1f}/100
- Duplicates Score: {quality_score['duplicates']:.1f}/100
- Outliers Score: {quality_score['outliers']:.1f}/100

MISSING VALUES:
- Total Missing: {df.isnull().sum().sum()}
- Percentage: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%

COLUMN DETAILS:
"""
    
    for col in df.columns:
        col_type = str(df[col].dtype)
        missing_count = df[col].isnull().sum()
        unique_count = df[col].nunique()
        
        summary += f"- {col}: {col_type}, Missing: {missing_count}, Unique: {unique_count}\n"
    
    return summary

def generate_comprehensive_report(available_data: dict) -> str:
    """Generate a comprehensive markdown report."""
    
    report = f"""# Data Cleaning Practice Platform - Processing Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This report summarizes the data processing activities performed using the Data Cleaning Practice Platform.

### Datasets Processed

"""
    
    for name, df in available_data.items():
        quality_score = st.session_state.data_processor.get_data_quality_score(df)
        
        report += f"""
#### {name}

- **Dimensions**: {df.shape[0]} rows × {df.shape[1]} columns
- **Missing Values**: {df.isnull().sum().sum()} ({(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%)
- **Data Quality Score**: {quality_score['overall']:.1f}/100
- **Memory Usage**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

**Column Information**:
"""
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            missing_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            report += f"- `{col}`: {col_type} (Missing: {missing_count}, Unique: {unique_count})\n"
    
    # Add comparison if multiple datasets
    if len(available_data) > 1:
        report += "\n## Data Transformation Summary\n\n"
        
        if 'Original Data' in available_data and 'Dirty Data' in available_data:
            original = available_data['Original Data']
            dirty = available_data['Dirty Data']
            
            report += "### Original → Dirty Data\n"
            report += f"- Missing values added: {dirty.isnull().sum().sum() - original.isnull().sum().sum()}\n"
            
        if 'Dirty Data' in available_data and 'Cleaned Data' in available_data:
            dirty = available_data['Dirty Data']
            cleaned = available_data['Cleaned Data']
            
            report += "### Dirty → Cleaned Data\n"
            report += f"- Missing values removed: {dirty.isnull().sum().sum() - cleaned.isnull().sum().sum()}\n"
            report += f"- Rows removed: {dirty.shape[0] - cleaned.shape[0]}\n"
            report += f"- Columns modified: {cleaned.shape[1] - dirty.shape[1]}\n"
    
    report += """
## Data Quality Metrics

The following metrics were used to assess data quality:

1. **Missing Values Score**: Percentage of complete data
2. **Duplicates Score**: Percentage of unique records  
3. **Outliers Score**: Assessment of extreme values
4. **Overall Score**: Weighted average of all metrics

## Recommendations

Based on the analysis:

1. **Data Collection**: Implement validation at source to reduce missing values
2. **Data Processing**: Regular cleaning schedules to maintain quality
3. **Monitoring**: Continuous quality assessment during data pipeline
4. **Documentation**: Maintain data dictionaries and processing logs

---

*Generated by Data Cleaning Practice Platform*
"""
    
    return report

if __name__ == "__main__":
    main()
