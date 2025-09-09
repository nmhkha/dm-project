import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Tải Dữ Liệu", page_icon="📁", layout="wide")

def main():
    st.title("📁 Tải Dữ Liệu Lên")
    st.markdown("Tải lên các file CSV của bạn để bắt đầu thực hành làm sạch dữ liệu.")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Chọn các file CSV",
        type=['csv'],
        accept_multiple_files=True,
        help="Tải lên một hoặc nhiều file CSV với cấu trúc dữ liệu bảo trì dự đoán"
    )
    
    if uploaded_files:
        st.success(f"Đã tải lên {len(uploaded_files)} file")
        
        # Process each file
        for i, uploaded_file in enumerate(uploaded_files):
            with st.expander(f"📄 {uploaded_file.name}", expanded=i==0):
                try:
                    # Load the data
                    df = st.session_state.data_processor.load_csv(uploaded_file)
                    
                    if df is not None:
                        # Show basic info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Số dòng", df.shape[0])
                        with col2:
                            st.metric("Số cột", df.shape[1])
                        with col3:
                            st.metric("Bộ nhớ sử dụng", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                        
                        # Show first few rows
                        st.subheader("Xem Trước Dữ Liệu")
                        st.dataframe(df.head(), use_container_width=True)
                        
                        # Data validation
                        st.subheader("Kiểm Tra Tính Hợp Lệ")
                        validation = st.session_state.data_processor.validate_data_structure(df)
                        
                        if validation['is_valid']:
                            st.success("✅ Kiểm tra cấu trúc dữ liệu thành công!")
                        
                        if validation['messages']:
                            for message in validation['messages']:
                                st.info(f"ℹ️ {message}")
                        
                        if validation['warnings']:
                            for warning in validation['warnings']:
                                st.warning(f"⚠️ {warning}")
                        
                        # Action buttons
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button(f"Sử dụng {uploaded_file.name} làm Dữ Liệu Gốc", key=f"use_original_{i}"):
                                st.session_state.original_data = df.copy()
                                st.session_state.dirty_data = None
                                st.session_state.cleaned_data = None
                                st.success("✅ Đã tải dữ liệu làm bộ dữ liệu gốc!")
                                st.rerun()
                        
                        with col2:
                            if st.button(f"Sử dụng {uploaded_file.name} làm Dữ Liệu Bẩn", key=f"use_dirty_{i}"):
                                st.session_state.dirty_data = df.copy()
                                st.session_state.cleaned_data = None
                                st.success("✅ Đã tải dữ liệu làm bộ dữ liệu bẩn!")
                                st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Lỗi khi xử lý {uploaded_file.name}: {str(e)}")
    
    else:
        st.info("👆 Vui lòng tải lên các file CSV để bắt đầu")
        
        # Show sample data format
        with st.expander("📋 Định Dạng Dữ Liệu Dự Kiến", expanded=False):
            st.markdown("""
            Nền tảng hoạt động tốt nhất với các bộ dữ liệu bảo trì dự đoán có chứa các cột như:
            
            **Các Cột Điển Hình:**
            - **UDI**: Mã định danh duy nhất
            - **Product ID**: Mã định danh sản phẩm
            - **Type**: Loại sản phẩm (ví dụ: L, M, H)
            - **Air temperature [K]**: Nhiệt độ không khí (Kelvin)
            - **Process temperature [K]**: Nhiệt độ quy trình (Kelvin)
            - **Rotational speed [rpm]**: Tốc độ quay (vòng/phút)
            - **Torque [Nm]**: Mô-men xoắn (Newton-mét)
            - **Tool wear [min]**: Độ mài mòn dụng cụ (phút)
            - **Target**: Chỉ báo hỏng hóc máy (0/1)
            - **TWF, HDF, PWF, OSF, RNF**: Các loại hỏng hóc khác nhau
            
            **Yêu Cầu:**
            - Định dạng CSV
            - Dòng tiêu đề có tên cột
            - Kiểu dữ liệu hỗn hợp (số và phân loại)
            - Khuyến nghị ít nhất 100 dòng để thực hành hiệu quả
            """)
    
    # Current data status
    if st.session_state.original_data is not None or st.session_state.dirty_data is not None:
        st.markdown("---")
        st.subheader("📊 Trạng Thái Dữ Liệu Hiện Tại")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.original_data is not None:
                st.success("✅ Đã Tải Dữ Liệu Gốc")
                st.write(f"Kích thước: {st.session_state.original_data.shape}")
                st.write(f"Các cột: {', '.join(st.session_state.original_data.columns[:5])}...")
            else:
                st.info("📁 Chưa tải dữ liệu gốc")
        
        with col2:
            if st.session_state.dirty_data is not None:
                st.warning("🗂️ Đã Tải Dữ Liệu Bẩn")
                st.write(f"Kích thước: {st.session_state.dirty_data.shape}")
                st.write(f"Giá trị thiếu: {st.session_state.dirty_data.isnull().sum().sum()}")
            else:
                st.info("🗂️ Chưa tải dữ liệu bẩn")

if __name__ == "__main__":
    main()
