import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.visualizer import Visualizer

# Configure page
st.set_page_config(
    page_title="Nền Tảng Thực Hành Làm Sạch Dữ Liệu",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()

if 'original_data' not in st.session_state:
    st.session_state.original_data = None

if 'dirty_data' not in st.session_state:
    st.session_state.dirty_data = None

if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None

def main():
    st.title("🔧 Nền Tảng Thực Hành Làm Sạch Dữ Liệu")
    st.markdown("### Xử Lý & Làm Sạch Dữ Liệu Bảo Trì Dự Đoán")
    
    st.markdown("""
    Chào mừng đến với Nền Tảng Thực Hành Làm Sạch Dữ Liệu! Công cụ này giúp bạn:
    
    - 📁 **Tải lên** file CSV với dữ liệu bảo trì dự đoán
    - 📊 **Phân tích** cấu trúc và chất lượng dữ liệu
    - 🗂️ **Tạo ra** bộ dữ liệu "bẩn" có kiểm soát để thực hành
    - 🧹 **Làm sạch** dữ liệu bằng các kỹ thuật tiền xử lý
    - 📈 **Trực quan hóa** cải thiện chất lượng dữ liệu
    - 💾 **Xuất** bộ dữ liệu đã làm sạch
    
    Sử dụng menu điều hướng bên trái để bắt đầu!
    """)
    
    # Sidebar navigation info
    with st.sidebar:
        st.markdown("### Hướng Dẫn Điều Hướng")
        st.markdown("""
        1. **Tải Dữ Liệu**: Bắt đầu bằng việc tải lên file CSV
        2. **Tổng Quan Dữ Liệu**: Khám phá cấu trúc bộ dữ liệu
        3. **Tạo Dữ Liệu Bẩn**: Tạo bộ dữ liệu để thực hành
        4. **Làm Sạch Dữ Liệu**: Áp dụng các kỹ thuật làm sạch
        5. **Xuất Dữ Liệu**: Tải xuống dữ liệu đã làm sạch
        """)
        
        # Show current data status
        st.markdown("### Trạng Thái Hiện Tại")
        if st.session_state.original_data is not None:
            st.success(f"✅ Dữ Liệu Gốc: {st.session_state.original_data.shape[0]} dòng")
        else:
            st.info("📁 Chưa tải lên dữ liệu")
            
        if st.session_state.dirty_data is not None:
            st.warning(f"🗂️ Dữ Liệu Bẩn: {st.session_state.dirty_data.shape[0]} dòng")
        else:
            st.info("🗂️ Chưa tạo dữ liệu bẩn")
            
        if st.session_state.cleaned_data is not None:
            st.success(f"🧹 Dữ Liệu Đã Làm Sạch: {st.session_state.cleaned_data.shape[0]} dòng")
        else:
            st.info("🧹 Chưa có dữ liệu đã làm sạch")

    # Sample data information
    with st.expander("📋 Về Bộ Dữ Liệu Bảo Trì Dự Đoán AI4I 2020"):
        st.markdown("""
        Nền tảng này được thiết kế để làm việc với các bộ dữ liệu bảo trì dự đoán, đặc biệt là định dạng bộ dữ liệu AI4I 2020.
        
        **Các cột dự kiến:**
        - UDI: Mã định danh duy nhất
        - Product ID: Mã định danh sản phẩm
        - Type: Loại sản phẩm (L, M, H)
        - Air temperature [K]: Nhiệt độ không khí (Kelvin)
        - Process temperature [K]: Nhiệt độ quy trình (Kelvin)
        - Rotational speed [rpm]: Tốc độ quay (vòng/phút)
        - Torque [Nm]: Mô-men xoắn (Newton-mét)
        - Tool wear [min]: Độ mài mòn dụng cụ (phút)
        - Target: Chỉ báo hỏng hóc máy (0/1)
        - TWF, HDF, PWF, OSF, RNF: Các loại hỏng hóc khác nhau
        
        Nền tảng tự động phát hiện cấu trúc tương tự và thích ứng phù hợp.
        """)

if __name__ == "__main__":
    main()
