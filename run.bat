@echo off
REM Tạo môi trường ảo
python -m venv venv

REM Kích hoạt môi trường ảo
call venv\Scripts\activate

REM Cài đặt thư viện
pip install -r requirements.txt

REM Chạy ứng dụng
streamlit run app.py
