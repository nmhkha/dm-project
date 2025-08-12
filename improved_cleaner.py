"""
Improved Data Cleaner based on the uploaded notebooks
Phiên bản cải tiến từ cleandata_1755016864578.ipynb
"""

import pandas as pd
import re
import numpy as np
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ImprovedDataCleaner:
    def __init__(self):
        self.cleaning_log = []
    
    def clean_text(self, val):
        """Làm sạch dữ liệu văn bản theo cách đơn giản nhất"""
        if pd.isna(val):
            return val
        
        val = str(val).strip()
        
        # Bỏ ký tự đặc biệt
        val = re.sub(r"[!@#$%^&*?]+", "", val)
        
        # Bỏ khoảng trắng thừa
        val = re.sub(r"\s+", " ", val)
        
        return val
    
    def clean_price(self, val):
        """Trích xuất số từ cột giá - xử lý nhiều định dạng khác nhau"""
        if pd.isna(val):
            return val
        
        val_str = str(val).strip()
        
        # Xử lý các trường hợp đặc biệt
        if val_str.lower() in ['null', 'unknown price', '#error#', 'error_price']:
            return None
        
        # Loại bỏ các ký tự đặc biệt và chỉ giữ số, dấu chấm, dấu phẩy
        val_clean = re.sub(r"[^0-9.,]", "", val_str)
        
        # Xử lý định dạng số với dấu phẩy (VD: 150,000)
        if ',' in val_clean:
            val_clean = val_clean.replace(',', '')
        
        # Xử lý định dạng với dấu chấm (VD: 150.000)
        if '.' in val_clean and len(val_clean.split('.')[-1]) == 3:
            val_clean = val_clean.replace('.', '')
        
        try:
            return float(val_clean) if val_clean else None
        except:
            return None
    
    def clean_date(self, val):
        """Chuẩn hóa định dạng ngày - xử lý nhiều định dạng khác nhau"""
        if pd.isna(val):
            return val
        
        val = str(val).strip()
        
        # Xử lý các trường hợp đặc biệt
        if val.lower() in ['invalid date', 'invalid_date', '??/??/????', 'null']:
            return None
            
        # Các định dạng ngày phổ biến
        date_formats = [
            "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d.%m.%y",
            "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", 
            "%d/%m/%y", "%d-%m-%y", "%d/%-m/%Y",
            "%d.%-m.%Y", "%d.%-m.%y"
        ]
        
        # Xử lý định dạng rút gọn (VD: 28.3.23)
        if '.' in val and len(val.split('.')) == 3:
            parts = val.split('.')
            if len(parts[2]) == 2:  # Năm 2 chữ số
                val = f"{parts[0]}.{parts[1]}.20{parts[2]}"
        
        # Xử lý định dạng với tháng dạng text (VD: 12-Apr-2023)
        month_mappings = {
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
            'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
            'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
        }
        
        val_lower = val.lower()
        for month_name, month_num in month_mappings.items():
            if month_name in val_lower:
                val = val_lower.replace(month_name, month_num)
                break
        
        for fmt in date_formats:
            try:
                date_obj = datetime.strptime(val, fmt)
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                continue
                
        return None
    
    def fix_type_column(self, series):
        """Sửa cột Type dựa trên phân tích từ EDA - xử lý nhiều variant"""
        if series.name != 'Type':
            return series
        
        # Mapping để sửa các giá trị không nhất quán
        mapping = {
            'h': 'H',
            'm': 'M', 
            'L ': 'L',  # L có khoảng trắng
            '??': np.nan,
            'HIGH': 'H',
            'high': 'H',
            'MEDIUM': 'M',
            'Medium': 'M',
            'MED': 'M',
            'medium': 'M',
            'low': 'L',
            'LOW': 'L'
        }
        
        # Áp dụng mapping
        cleaned_series = series.replace(mapping)
        
        # Loại bỏ khoảng trắng đầu cuối
        cleaned_series = cleaned_series.astype(str).str.strip()
        
        # Chuyển các giá trị không hợp lệ thành NaN
        valid_types = ['L', 'M', 'H']
        cleaned_series = cleaned_series.apply(lambda x: x if x in valid_types else np.nan)
        
        self.cleaning_log.append(f"Fixed Type column inconsistencies: {len(mapping)} values corrected")
        
        return cleaned_series
    
    def detect_and_handle_outliers(self, df, numeric_columns):
        """Phát hiện và xử lý outliers theo phương pháp IQR"""
        outliers_found = 0
        
        for col in numeric_columns:
            if col in df.columns and df[col].notna().sum() > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Đếm outliers
                outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outliers_found += outlier_count
                
                # Có thể thêm cờ outlier
                if outlier_count > 0:
                    if 'outlier_flag' not in df.columns:
                        df['outlier_flag'] = False
                    df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), 'outlier_flag'] = True
        
        if outliers_found > 0:
            self.cleaning_log.append(f"Detected {outliers_found} outliers across numeric columns")
        
        return df
    
    def clean_data_simple(self, df):
        """
        Phương thức làm sạch đơn giản dựa trên notebook
        """
        self.cleaning_log = []
        df_cleaned = df.copy()
        
        # 1. Làm sạch cột Type nếu có
        if 'Type' in df_cleaned.columns:
            df_cleaned['Type'] = self.fix_type_column(df_cleaned['Type'])
        
        # 2. Làm sạch các cột text
        text_columns = df_cleaned.select_dtypes(include=['object']).columns
        for col in text_columns:
            if not any(keyword in col.lower() for keyword in ['id', 'product']):
                df_cleaned[col] = df_cleaned[col].apply(self.clean_text)
        
        self.cleaning_log.append(f"Cleaned {len(text_columns)} text columns")
        
        # 3. Xử lý cột giá nếu có
        price_columns = [col for col in df_cleaned.columns if 'giá' in col.lower() or 'price' in col.lower()]
        for col in price_columns:
            df_cleaned[col] = df_cleaned[col].apply(self.clean_price)
            self.cleaning_log.append(f"Cleaned price column: {col}")
        
        # 4. Xử lý cột ngày nếu có  
        date_columns = [col for col in df_cleaned.columns if 'ngày' in col.lower() or 'date' in col.lower()]
        for col in date_columns:
            df_cleaned[col] = df_cleaned[col].apply(self.clean_date)
            self.cleaning_log.append(f"Cleaned date column: {col}")
        
        # 5. Phát hiện outliers
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        # Loại bỏ cột binary/flag
        numeric_columns = [col for col in numeric_columns if df_cleaned[col].nunique() > 2]
        
        if numeric_columns:
            df_cleaned = self.detect_and_handle_outliers(df_cleaned, numeric_columns)
        
        # 6. Xử lý missing values đơn giản
        missing_before = df_cleaned.isnull().sum().sum()
        
        # Với numeric columns: dùng median
        for col in df_cleaned.select_dtypes(include=[np.number]).columns:
            if df_cleaned[col].isnull().any():
                median_val = df_cleaned[col].median()
                df_cleaned[col] = df_cleaned[col].fillna(median_val)
        
        # Với categorical columns: dùng mode
        for col in df_cleaned.select_dtypes(include=['object']).columns:
            if df_cleaned[col].isnull().any():
                mode_val = df_cleaned[col].mode().iloc[0] if not df_cleaned[col].mode().empty else 'Unknown'
                df_cleaned[col] = df_cleaned[col].fillna(mode_val)
        
        missing_after = df_cleaned.isnull().sum().sum()
        if missing_before > missing_after:
            self.cleaning_log.append(f"Filled {missing_before - missing_after} missing values")
        
        # 7. Bỏ trùng lặp
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        removed_duplicates = initial_rows - len(df_cleaned)
        
        if removed_duplicates > 0:
            self.cleaning_log.append(f"Removed {removed_duplicates} duplicate rows")
        
        return df_cleaned
    
    def get_cleaning_summary(self):
        """Trả về tóm tắt quá trình làm sạch"""
        return self.cleaning_log

# Wrapper function để tích hợp dễ dàng
def clean_maintenance_data(df):
    """
    Hàm tiện ích để làm sạch dữ liệu predictive maintenance
    """
    cleaner = ImprovedDataCleaner()
    cleaned_df = cleaner.clean_data_simple(df)
    summary = cleaner.get_cleaning_summary()
    
    return cleaned_df, summary