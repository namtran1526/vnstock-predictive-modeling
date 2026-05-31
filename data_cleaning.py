import pandas as pd
import numpy as np

# Đọc file data
file_path = "VN30_Advanced_Quant_Dataset.csv"
df = pd.read_csv(file_path)

print(f"Kích thước dữ liệu ban đầu: {df.shape[0]} dòng, {df.shape[1]} cột")

# Khám phá xem cột nào đang bị lỗi dữ liệu
missing_count = df.isnull().sum()
columns_with_nan = missing_count[missing_count > 0]

# Nếu có cột 'ticker' (chữ thường), xóa đi để tránh rác mô hình
if 'ticker' in df.columns:
    df = df.drop(columns=['ticker'])
    print("Đã dọn dẹp xong cột 'ticker' bị trùng lặp.")

if not columns_with_nan.empty:
    print("\nDanh sách các cột có chứa NaN:")
    print(columns_with_nan)
else:
    print("\nKhông có dòng NaN nào.")

# Quét và tiêu diệt giá trị vô cực
# Trong tài chính, đôi khi chia cho volume hoặc giá = 0 sẽ sinh ra lỗi vô cực.
# Chuyển nó thành NaN để xóa luôn tránh lỗi mô hình.
df = df.replace([np.inf, -np.inf], np.nan)

# Xóa bỏ hoàn toàn các dòng chứa lỗi
# 2 cách xử lý NaN: Điền số 0 hoặc Xóa.
# Với chứng khoán, điền số 0 vào chỉ báo kỹ thuật sẽ làm sai lệch hoàn toàn biểu đồ.
# Việc xóa đi 30-50 dòng đầu tiên của mỗi mã là quá nhỏ bé so với 5 năm dữ liệu, nên xóa là tối ưu nhất.
df_cleaned = df.dropna()

print(f"\nKích thước sau khi dọn dẹp: {df_cleaned.shape[0]} dòng")
print(f"Số lượng dòng đã bị loại bỏ: {df.shape[0] - df_cleaned.shape[0]} dòng")

# Lưu lại thành file data sạch
cleaned_file_name = "VN30_Cleaned_Dataset.csv"
df_cleaned.to_csv(cleaned_file_name, index=False)
print(f"\nĐã lưu file dữ liệu sạch: {cleaned_file_name}")