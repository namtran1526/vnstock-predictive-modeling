import pandas as pd

print("THỰC NGHIỆM 1C")

# Đọc file data gốc
file_path = "VN30_Cleaned_Dataset.csv"
df = pd.read_csv(file_path)

# Xóa cột Target cũ
if 'Target_T3' in df.columns:
    df = df.drop(columns=['Target_T3'])

# Tạo cột Target MỚI: Tăng > 3% mới được tính là 1
# Bạn có thể đổi 0.03 thành 0.025 (2.5%) nếu muốn nới lỏng một chút
MUC_TIEU_LOI_NHUAN = 0.03

# Giả định cột Future_Return_3D của bạn là dạng số thập phân (0.03 = 3%)
df['Target_T3'] = (df['Future_Return_3D'] > MUC_TIEU_LOI_NHUAN).astype(int)

# Thống kê nhanh sự thay đổi
so_nhan_0 = (df['Target_T3'] == 0).sum()
so_nhan_1 = (df['Target_T3'] == 1).sum()
ty_le_nhan_1 = (so_nhan_1 / len(df)) * 100

print(f"\nĐã cập nhật xong Nhãn mới (Mục tiêu > {MUC_TIEU_LOI_NHUAN * 100}%):")
print(f" - Số ngày KHÔNG ĐẠT (Nhãn 0): {so_nhan_0}")
print(f" - Số ngày ĐẠT CHUẨN (Nhãn 1):  {so_nhan_1} (Chiếm {ty_le_nhan_1:.2f}% tổng dữ liệu)")

# Ghi đè lại file
df.to_csv(file_path, index=False)
print(f"\nĐã ghi đè thành công vào file: {file_path}")
print("Chạy lại lần lượt file feature_split và model_01!")