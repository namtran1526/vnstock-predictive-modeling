import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

print("=" * 60)
print("--- HỆ THỐNG XỬ LÝ DỮ LIỆU CHỨNG KHOÁN (BƯỚC 2 & BƯỚC 3) ---")
print("=" * 60)

# 1. ĐỌC DỮ LIỆU ĐÃ LÀM SẠCH
file_path = "VN30_Cleaned_Dataset.csv"
try:
    df = pd.read_csv(file_path)
    print(f"Nạp dữ liệu thành công! Tổng số dòng gốc: {len(df)}")
except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file '{file_path}' trong thư mục project!")
    print("Vui lòng đảm bảo bạn đã chạy file tạo dataset sạch trước đó.")
    exit()

# Sắp xếp chuẩn xác theo Mã cổ phiếu và Thời gian để bảo toàn tính chuỗi thời gian
df = df.sort_values(by=['Ticker', 'time'])

# =======================================================
# THỰC THI BƯỚC 3: CHIA DỮ LIỆU CHUẨN XÁC THEO MỐC THỜI GIAN
# =======================================================
print("\n[Bước 3] Đang phân tách dòng thời gian (2021-2024 làm Train, 2025-2026 làm Test)...")

# 1. Đảm bảo cột 'time' là định dạng ngày tháng chuẩn của Pandas
df['time'] = pd.to_datetime(df['time'])

# 2. Đặt "lưỡi dao" cắt thời gian vào ngày cuối cùng của năm 2024
cutoff_date = pd.to_datetime("2024-12-31")

# 3. Lọc tách dữ liệu cho toàn bộ 30 mã cùng lúc
train_df = df[df['time'] <= cutoff_date]
test_df = df[df['time'] > cutoff_date]

print(f" - Dữ liệu Train (Trước 2025): {len(train_df)} dòng")
print(f" - Dữ liệu Test  (Từ 2025):    {len(test_df)} dòng")

# Tách riêng phần biến độc lập (X_train) để đưa vào mô hình trinh sát tìm biến quan trọng
cols_to_exclude = ['Ticker', 'time', 'open', 'high', 'low', 'close', 'volume', 'Future_Return_3D', 'Target_T3']
X_train_full = train_df.drop(columns=[col for col in cols_to_exclude if col in train_df.columns])
y_train = train_df['Target_T3']


# =======================================================
# THỰC THI BƯỚC 2: CHỌN LỌC ĐẶC TRƯNG (FEATURE SELECTION)
# =======================================================
print("[Bước 2] Đang chạy Random Forest trinh sát để chấm điểm 80 chỉ báo...")
# Sử dụng class_weight='balanced' để mô hình trinh sát không bị lệch về phe đa số
scout_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
scout_model.fit(X_train_full, y_train)

# Rút trích điểm số quan trọng của từng đặc trưng
feature_importance_df = pd.DataFrame({
    'Feature': X_train_full.columns,
    'Importance': scout_model.feature_importances_
})

# Lấy ra Top 15 đặc trưng mạnh nhất
top_15_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(15)

# FIX LỖI THIẾU SÓT: IN BẢNG KẾT QUẢ TOP 15 KÈM PHẦN TRĂM ĐÓNG GÓP
print("\n🏆 BẢNG XẾP HẠNG TOP 15 CHỈ BÁO QUYỀN LỰC NHẤT THỊ TRƯỜNG:")
print("-" * 55)
print(f" {'STT':<4} | {'Tên Chỉ Báo Kỹ Thuật (Feature)':<28} | Trọng số đóng góp")
print("-" * 55)
for i, (index, row) in enumerate(top_15_df.iterrows(), 1):
    diem_so = row['Importance'] * 100  # Chuyển sang phần trăm (%)
    print(f"  {i:<3} | {row['Feature']:<28} | {diem_so:.2f}%")
print("-" * 55)

# Trích xuất danh sách tên của 15 cột này để tiến hành lọc file
top_15_features = top_15_df['Feature'].tolist()


# =======================================================
# FIX LỖI THIẾU SÓT: TÍNH TRỌNG SỐ MẤT CÂN BẰNG TRÊN TẬP TRAIN
# =======================================================
print("\n[Bước 3 bổ sung] Đang tính toán trọng số phạt phục vụ cho mô hình Bước 4...")

# 1. Tính trọng số phạt chuẩn (Class Weights)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
weight_dict = {0: class_weights[0], 1: class_weights[1]}

# 2. Tính riêng tham số scale_pos_weight chuyên dụng cho thuật toán XGBoost
num_class_0 = (y_train == 0).sum()
num_class_1 = (y_train == 1).sum()
scale_weight_xgb = num_class_0 / num_class_1

print("\n⚖️ THÔNG SỐ CÂN BẰNG DỮ LIỆU ĐƯỢC TRÍCH XUẤT:")
print("-" * 60)
print(f" - Tổng mẫu Nhãn 0 (Không Mua) trong tập Train: {num_class_0} ngày")
print(f" - Tổng mẫu Nhãn 1 (MUA) trong tập Train:       {num_class_1} ngày")
print(f" - Trọng số phạt nếu AI đoán sai Nhãn 0:        {weight_dict[0]:.2f}")
print(f" - Trọng số phạt nếu AI bỏ lỡ Nhãn 1:           {weight_dict[1]:.2f}")
print(f" 👉 Tham số 'scale_pos_weight' cấu hình cho XGBoost: {scale_weight_xgb:.2f}")
print("-" * 60)


# =======================================================
# ĐÓNG GÓI VÀ XUẤT FILE DATASET TỐI ƯU
# =======================================================
print("\n[Kết quả] Đang đóng gói dữ liệu và lọc bỏ các biến rác...")

# Giữ lại các cột cốt lõi làm Metadata (Dùng cho Backtest tính tiền lời lỗ sau này) + Top 15 đặc trưng
final_columns_to_keep = ['Ticker', 'time', 'close', 'Target_T3'] + top_15_features

# Ép xung 2 tập dữ liệu theo danh sách cột tối ưu
train_optimized = train_df[final_columns_to_keep]
test_optimized = test_df[final_columns_to_keep]

# Lưu thành 2 file CSV riêng biệt trong thư mục dự án
train_optimized.to_csv("VN30_Train_Optimized.csv", index=False)
test_optimized.to_csv("VN30_Test_Optimized.csv", index=False)

print("\n✅ HOÀN TẤT THÀNH CÔNG QUY TRÌNH XỬ LÝ SƠ CHẾ DATA!")
print(f" 📂 Đã tạo file học tập:  VN30_Train_Optimized.csv ({train_optimized.shape[0]} dòng, {train_optimized.shape[1]} cột)")
print(f" 📂 Đã tạo file thi cử:   VN30_Test_Optimized.csv ({test_optimized.shape[0]} dòng, {test_optimized.shape[1]} cột)")
print("=" * 60)