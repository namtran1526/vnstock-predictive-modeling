import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("THỰC NGHIỆM 2A")

# 1. TẢI TẬP DỮ LIỆU MỤC TIÊU ĐỘNG (TRIPLE BARRIER)
print("\nĐang tải data từ VN30_Dynamic_Dataset.csv")
try:
    df = pd.read_csv("VN30_Dynamic_Dataset.csv")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file VN30_Dynamic_Dataset.csv. Hãy chạy file tạo nhãn động trước!")
    exit()

# Phân tách các cột thông tin định danh (Metadata) và các cột chỉ báo kỹ thuật (Features)
metadata_cols = ['Ticker', 'time', 'close', 'Target_Dynamic']
features_cols = [col for col in df.columns if col not in metadata_cols]

# Chia tách tập Train và Tập Test theo trục thời gian
train_mask = df['time'] < '2025-01-01'
test_mask = df['time'] >= '2025-01-01'

train_df = df[train_mask].reset_index(drop=True)
test_df = df[test_mask].reset_index(drop=True)

X_train_raw = train_df[features_cols]
y_train = train_df['Target_Dynamic']
X_test_raw = test_df[features_cols]
y_test = test_df['Target_Dynamic']

print(f" - Số lượng chỉ báo đầu vào ban đầu: {len(features_cols)} cột")
print(f" - Số lượng dòng tập Train: {X_train_raw.shape[0]} | Tập Test: {X_test_raw.shape[0]}")

# CHUẨN HÓA DỮ LIỆU & NÉN BẰNG PCA (CHỐNG DATA LEAKAGE)
print("\nTiến hành chuẩn hóa Z-Score và nén dữ liệu bằng PCA")

# PCA bắt buộc phải đi kèm với StandardScaler vì các chỉ báo lệch thang đo (RSI vs Volume)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw) # Chỉ transform, không fit dữ liệu tương lai tránh overfitting.

# Khởi tạo PCA giữ lại đúng 15 cấu phần độc lập (Thay thế cho cách chọn Top 15 thủ công)
SO_CAU_PHAN_PCA = 15
pca = PCA(n_components=SO_CAU_PHAN_PCA, random_state=42)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Tính toán lượng thông tin được giữ lại sau khi nén
tong_thong_tin = np.sum(pca.explained_variance_ratio_) * 100
print(f"Nén thành công thành {SO_CAU_PHAN_PCA} siêu biến độc lập.")
print(f" -> 15 siêu biến này giữ lại được: {tong_thong_tin:.2f}% tổng lượng thông tin gốc.")

# HUẤN LUYỆN XGBOOST TRÊN KHÔNG GIAN PCA
scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nKhởi tạo XGBoost Core trên nền tảng PCA (Weight Phạt: {scale_weight:.2f})...")

xgb_pca = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=scale_weight,
    random_state=42,
    n_jobs=-1
)

print(" Bắt đầu huấn luyện mô hình")
xgb_pca.fit(X_train_pca, y_train)
print("Huấn luyện thành công!")

# ĐÁNH GIÁ
print("\nĐang dự đoán trên tập test pca mới")
y_pred = xgb_pca.predict(X_test_pca)

print("\n" + "=" * 55)
print("BÁO CÁO KẾT QUẢ")
print("=" * 55)
print(classification_report(y_test, y_pred, target_names=['Nhãn 0 (Không Mua)', 'Nhãn 1 (MUA)']))

cm = confusion_matrix(y_test, y_pred)
print("\nChi tiết Ma trận nhầm lẫn của thực nghiệm PCA:")
print(f" - Né bẫy đúng (TN):       {cm[0][0]} ngày")
print(f" - Mua sai/Đu đỉnh (FP):  {cm[0][1]} ngày")
print(f" - Bỏ lỡ cơ hội (FN):     {cm[1][0]} ngày")
print(f" - Bắt trúng sóng (TP):   {cm[1][1]} ngày")

win_rate = cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
print(f"\nTỷ lệ lệnh MUA chính xác (Win-rate mới): {win_rate*100:.2f}%")

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=['Đoán: KHÔNG MUA', 'Đoán: MUA'],
            yticklabels=['Thực tế: KHÔNG TĂNG', 'Thực tế: ĐẠT TARGET ĐỘNG'])
plt.title(f'Ma trận nhầm lẫn Thực nghiệm 2a (Win-rate = {win_rate*100:.2f}%)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('chart/TN2a.png', bbox_inches='tight')
plt.close()