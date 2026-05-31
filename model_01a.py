import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("THỰC NGHIỆM 1A")

# TẢI DATA ĐÃ LỌC TOP 15
try:
    train_df = pd.read_csv("VN30_Train.csv")
    test_df = pd.read_csv("VN30_Test.csv")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy các file data!")
    exit()

# Gộp tạm thời để làm Feature Engineering đồng bộ cho cả 2 tập, tránh lệch cấu trúc
train_len = len(train_df)
full_df = pd.concat([train_df, test_df], ignore_index=True)

# FEATURE ENGINEERING (ĐẶC TRƯNG VĨ MÔ)
print("\nTiến hành thiết kế đặc trưng bối cảnh thị trường.")

# Xác định danh sách các chỉ báo kỹ thuật đang có trong file (bỏ các cột metadata)
metadata_cols = ['Ticker', 'time', 'close', 'Target_T3']
indicators = [col for col in train_df.columns if col not in metadata_cols]

# Lấy ra 2 chỉ báo đứng đầu để làm đặc trưng thị trường (Giúp mô hình không bị quá nặng nhưng vẫn đủ thông tin vĩ mô)
top_1_indicator = indicators[0]
top_2_indicator = indicators[1]

print(f" - Tạo đặc trưng trung bình thị trường cho chỉ báo: {top_1_indicator}")
print(f" - Tạo đặc trưng trung bình thị trường cho chỉ báo: {top_2_indicator}")

# Tính toán giá trị trung bình của toàn bộ rổ VN30 theo từng ngày
market_trend_1 = full_df.groupby('time')[top_1_indicator].transform('mean')
market_trend_2 = full_df.groupby('time')[top_2_indicator].transform('mean')

# Bơm các đặc trưng bối cảnh này vào dữ liệu
full_df[f'market_avg_{top_1_indicator}'] = market_trend_1
full_df[f'market_avg_{top_2_indicator}'] = market_trend_2

# Tách lại thành tập Train và Test sau khi đã có đặc trưng mới
train_df_ext = full_df.iloc[:train_len]
test_df_ext = full_df.iloc[train_len:]

# Phân tách X (Đầu vào) và y (Nhãn)
X_train = train_df_ext.drop(columns=metadata_cols)
y_train = train_df_ext['Target_T3']
X_test = test_df_ext.drop(columns=metadata_cols)
y_test = test_df_ext['Target_T3']

print(f" -> Cấu trúc dữ liệu mới: {X_train.shape[1]} đặc trưng (Đã thêm biến vĩ mô)")

# HYPERPARAMETER TUNING (DÒ TÌM THAM SỐ)
print("\nĐang thiết lập lưới tham số để dò tìm cấu hình tối ưu.")

# Tính toán lại trọng số phạt mất cân bằng cho tập Train mới
scale_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Khởi tạo mô hình nền tảng
xgb_base = XGBClassifier(scale_pos_weight=scale_weight, random_state=42, n_jobs=-1)

# Định nghĩa không gian tìm kiếm (Lưới tham số)
# Giới hạn không gian vừa phải để chạy không bị quá tải
param_grid = {
    'n_estimators': [100, 200],       # Số lượng cây quyết định
    'max_depth': [4, 6],              # Độ sâu của cây để kiểm soát học vẹt
    'learning_rate': [0.01, 0.05]     # Biên độ sửa sai của thuật toán
}

print(" Bắt đầu quét lưới tham số (Grid Search).")
# Tìm kiếm tổ hợp tốt nhất, ưu tiên tối ưu hóa điểm Precision cho Nhãn 1
grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring='precision',
    cv=3, # Chia nhỏ tập train thành 3 phần để kiểm tra chéo nội bộ
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Rút trích bộ não tốt nhất sau khi dò tìm
best_model = grid_search.best_estimator_

print("\nBỘ THAM SỐ ĐƯỢC TÌM THẤY:")
for param_name, param_value in grid_search.best_params_.items():
    print(f"  - {param_name}: {param_value}")

# ĐÁNH GIÁ KẾT QUẢ MÔ HÌNH
print("\nĐang kiểm tra khả năng mô hình tối ưu trên tập Test (2025-2026)")
y_pred_opt = best_model.predict(X_test)

print("\nBÁO CÁO KẾT QUẢ THỰC NGHIỆM 1A")
print(classification_report(y_test, y_pred_opt, target_names=['Nhãn 0 (Không Mua)', 'Nhãn 1 (MUA)']))

# Tính toán ma trận nhầm lẫn mới
cm_opt = confusion_matrix(y_test, y_pred_opt)
print("\nChi tiết Ma trận nhầm lẫn mới:")
print(f" - Né bẫy đúng (TN): {cm_opt[0][0]} ngày")
print(f" - Mua sai/Đu đỉnh (FP): {cm_opt[0][1]} ngày")
print(f" - Bỏ lỡ cơ hội (FN): {cm_opt[1][0]} ngày")
print(f" - Bắt trúng sóng (TP): {cm_opt[1][1]} ngày")

# Tính toán lại Win-rate mới để so sánh
win_rate = cm_opt[1][1] / (cm_opt[1][1] + cm_opt[0][1])
print(f"\nTỷ lệ lệnh MUA chính xác (Win-rate): {win_rate*100:.2f}%")

# VẼ VÀ XUẤT MA TRẬN NHẦM LẪN
cm_opt = confusion_matrix(y_test, y_pred_opt) # Sử dụng đúng biến y_pred_opt của code trên
plt.figure(figsize=(7, 5))
sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Đoán: KHÔNG MUA', 'Đoán: MUA'],
            yticklabels=['Thực tế: KHÔNG TĂNG', 'Thực tế: TĂNG > 1.5%'])
plt.title(f'Ma trận nhầm lẫn Thực nghiệm 1a (Win-rate = {win_rate*100:.2f}%)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('chart/TN1a.png', bbox_inches='tight')
plt.close()
