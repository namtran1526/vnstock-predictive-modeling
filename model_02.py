import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

print("HUẤN LUYỆN MÔ HÌNH")

# TẢI DATA ĐÃ CORE LẠI VÀ ĐÃ TỐI ƯU
try:
    train_df = pd.read_csv("VN30_Train.csv")
    test_df = pd.read_csv("VN30_Test.csv")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file data tối ưu.")
    exit()

# Tách riêng Đầu vào (X) và Nhãn (y)
# Lọc bỏ các cột định danh để model chỉ tập trung vào 15 chỉ báo kỹ thuật
cols_to_drop = ['Ticker', 'time', 'close', 'Target_Dynamic']

X_train = train_df.drop(columns=cols_to_drop)
y_train = train_df['Target_Dynamic']

X_test = test_df.drop(columns=cols_to_drop)
y_test = test_df['Target_Dynamic']

print(f"Số lượng Đặc trưng đưa vào học: {X_train.shape[1]} cột")

# TÍNH TOÁN LẠI TRỌNG SỐ
num_class_0 = (y_train == 0).sum()
num_class_1 = (y_train == 1).sum()
scale_weight_xgb = num_class_0 / num_class_1

print(f"\nKhởi tạo XGBoost với Trọng số phạt (Scale Pos Weight) = {scale_weight_xgb:.2f}")

# KHỞI TẠO VÀ HUẤN LUYỆN CỖ MÁY
xgb_model = XGBClassifier(
    n_estimators=300,          # 300 cây
    learning_rate=0.05,        # Tốc độ học
    max_depth=6,               # Độ sâu giới hạn để không bị overfitting
    scale_pos_weight=scale_weight_xgb, # Ép mô hình tập trung vào lệnh MUA
    random_state=42,           # Cố định random
    n_jobs=-1                  # Kích hoạt đa luồng CPU
)

print("\nBắt đầu quá trình huấn luyện ")
xgb_model.fit(X_train, y_train)
print("Huấn luyện thành công!")

# DỰ BÁO TRÊN TẬP TEST (2025-2026)
print("\nĐang dự báo trên dữ liệu 2025-2026")
y_pred = xgb_model.predict(X_test)

# XUẤT BÁO CÁO KẾT QUẢ
print("BÁO CÁO KẾT QUẢ")
print(classification_report(y_test, y_pred, target_names=['Nhãn 0 (Không Mua)', 'Nhãn 1 (MUA)']))

# VẼ MA TRẬN NHẦM LẪN
cm = confusion_matrix(y_test, y_pred)
true_positives = cm[1, 1]
false_positives = cm[0, 1]
total_predicted_buys = true_positives + false_positives
win_rate = true_positives / total_predicted_buys
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Đoán: KHÔNG MUA', 'Đoán: MUA'],
            yticklabels=['Thực tế: KHÔNG TĂNG', 'Thực tế: TĂNG > 1.5%'])
plt.title(f'Ma trận nhầm lẫn Thực nghiệm 2 (Win-rate = {win_rate*100:.2f}%)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('chart/TN2.png', bbox_inches='tight')
plt.close()

# LƯU LẠI MÔ HÌNH ĐÃ TRAIN XONG
model_filename = "xgboost_quant_model.pkl"
joblib.dump(xgb_model, model_filename)
print(f"\nĐã đóng gói dự đoán thành công vào file: {model_filename}")