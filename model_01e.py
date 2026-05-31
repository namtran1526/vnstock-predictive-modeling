import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("THỰC NGHIỆM 1E")

# TẢI DATA & MÔ HÌNH GỐC (BASELINE)
print("\nĐang tải data gốc (Target 1.5%) và model_01")
train_df = pd.read_csv("VN30_Train.csv")
test_df = pd.read_csv("VN30_Test.csv")

cols_to_drop = ['Ticker', 'time', 'close', 'Target_T3']
X_train = train_df.drop(columns=cols_to_drop)
y_train = train_df['Target_T3']
X_test = test_df.drop(columns=cols_to_drop)
y_test = test_df['Target_T3']

# Tải lại mô hình Baseline
model_1 = joblib.load("xgboost_quant_model.pkl")

# THU THẬP KINH NGHIỆM CHO MÔ HÌNH THỨ 2
print("\nMô hình thứ nhất đang dự đoán trên tập Train để mô hình thứ 2 quan sát")
# Dùng Cross-validation để ép mô hình 1 dự đoán khách quan, không bị học vẹt
y_train_pred_m1 = cross_val_predict(model_1, X_train, y_train, cv=3)

# Lọc ra CHỈ NHỮNG NGÀY mà mô hình 1 hô "MUA" (Dự đoán = 1)
mask_m1_buys = (y_train_pred_m1 == 1)
X_meta_train = X_train[mask_m1_buys]
y_meta_train = y_train[mask_m1_buys]

print(f" -> Mô hình Số 1 đã hô MUA tổng cộng {mask_m1_buys.sum()} lần trong quá khứ.")
print(f" -> Trong đó có {(y_meta_train==1).sum()} lần ĐÚNG và {(y_meta_train==0).sum()} lần SAI (Đu đỉnh).")

# HUẤN LUYỆN MÔ HÌNH SỐ 2
print("\nĐang huấn luyện mô hình Số 2 bắt lỗi mô hình Số 1")
# Mô hình cần nhỏ gọn, khắt khe và không cần quá phức tạp
scale_weight_meta = (y_meta_train == 0).sum() / (y_meta_train == 1).sum()

model_2_bodyguard = XGBClassifier(
    n_estimators=100,         # Chỉ cần 100 cây
    max_depth=4,              # Suy nghĩ đơn giản để tránh Overfitting
    learning_rate=0.05,
    scale_pos_weight=scale_weight_meta,
    random_state=42,
    n_jobs=-1
)
model_2_bodyguard.fit(X_meta_train, y_meta_train)

# DỰ ĐOÁN TRÊN TẬP TEST (2025-2026)
print("\nHệ thống mô hình kép tiến vào thực chiến (Tập Test)")
# Mô hình 1 đi càn quét thị trường
y_test_pred_m1 = model_1.predict(X_test)

# Mảng chứa quyết định cuối cùng (Mặc định là 0 - Không Mua)
final_predictions = np.zeros(len(y_test))

# Mô hình 2 kiểm duyệt từng lệnh MUA của mô hình 1
for i in range(len(y_test)):
    if y_test_pred_m1[i] == 1: # Nếu AI 1 hô Mua
        row_data = X_test.iloc[[i]]
        # Mô hình 2 phán xét lệnh này (1 = Đồng ý, 0 = Bác bỏ)
        approval = model_2_bodyguard.predict(row_data)[0]
        final_predictions[i] = approval

# BÁO CÁO KẾT QUẢ HỆ THỐNG KÉP
print("\nBÁO CÁO THÀNH TÍCH (HỆ THỐNG META-LABELING)")
print(classification_report(y_test, final_predictions, target_names=['Nhãn 0 (Không Mua)', 'Nhãn 1 (MUA)']))

cm = confusion_matrix(y_test, final_predictions)
print("\nChi tiết Ma trận nhầm lẫn CỦA HỆ THỐNG KÉP:")
print(f" - Né bẫy đúng (TN):       {cm[0][0]} ngày")
print(f" - Mua sai/Đu đỉnh (FP):  {cm[0][1]} ngày")
print(f" - Bỏ lỡ cơ hội (FN):     {cm[1][0]} ngày")
print(f" - Bắt trúng sóng (TP):   {cm[1][1]} ngày")

win_rate = cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
print(f"\nTỷ lệ lệnh MUA chính xác (Win-rate mới): {win_rate*100:.2f}%")

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Đoán: KHÔNG MUA', 'Đoán: MUA'],
            yticklabels=['Thực tế: KHÔNG TĂNG', 'Thực tế: TĂNG > 1.5%'])
plt.title(f'Ma trận nhầm lẫn Thực nghiệm 1e (Win-rate = {win_rate*100:.2f}%)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('chart/TN1e.png', bbox_inches='tight')
plt.close()