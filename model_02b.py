import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("THỰC NGHIỆM 2B")

# TẢI DATA DỮ LIỆU MỤC TIÊU ĐỘNG (TRIPLE BARRIER)
try:
    df = pd.read_csv("VN30_Dynamic_Dataset.csv")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file VN30_Dynamic_Dataset.csv!")
    exit()

metadata_cols = ['Ticker', 'time', 'close', 'Target_Dynamic']
all_features = [col for col in df.columns if col not in metadata_cols]

# THUẬT TOÁN PHÂN LOẠI NHÓM CHỈ BÁO
print("\nĐang phân nhóm chỉ báo theo Domain")

trend_cols = [c for c in all_features if any(x in c.lower() for x in ['macd', 'sma', 'ema', 'trend', 'adx'])]
mom_cols = [c for c in all_features if any(x in c.lower() for x in ['rsi', 'stoch', 'cci', 'mfi', 'mom', 'roc'])]
volat_cols = [c for c in all_features if any(x in c.lower() for x in ['bb', 'atr', 'std', 'band', 'kc'])]
vol_cols = [c for c in all_features if any(x in c.lower() for x in ['vol', 'obv', 'cmf', 'vwap', 'adi'])]

# Lấy tối đa 4 chỉ báo đại diện cho mỗi nhóm để tạo thành cụm nhóm chỉ báo tốt (Khoảng 12-16 chỉ báo)
# Việc dùng list(set()) giúp loại bỏ các chỉ báo bị trùng lặp giữa các nhóm
selected_features = list(set(trend_cols[:4] + mom_cols[:4] + volat_cols[:4] + vol_cols[:4]))

print(f"Đã tuyển chọn được các nhóm {len(selected_features)} chỉ báo đa chiều:")
for f in selected_features:
    print(f"  - {f}")

# Cập nhật lại danh sách các cột cần lấy
final_columns = metadata_cols + selected_features
df_selected = df[final_columns]

# CHIA TẬP TRAIN / TEST VÀ HUẤN LUYỆN
print("\nĐang phân tách dữ liệu và khởi tạo mô hình")
train_mask = df_selected['time'] < '2025-01-01'
test_mask = df_selected['time'] >= '2025-01-01'

train_df = df_selected[train_mask]
test_df = df_selected[test_mask]

X_train = train_df[selected_features]
y_train = train_df['Target_Dynamic']
X_test = test_df[selected_features]
y_test = test_df['Target_Dynamic']

scale_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_cat = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=scale_weight,
    random_state=42,
    n_jobs=-1
)

xgb_cat.fit(X_train, y_train)
print("Huấn luyện thành công!")

# ĐÁNH GIÁ
print("\nĐang dự đoán trên tập Test (2025-2026)")
y_pred = xgb_cat.predict(X_test)

print("\nBÁO CÁO KẾT QUẢ")
print(classification_report(y_test, y_pred, target_names=['Nhãn 0 (Không Mua)', 'Nhãn 1 (MUA)']))

cm = confusion_matrix(y_test, y_pred)
print("\nChi tiết Ma trận nhầm lẫn:")
print(f" - Né bẫy đúng (TN):       {cm[0][0]} ngày")
print(f" - Mua sai/Đu đỉnh (FP):  {cm[0][1]} ngày")
print(f" - Bỏ lỡ cơ hội (FN):     {cm[1][0]} ngày")
print(f" - Bắt trúng sóng (TP):   {cm[1][1]} ngày")

win_rate = cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
print(f"\nTỷ lệ lệnh MUA chính xác (Win-rate mới): {win_rate*100:.2f}%")

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Đoán: KHÔNG MUA', 'Đoán: MUA'],
            yticklabels=['Thực tế: KHÔNG TĂNG', 'Thực tế: ĐẠT TARGET ĐỘNG'])
plt.title(f'Ma trận nhầm lẫn Thực nghiệm 2b (Win-rate = {win_rate*100:.2f}%)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('chart/TN2b.png', bbox_inches='tight')
plt.close()