import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("THỰC NGHIỆM 2D")

# TẢI TẬP DỮ LIỆU CỐT LÕI
print("\nĐang tải dữ liệu gốc VN30_Dynamic_Dataset.csv")
try:
    df = pd.read_csv("VN30_Dynamic_Dataset.csv")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file. Hãy đảm bảo bạn đang dùng file từ model_02!")
    exit()

# TẠO BỘ LỌC VĨ MÔ
print("\nĐang tính toán Chỉ số thị trường chung và xu hướng dài hạn (SMA 100)")

# Tính VN30 Index (Mô phỏng) bằng trung bình giá của toàn rổ mỗi ngày
market_index = df.groupby('time')['close'].mean().reset_index()
market_index.rename(columns={'close': 'Market_Index'}, inplace=True)

# Tính đường trung bình dài hạn (100 ngày giao dịch ~ 5 tháng)
market_index['SMA_100'] = market_index['Market_Index'].rolling(window=100, min_periods=1).mean()

# Xác định Chế độ thị trường (Regime): 1 = Uptrend, 0 = Downtrend
market_index['Macro_Uptrend'] = (market_index['Market_Index'] > market_index['SMA_100']).astype(int)

# Gộp lăng kính vĩ mô này vào dữ liệu của từng cổ phiếu
df = pd.merge(df, market_index[['time', 'Macro_Uptrend']], on='time', how='left')

# CHUẨN BỊ DỰ BÁO & HUẤN LUYỆN MÔ HÌNH
print("\nĐang huấn luyện mô hình cốt lõi")
metadata_cols = ['Ticker', 'time', 'close', 'Target_Dynamic', 'Macro_Uptrend']
features = [c for c in df.columns if c not in metadata_cols]

train_mask = df['time'] < '2025-01-01'
test_mask = df['time'] >= '2025-01-01'

train_df = df[train_mask]
test_df = df[test_mask]

X_train = train_df[features]
y_train = train_df['Target_Dynamic']
X_test = test_df[features]
y_test = test_df['Target_Dynamic']
# Lưu lại cột vĩ mô của tập Test để dùng làm khiên chắn ở bước sau
macro_shield_test = test_df['Macro_Uptrend'].values

scale_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=scale_weight,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

# KÍCH HOẠT KHIÊN VĨ MÔ & BÁO CÁO
print("\nMô hình đang thi hành lệnh: CẤM MUA KHI THỊ TRƯỜNG DOWNTREND")
# Mô hình đưa ra dự đoán kỹ thuật thuần túy
ai_raw_predictions = xgb_model.predict(X_test)

# Kích hoạt Khiên Vĩ Mô: Lệnh MUA chỉ có hiệu lực (1) khi Vĩ mô Uptrend (1). 1 AND 1 = 1.
# Nếu Vĩ mô Downtrend (0), lệnh MUA bị bóp nghẹt ngay lập tức. 1 AND 0 = 0.
final_predictions = ai_raw_predictions & macro_shield_test

print("\n" + "=" * 55)
print("BÁO CÁO KẾT QUẢ")
print("=" * 55)
print(classification_report(y_test, final_predictions, target_names=['Nhãn 0 (Không Mua)', 'Nhãn 1 (MUA)']))

cm = confusion_matrix(y_test, final_predictions)
print("\nChi tiết Ma trận nhầm lẫn:")
print(f" - Né bẫy đúng (TN):       {cm[0][0]} ngày")
print(f" - Mua sai/Đu đỉnh (FP):  {cm[0][1]} ngày (Kỳ vọng giảm cực mạnh!)")
print(f" - Bỏ lỡ cơ hội (FN):     {cm[1][0]} ngày")
print(f" - Bắt trúng sóng (TP):   {cm[1][1]} ngày")

win_rate = cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
print(f"\nTỶ LỆ THẮNG (WIN-RATE): {win_rate*100:.2f}%")

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Đoán: KHÔNG MUA', 'Đoán: MUA'],
            yticklabels=['Thực tế: KHÔNG TĂNG', 'Thực tế: ĐẠT TARGET ĐỘNG'])
plt.title(f'Ma trận nhầm lẫn Thực nghiệm 2c (Win-rate = {win_rate*100:.2f}%)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('chart/TN2c.png', bbox_inches='tight')
plt.close()