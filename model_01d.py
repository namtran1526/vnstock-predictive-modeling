import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("THỰC NGHỆM 1D")

# 1. ĐỌC DỮ LIỆU GỐC (Đảm bảo đã reset về Target 1.5%)
try:
    train_df = pd.read_csv("VN30_Train.csv")
    test_df = pd.read_csv("VN30_Test.csv")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file. Hãy đảm bảo đã tạo lại 2 file này ở chuẩn 1.5%.")
    exit()

cols_to_exclude = ['Ticker', 'time', 'close', 'Target_T3']
features = [c for c in train_df.columns if c not in cols_to_exclude]

# BIẾN ĐỔI CHỈ BÁO THÀNH ĐIỂM XẾP HẠNG
print("\nQuy đổi các chỉ báo tuyệt đối thành điểm xếp hạng (0.0 -> 1.0)")
for col in features:
    # Lệnh rank(pct=True) sẽ xếp hạng cổ phiếu trong cùng 1 ngày (time)
    # Ví dụ: Ngày 15/01/2025, FPT mạnh nhất rổ VN30 -> Điểm = 1.0, VCB bét bảng -> Điểm = 0.0
    train_df[col] = train_df.groupby('time')[col].rank(pct=True)
    test_df[col] = test_df.groupby('time')[col].rank(pct=True)

# Tách riêng Input (X) và Label (y)
X_train = train_df.drop(columns=cols_to_exclude)
y_train = train_df['Target_T3']
X_test = test_df.drop(columns=cols_to_exclude)
y_test = test_df['Target_T3']

# HUẤN LUYỆN LẠI MÔ HÌNH
# Tính lại tỷ lệ phạt cho chuẩn xác
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
print("Huấn luyện thành công!")

# KIỂM ĐỊNH TRÊN TẬP TEST
print("\nĐang chấm điểm trên tập Test (2025-2026)")
y_pred = xgb_model.predict(X_test)

print("\nBÁO CÁO THÀNH TÍCH (TƯ DUY XẾP HẠNG)")
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
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Đoán: KHÔNG MUA', 'Đoán: MUA'],
            yticklabels=['Thực tế: KHÔNG TĂNG', 'Thực tế: TĂNG > 1.5%'])
plt.title(f'Ma trận nhầm lẫn Thực nghiệm 1d (Win-rate = {win_rate*100:.2f}%)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('chart/TN1d.png', bbox_inches='tight')
plt.close()