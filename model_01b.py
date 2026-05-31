import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("THỰC NGHIỆM 1B")

# TẢI LẠI MÔ HÌNH GỐC VÀ DATA THÔ
try:
    test_df = pd.read_csv("VN30_Test.csv")
    model = joblib.load("xgboost_quant_model.pkl") # Tải bản gốc (model01)
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file mô hình hoặc dữ liệu Test.")
    exit()

X_test = test_df.drop(columns=['Ticker', 'time', 'close', 'Target_T3'])
y_test = test_df['Target_T3']

# XUẤT XÁC SUẤT THAY VÌ XUẤT NHÃN
print("\nÉp mô hình xuất ra xác suất % thay vì nhãn Mua/Bán.")

# Trả về 2 cột: [Xác suất không tăng, Xác suất TĂNG]
# Chỉ lấy cột số 1 (Xác suất TĂNG) bằng cách dùng [:, 1]
y_pred_probabilities = model.predict_proba(X_test)[:, 1]

# ÁP DỤNG NGƯỠNG KHẮT KHE
NGUONG_QUYET_DINH = 0.54

print(f"\nThiết lập bộ lọc cứng: CHỈ HÔ MUA KHI TỰ TIN >= {NGUONG_QUYET_DINH * 100}%")

# Tạo danh sách dự đoán mới: Nếu xác suất >= ngưỡng thì là 1 (MUA), ngược lại là 0
y_pred_strict = (y_pred_probabilities >= NGUONG_QUYET_DINH).astype(int)

# IN KẾT QUẢ SO SÁNH
print("\n" + "=" * 55)
print(f"BÁO CÁO KẾT QUẢ (NGƯỠNG {NGUONG_QUYET_DINH})")
print("=" * 55)
print(classification_report(y_test, y_pred_strict, target_names=['Nhãn 0 (Không Mua)', 'Nhãn 1 (MUA)']))

cm = confusion_matrix(y_test, y_pred_strict)
print("\nChi tiết Ma trận nhầm lẫn MỚI:")
print(f" - Né bẫy đúng (TN):       {cm[0][0]} ngày")
print(f" - Mua sai/Đu đỉnh (FP):  {cm[0][1]} ngày (Giảm bao nhiêu so với 1108 cũ?)")
print(f" - Bỏ lỡ cơ hội (FN):     {cm[1][0]} ngày")
print(f" - Bắt trúng sóng (TP):   {cm[1][1]} ngày")

# Tính tỷ lệ Win-rate
win_rate = cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
print(f"\nTỷ lệ lệnh MUA chính xác (Win-rate): {win_rate*100:.2f}%")

# Vẽ ma trận nhầm lẫn
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Đoán: KHÔNG MUA', 'Đoán: MUA'],
            yticklabels=['Thực tế: KHÔNG TĂNG', 'Thực tế: TĂNG > 1.5%'])
plt.title(f'Ma trận nhầm lẫn Thực nghiệm 1b (Win-rate = {win_rate*100:.2f}%)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('chart/TN1b.png', bbox_inches='tight')
plt.show()