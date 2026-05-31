import pandas as pd
import numpy as np
import warnings

# Tắt cảnh báo phân mảnh DataFrame của Pandas
warnings.filterwarnings("ignore")

print("KHỞI TẠO CORE MỚI: GÁN NHÃN MỤC TIÊU ĐỘNG (TRIPLE BARRIER)")

# ĐỌC DATA GỐC
file_path = "VN30_Cleaned_Dataset.csv"
try:
    df = pd.read_csv(file_path)
    # Sắp xếp lại chuẩn xác theo Mã và Thời gian để không bị tính chéo
    df = df.sort_values(by=['Ticker', 'time']).reset_index(drop=True)
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file {file_path}!")
    exit()

# CÀI ĐẶT 3 BỨC TƯỜNG
TAKE_PROFIT = 0.04  # Lãi 4% thì chốt
STOP_LOSS = -0.02  # Lỗ 2% thì cắt
MAX_HOLD_DAYS = 15  # Ôm tối đa 15 ngày

print(f" Thiết lập: Chốt lời {TAKE_PROFIT * 100}%, Cắt lỗ {STOP_LOSS * 100}%, Ôm tối đa {MAX_HOLD_DAYS} ngày.")
print(" Đang chạy thuật toán quét giá tương lai!")


# THUẬT TOÁN TÍNH TOÁN NHÃN ĐỘNG
def apply_triple_barrier(group):
    labels = []
    closes = group['close'].values
    n = len(closes)

    for i in range(n):
        label = 0  # Mặc định là Nhãn 0
        buy_price = closes[i]

        # Quét tương lai từ ngày T+1 đến T+MAX_HOLD_DAYS
        for j in range(1, MAX_HOLD_DAYS + 1):
            if i + j >= n:
                break  # Hết data của mã này

            future_price = closes[i + j]
            roi = (future_price - buy_price) / buy_price

            # Kiểm tra xem chạm tường nào trước
            if roi >= TAKE_PROFIT:
                label = 1  # Chạm tường trên (Thành công)
                break
            elif roi <= STOP_LOSS:
                label = 0  # Chạm tường dưới (Thất bại)
                break
            # Nếu chưa chạm tường nào, vòng lặp tiếp tục sang ngày hôm sau

        labels.append(label)

    group['Target_Dynamic'] = labels
    return group


# Áp dụng thuật toán cho từng cổ phiếu riêng biệt
df = df.groupby('Ticker', group_keys=False).apply(apply_triple_barrier)

# DỌN DẸP VÀ LƯU FILE
# Xóa bỏ các cột Target cũ (nếu có) để tránh mô hình bị lỗi overfitting
cols_to_drop = ['Target_T3', 'Future_Return_3D']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

# Thống kê kết quả
so_nhan_0 = (df['Target_Dynamic'] == 0).sum()
so_nhan_1 = (df['Target_Dynamic'] == 1).sum()
ty_le_nhan_1 = so_nhan_1 / len(df) * 100

print("\nĐã tái cấu trúc data thành công!")
print(f" - Số lệnh Giao dịch Thất bại (Nhãn 0): {so_nhan_0} ngày")
print(f" - Số lệnh Giao dịch Thành công (Nhãn 1): {so_nhan_1} ngày (Chiếm {ty_le_nhan_1:.2f}%)")

df.to_csv("VN30_Dynamic_Dataset.csv", index=False)
print("\nĐã xuất file mới: VN30_Dynamic_Dataset.csv")