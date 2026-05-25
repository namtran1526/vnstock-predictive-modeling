import pandas as pd
import numpy as np
import time
from vnstock import stock_historical_data
from ta import add_all_ta_features
import warnings

# Tắt cảnh báo phân mảnh DataFrame của Pandas
warnings.filterwarnings("ignore")

# 1. Danh sách VN30
vn30_tickers = [
    'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
    'MBB', 'MSN', 'MWG', 'PLX', 'POW', 'SAB', 'SHB', 'SSB', 'SSI', 'STB',
    'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE'
]

all_data = []
print("Bắt đầu khởi tạo Dataset nâng cao...")

for ticker in vn30_tickers:
    try:
        print(f"Đang trích xuất và tính toán Features cho mã: {ticker}...")

        # 1. Lấy dữ liệu thô
        df = stock_historical_data(symbol=ticker, start_date='2021-01-01', end_date='2026-05-20')

        if df.empty:
            continue

        df = df.sort_values('time')

        # Đảm bảo các cột có định dạng số (float/int) để thư viện 'ta' hoạt động
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

        # 2. BƠM >80 CHỈ BÁO KỸ THUẬT (Tự động hóa hoàn toàn)
        # Bao gồm: Momentum (RSI, MACD, TSI), Volume (OBV, Chaikin), Volatility (Bollinger, ATR), Trend (ADX, Ichimoku)
        df = add_all_ta_features(
            df, open="open", high="high", low="low", close="close", volume="volume", fillna=True
        )

        # 3. TẠO BIẾN DỰ ĐOÁN: Tầm nhìn T+3 (Thực tế hơn T+1)
        # Tính tỷ suất sinh lời sau 3 ngày tới
        df['Future_Return_3D'] = df['close'].shift(-3) / df['close'] - 1

        # Phân loại Nhãn (Target):
        # Sinh lời > 1.5% sau 3 ngày -> Nhãn 1 (Đáng để Mua)
        # Ngược lại -> Nhãn 0 (Không mua/Bán)
        df['Target_T3'] = np.where(df['Future_Return_3D'] > 0.015, 1, 0)

        # Xóa các dòng cuối cùng bị NaN do lệnh shift(-3)
        df = df.dropna(subset=['Target_T3'])

        # Thêm Metadata
        df['Ticker'] = ticker

        # Chuyển cột Ticker và Time lên đầu cho dễ nhìn
        cols = ['Ticker', 'time', 'close', 'volume', 'Target_T3'] + [c for c in df.columns if
                                                                     c not in ['Ticker', 'time', 'close', 'volume',
                                                                               'Target_T3']]
        df = df[cols]

        all_data.append(df)
        time.sleep(1)

    except Exception as e:
        print(f"Bỏ qua mã {ticker} do lỗi: {e}")

# Gộp dữ liệu
final_dataset = pd.concat(all_data, ignore_index=True)

# Lưu CSV
file_name = "VN30_Advanced_Quant_Dataset.csv"
final_dataset.to_csv(file_name, index=False)

print(f"\nThành công! Cỗ máy Dataset đã hoàn thiện với {len(final_dataset.columns)} cột (Features).")
print(f"Lưu tại: {file_name}")