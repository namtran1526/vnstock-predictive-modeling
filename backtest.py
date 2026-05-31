import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

print("HỆ THỐNG GIẢ LẬP GIAO DỊCH THỰC CHIẾN (FINAL BACKTEST)")

# KHỞI TẠO THÔNG SỐ VÀ TẢI MÔ HÌNH
VON_BAN_DAU = 100000000.0  # 100 triệu VNĐ
TY_TRONG_LENH = 0.20  # Đi tiền 20% vốn/lệnh
PHI_MUA = 0.0015  # 0.15%
PHI_BAN = 0.0025  # 0.25% (Đã gồm 0.1% thuế)

TAKE_PROFIT = 0.04
STOP_LOSS = -0.02
MAX_HOLD = 15

try:
    test_df = pd.read_csv("VN30_Test.csv")
    model = joblib.load("xgboost_quant_model.pkl")
except FileNotFoundError:
    print("Lỗi: Thiếu file VN30_Test.csv hoặc xgboost_quant_model.pkl của model_02!")
    exit()

# Sắp xếp dữ liệu theo trình tự thời gian tăng dần để giả lập đúng lịch sử
test_df = test_df.sort_values(by='time').reset_index(drop=True)

X_test = test_df.drop(columns=['Ticker', 'time', 'close', 'Target_Dynamic'])
test_df['AI_Signal'] = model.predict(X_test)

# VẬN HÀNH VÒNG LẶP SIMULATION (GIẢ LẬP THEO THỜI GIAN)
print("\nMô hình bắt đầu tiến vào thị trường 2025-2026")

tien_mat = VON_BAN_DAU
danh_muc_dang_nam_giu = []  # Lưu các lệnh đang mở: {'Ticker', 'buy_price', 'buy_time', 'hold_days', 'capital'}
lich_su_tai_san = []
cac_ngay_giao_dich = test_df['time'].unique()

tong_so_lenh = 0
lenh_thang = 0
lenh_thua = 0

for ngay in cac_ngay_giao_dich:
    du_lieu_ngay = test_df[test_df['time'] == ngay]

    # KIỂM TRA VÀ CẬP NHẬT CÁC LỆNH ĐANG NẮM GIỮ
    cac_lenh_con_lai = []
    for lenh in danh_muc_dang_nam_giu:
        ticker = lenh['Ticker']
        gia_mua = lenh['buy_price']
        von_vao_lenh = lenh['capital']

        # Lấy giá đóng cửa hôm nay của mã này
        row_hien_tai = du_lieu_ngay[du_lieu_ngay['Ticker'] == ticker]

        if row_hien_tai.empty:
            # Nếu ngày hôm nay mã này không có dữ liệu, giữ nguyên trạng thái
            lenh['hold_days'] += 1
            cac_lenh_con_lai.append(lenh)
            continue

        gia_hien_tai = row_hien_tai['close'].values[0]
        ty_suat_loi_nhuan = (gia_hien_tai - gia_mua) / gia_mua
        lenh['hold_days'] += 1

        # Kiểm tra điều kiện thoát hàng (Triple Barrier)
        da_ban = False
        loi_nhuan_rong = 0

        if ty_suat_loi_nhuan >= TAKE_PROFIT:  # Chạm trần chốt lời
            loi_nhuan_rong = von_vao_lenh * (1 + TAKE_PROFIT) * (1 - PHI_BAN) - von_vao_lenh
            tien_mat += (von_vao_lenh + loi_nhuan_rong)
            lenh_thang += 1
            da_ban = True
        elif ty_suat_loi_nhuan <= STOP_LOSS:  # Chạm sàn cắt lỗ
            loi_nhuan_rong = von_vao_lenh * (1 + STOP_LOSS) * (1 - PHI_BAN) - von_vao_lenh
            tien_mat += (von_vao_lenh + loi_nhuan_rong)
            lenh_thua += 1
            da_ban = True
        elif lenh['hold_days'] >= MAX_HOLD:  # Hết hạn thời gian ôm hàng
            loi_nhuan_rong = von_vao_lenh * (1 + ty_suat_loi_nhuan) * (1 - PHI_BAN) - von_vao_lenh
            tien_mat += (von_vao_lenh + loi_nhuan_rong)
            if ty_suat_loi_nhuan > 0:
                lenh_thang += 1
            else:
                lenh_thua += 1
            da_ban = True

        if not da_ban:
            cac_lenh_con_lai.append(lenh)

    danh_muc_dang_nam_giu = cac_lenh_con_lai

    # BƯỚC B: QUÉT TÍN HIỆU MUA MỚI TỪ MÔ HÌNH
    # Chỉ mua nếu danh mục chưa đầy (tối đa 5 mã) và mô hình báo hiệu lệnh MUA (Signal = 1)
    tin_hieu_mua_ngay = du_lieu_ngay[du_lieu_ngay['AI_Signal'] == 1]

    for _, row in tin_hieu_mua_ngay.iterrows():
        if len(danh_muc_dang_nam_giu) >= 5:
            break  # Danh mục đã hết chỗ, bỏ qua các tín hiệu mua khác

        ticker_moi = row['Ticker']
        # Tránh mua trùng mã đang có sẵn trong danh mục
        if any(l['Ticker'] == ticker_moi for l in danh_muc_dang_nam_giu):
            continue

        # Tính toán số tiền giải ngân dựa trên tổng tài sản hiện tại
        tong_tai_san_hien_tai = tien_mat + sum(l['capital'] for l in danh_muc_dang_nam_giu)
        tien_mua = tong_tai_san_hien_tai * TY_TRONG_LENH

        if tien_mat >= tien_mua:
            tien_mat -= tien_mua
            von_sau_phi_mua = tien_mua * (1 - PHI_MUA)  # Khấu trừ phí mua ngay khi vào lệnh

            danh_muc_dang_nam_giu.append({
                'Ticker': ticker_moi,
                'buy_price': row['close'],
                'hold_days': 0,
                'capital': von_sau_phi_mua
            })
            tong_so_lenh += 1

    # GHI NHẬN TỔNG TÀI SẢN CUỐI NGÀY
    gia_tri_co_phieu_hien_tai = 0
    for lenh in danh_muc_dang_nam_giu:
        tk = lenh['Ticker']
        g_mua = lenh['buy_price']
        v_goc = lenh['capital']
        row_match = du_lieu_ngay[du_lieu_ngay['Ticker'] == tk]
        if not row_match.empty:
            g_dong_cua = row_match['close'].values[0]
            gia_tri_co_phieu_hien_tai += v_goc * (g_dong_cua / g_mua)
        else:
            gia_tri_co_phieu_hien_tai += v_goc

    tong_tai_san = tien_mat + gia_tri_co_phieu_hien_tai
    lich_su_tai_san.append({'time': ngay, 'Equity': tong_tai_san})

# TỔNG HỢP VÀ ĐỒ THỊ HÓA KẾT QUẢ
result_df = pd.DataFrame(lich_su_tai_san)
tai_san_cuoi_cung = result_df['Equity'].iloc[-1]
tong_loi_nhuan_pct = ((tai_san_cuoi_cung - VON_BAN_DAU) / VON_BAN_DAU) * 100

print("\n" + "=" * 55)
print(" 🏆 BÁO CÁO THÀNH QUẢ BACKTEST TÀI CHÍNH THỰC TẾ")
print("=" * 55)
print(f" 💰 Vốn ban đầu nạp vào:         {VON_BAN_DAU:,.0f} VNĐ")
print(f" 💵 Tổng tài sản cuối kỳ (2026): {tai_san_cuoi_cung:,.0f} VNĐ")
print(f" 📈 Tổng tỷ suất lợi nhuận ròng: {tong_loi_nhuan_pct:.2f}%")
print(f" 📊 Tổng số lệnh giải ngân:      {tong_so_lenh} lệnh")
print(
    f" 🎯 Tỷ lệ Win-rate thực tế:      {(lenh_thang / (lenh_thang + lenh_thua)) * 100:.2f}% (Bao gồm lệnh đóng hạn T+15)")

# Vẽ biểu đồ Equity Curve
plt.figure(figsize=(12, 6))
plt.plot(pd.to_datetime(result_df['time']), result_df['Equity'], label='Đường Cong Tài Sản (Equity Curve)',
         color='#11CAA0', linewidth=2.5)
plt.axhline(y=VON_BAN_DAU, color='#005088', linestyle='--', label='Vạch Vốn Ban Đầu (100tr)')
plt.title('KẾT QUẢ GIẢ LẬP TĂNG TRƯỞNG VỐN 2025-2026', fontsize=14, fontweight='bold', color='#005088')
plt.xlabel('Thời gian', fontsize=12)
plt.ylabel('Tổng tài sản (VNĐ)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.savefig('chart/BACKTEST.png', bbox_inches='tight')
plt.close()