import pandas as pd

# Đọc file CSV
df = pd.read_csv("VN30_Advanced_Quant_Dataset.csv")

# In ra danh sách tất cả các mã chứng khoán có trong file
danh_sach_ma = df['Ticker'].unique()
print(f"Tổng số mã đang có: {len(danh_sach_ma)} mã")
print("Danh sách chi tiết:", danh_sach_ma)