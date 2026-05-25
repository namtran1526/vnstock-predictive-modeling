import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu đã làm sạch
df = pd.read_csv("VN30_Cleaned_Dataset.csv")

# Đếm số lượng mỗi nhãn
target_counts = df['Target_T3'].value_counts()
tong_so_dong = len(df)

# Tính phần trăm
phan_tram_0 = (target_counts[0] / tong_so_dong) * 100
phan_tram_1 = (target_counts[1] / tong_so_dong) * 100

print(f"Tổng số mẫu: {tong_so_dong}")
print(f"Nhãn 0 (Giảm/Đi ngang): {target_counts[0]} mẫu ({phan_tram_0:.2f}%)")
print(f"Nhãn 1 (Tăng > 1.5%):  {target_counts[1]} mẫu ({phan_tram_1:.2f}%)")

# Vẽ biểu đồ trực quan
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Target_T3', palette=['#ff9999', '#66b3ff'])

plt.title('Phân phối Nhãn Dự đoán (Target_T3)', fontsize=14, fontweight='bold')
plt.xlabel('Nhãn (0: Không mua, 1: Mua)', fontsize=12)
plt.ylabel('Số lượng (Ngày)', fontsize=12)

# Thêm số liệu lên đầu cột
for i, count in enumerate(target_counts):
    plt.text(i, count + 50, str(count), ha='center', fontsize=12, fontweight='bold')

plt.show()