# VN30 Predictive Modeling

Dự án xây dựng, đánh giá và backtest mô hình dự báo tín hiệu MUA trên rổ VN30 bằng dữ liệu OHLCV, chỉ báo kỹ thuật, XGBoost và nhiều biến thể thực nghiệm. Repo này ghi lại toàn bộ quá trình đi từ mô hình cơ sở T+3 đến phiên bản dùng nhãn động Triple Barrier.

> Lưu ý quan trọng: đây là dự án nghiên cứu/thực nghiệm mô hình định lượng, không phải khuyến nghị đầu tư. Kết quả phụ thuộc vào dữ liệu lịch sử, cách gán nhãn, chi phí giao dịch giả định và cấu hình mô hình tại thời điểm chạy.

## 1. Mục tiêu dự án

Mục tiêu chính là tìm một cấu hình mô hình có khả năng phát hiện các nhịp tăng có xác suất cao trên rổ VN30, đồng thời kiểm soát số lệnh MUA sai. Dự án thử nghiệm nhiều hướng cải tiến:

- Nhãn cứng T+3: dự đoán cổ phiếu tăng hơn 1.5% sau 3 ngày giao dịch.
- Tối ưu siêu tham số bằng GridSearchCV.
- Ép ngưỡng xác suất quyết định bằng `predict_proba`.
- Nâng chuẩn lợi nhuận mục tiêu lên 3%.
- Chuẩn hóa tương quan chéo bằng Cross-Sectional Ranking.
- Meta-Labeling với hệ thống hai mô hình.
- Nhãn động Triple Barrier: chốt lời +4%, cắt lỗ -2%, nắm giữ tối đa 15 ngày.
- PCA để khử đa cộng tuyến.
- Chọn lọc đặc trưng theo domain knowledge.
- Bộ lọc vĩ mô theo trạng thái VN30 mô phỏng.
- Backtest giả lập giao dịch thực tế.
- Slide thuyết trình case study tổng hợp vấn đề, phương pháp, thực nghiệm và kết luận.

Kết luận thực nghiệm hiện tại: Thực nghiệm 2 với dữ liệu nguyên bản và nhãn Triple Barrier đạt Win-rate 42.58% và được chọn làm mô hình chính để backtest.

## 2. Tài liệu thuyết trình case study

Repo có kèm file `Slide.pdf` dùng cho phần thuyết trình case study. Slide này đóng vai trò bản trình bày ngắn gọn cho người nghe, trong khi `README.md` là tài liệu kỹ thuật chi tiết để tái lập pipeline và kiểm tra từng thực nghiệm.

Nội dung thuyết trình nên được đọc theo luồng:

- Bối cảnh bài toán: dự báo tín hiệu MUA trên rổ VN30 bằng dữ liệu OHLCV và chỉ báo kỹ thuật.
- Cách xây dựng dữ liệu: tải dữ liệu, làm sạch, tạo chỉ báo, chia train/test theo thời gian.
- Mô hình baseline T+3 và các giới hạn của nhãn cứng.
- Chuỗi thực nghiệm cải tiến từ 1a đến 1e.
- Thực nghiệm Triple Barrier và lý do chọn nhãn động để mô phỏng quản trị vị thế thực tế.
- Backtest kiểm chứng khả năng sinh lời sau khi chọn mô hình tốt nhất trong phạm vi thực nghiệm.
- Kết luận case study: chất lượng định nghĩa mục tiêu quan trọng hơn việc tăng độ phức tạp mô hình.

Cách sử dụng:

- Mở `Slide.pdf` để trình bày tổng quan trong buổi báo cáo.
- Dùng các mục `Pipeline`, `Hướng dẫn chạy đúng theo từng thực nghiệm` và `Báo cáo chi tiết các thực nghiệm` trong README để giải thích sâu khi cần truy vết số liệu.
- Khi demo live, nên chạy lại đúng pipeline của Thực nghiệm 2 trước rồi mới chạy `backtest.py`, vì `VN30_Train.csv`, `VN30_Test.csv` và `xgboost_quant_model.pkl` đều có thể bị ghi đè bởi các thực nghiệm khác.

## 3. Cấu hình môi trường

### 3.1. Yêu cầu hệ thống

- Python 3.10 trở lên được khuyến nghị.
- Hệ điều hành: Windows, macOS hoặc Linux. Repo hiện đang được phát triển trên Windows.
- Kết nối internet khi chạy `dataset.py`, vì script tải dữ liệu lịch sử qua thư viện `vnstock`.
- Dung lượng trống tối thiểu khoảng 1 GB để lưu CSV, biểu đồ và model.
- CPU đa nhân được khuyến nghị vì các mô hình dùng `n_jobs=-1`.

### 3.2. Tạo môi trường ảo

Trên Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Trên macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3.3. Cài thư viện

Repo hiện chưa có `requirements.txt`, vì vậy có thể cài trực tiếp các thư viện được dùng trong code:

```bash
pip install pandas numpy vnstock ta scikit-learn xgboost matplotlib seaborn joblib
```

Các nhóm thư viện chính:

- `vnstock`: tải dữ liệu lịch sử cổ phiếu Việt Nam.
- `pandas`, `numpy`: xử lý dữ liệu bảng và tính toán.
- `ta`: sinh bộ chỉ báo kỹ thuật.
- `scikit-learn`: RandomForest, PCA, GridSearchCV, metric đánh giá.
- `xgboost`: mô hình phân loại chính.
- `matplotlib`, `seaborn`: vẽ ma trận nhầm lẫn, phân phối nhãn, equity curve.
- `joblib`: lưu và tải model `.pkl`.

### 3.4. Tạo thư mục biểu đồ nếu chưa có

Các script lưu hình vào thư mục `chart/`. Nếu clone repo mới và chưa có thư mục này, tạo trước:

```bash
mkdir chart
```

## 4. Cấu trúc thư mục và file

```text
vnstock-predictive-modeling/
├── chart/
│   ├── BACKTEST.png
│   ├── phan_phoi_nhan_T3.png
│   ├── TN1.png
│   ├── TN1a.png
│   ├── TN1b.png
│   ├── TN1c.png
│   ├── TN1d.png
│   ├── TN1e.png
│   ├── TN2.png
│   ├── TN2a.png
│   ├── TN2b.png
│   └── TN2c.png
├── dataset.py
├── data_cleaning.py
├── examine_data.py
├── feature_split.py
├── feature_split_02.py
├── model_01.py
├── model_01a.py
├── model_01b.py
├── redefine_target_01c.py
├── model_01d.py
├── model_01e.py
├── redefine_dynamic_target_02.py
├── model_02.py
├── model_02a.py
├── model_02b.py
├── model_02c.py
├── backtest.py
├── Slide.pdf
├── VN30_Advanced_Quant_Dataset.csv
├── VN30_Cleaned_Dataset.csv
├── VN30_Dynamic_Dataset.csv
├── VN30_Train.csv
├── VN30_Test.csv
├── xgboost_quant_model.pkl
└── README.md
```

### 4.1. Nhóm tạo và làm sạch dữ liệu

| File | Vai trò | Input chính | Output chính |
|---|---|---|---|
| `dataset.py` | Tải dữ liệu OHLCV VN30 từ `vnstock`, tính hơn 80 chỉ báo kỹ thuật, tạo `Future_Return_3D` và `Target_T3` với ngưỡng tăng hơn 1.5% sau T+3. | Internet/API `vnstock` | `VN30_Advanced_Quant_Dataset.csv` |
| `data_cleaning.py` | Làm sạch dữ liệu, xóa cột trùng `ticker` nếu có, thay `inf/-inf` bằng `NaN`, drop các dòng lỗi. | `VN30_Advanced_Quant_Dataset.csv` | `VN30_Cleaned_Dataset.csv` |
| `examine_data.py` | Kiểm tra phân phối nhãn T+3 và vẽ biểu đồ. | `VN30_Cleaned_Dataset.csv` | `chart/phan_phoi_nhan_T3.png` |

### 4.2. Nhóm chia tập train/test

| File | Vai trò | Input chính | Output chính |
|---|---|---|---|
| `feature_split.py` | Dùng RandomForest chọn Top 15 đặc trưng cho nhãn `Target_T3`; chia train/test theo thời gian. | `VN30_Cleaned_Dataset.csv` | `VN30_Train.csv`, `VN30_Test.csv` |
| `feature_split_02.py` | Dùng RandomForest chọn Top 15 đặc trưng cho nhãn `Target_Dynamic`; chia train/test theo thời gian. | `VN30_Dynamic_Dataset.csv` | `VN30_Train.csv`, `VN30_Test.csv` |

Mốc chia dữ liệu:

- Train: dữ liệu có `time <= 2024-12-31`.
- Test: dữ liệu có `time > 2024-12-31`, tức giai đoạn 2025-2026 theo dataset hiện tại.

### 4.3. Nhóm mô hình và thực nghiệm

| File | Thực nghiệm | Vai trò | Output |
|---|---|---|---|
| `model_01.py` | Baseline T+3 | Huấn luyện XGBoost trên nhãn `Target_T3`. | `xgboost_quant_model.pkl`, `chart/TN1.png` |
| `model_01a.py` | 1a | GridSearchCV và đặc trưng vĩ mô trung bình VN30. | `chart/TN1a.png` |
| `model_01b.py` | 1b | Siết ngưỡng xác suất từ model baseline. | `chart/TN1b.png` |
| `redefine_target_01c.py` | 1c | Ghi đè `Target_T3` thành mục tiêu tăng hơn 3%. | Cập nhật `VN30_Cleaned_Dataset.csv` |
| `model_01d.py` | 1d | Biến đổi đặc trưng thành điểm xếp hạng tương đối theo ngày. | `chart/TN1d.png` |
| `model_01e.py` | 1e | Meta-Labeling: mô hình thứ nhất tìm cơ hội, mô hình thứ hai kiểm duyệt lệnh. | `chart/TN1e.png` |
| `redefine_dynamic_target_02.py` | 2 | Tạo nhãn động Triple Barrier. | `VN30_Dynamic_Dataset.csv` |
| `model_02.py` | 2 | Huấn luyện XGBoost trên nhãn động `Target_Dynamic`. | `xgboost_quant_model.pkl`, `chart/TN2.png` |
| `model_02a.py` | 2a | Chuẩn hóa Z-score, PCA 15 thành phần, XGBoost. | `chart/TN2a.png` |
| `model_02b.py` | 2b | Chọn đặc trưng trực giao theo domain knowledge. | `chart/TN2b.png` |
| `model_02c.py` | 2c | Bộ lọc vĩ mô bằng VN30 mô phỏng và SMA 100. | `chart/TN2c.png` |
| `backtest.py` | Backtest | Giả lập giao dịch thực chiến với mô hình Triple Barrier. | `chart/BACKTEST.png` |
| `Slide.pdf` | Thuyết trình | Slide trình bày case study, tóm tắt bài toán, hướng tiếp cận, kết quả thực nghiệm và kết luận. | File PDF dùng khi báo cáo |

## 5. Các file dữ liệu chuẩn

| File | Ý nghĩa | Cách tạo lại |
|---|---|---|
| `VN30_Advanced_Quant_Dataset.csv` | Dataset thô nâng cao: OHLCV + toàn bộ chỉ báo kỹ thuật + nhãn T+3 ban đầu. | `python dataset.py` |
| `VN30_Cleaned_Dataset.csv` | Dataset đã làm sạch, dùng làm nền cho cả nhánh T+3 và Triple Barrier. | `python data_cleaning.py` |
| `VN30_Dynamic_Dataset.csv` | Dataset dùng nhãn động `Target_Dynamic`, xóa nhãn T+3 cũ khỏi bộ dữ liệu động. | `python redefine_dynamic_target_02.py` |
| `VN30_Train.csv` | Tập train đã lọc Top 15 đặc trưng. Nội dung phụ thuộc vào script split chạy gần nhất. | `python feature_split.py` hoặc `python feature_split_02.py` |
| `VN30_Test.csv` | Tập test đã lọc Top 15 đặc trưng. Nội dung phụ thuộc vào script split chạy gần nhất. | `python feature_split.py` hoặc `python feature_split_02.py` |
| `xgboost_quant_model.pkl` | Model XGBoost đã train. Nội dung phụ thuộc vào model chạy gần nhất. | `python model_01.py` hoặc `python model_02.py` |

Quy ước quan trọng:

- Nếu `VN30_Train.csv` và `VN30_Test.csv` được tạo bởi `feature_split.py`, chúng chứa cột nhãn `Target_T3`.
- Nếu được tạo bởi `feature_split_02.py`, chúng chứa cột nhãn `Target_Dynamic`.
- `model_01.py`, `model_01a.py`, `model_01b.py`, `model_01d.py`, `model_01e.py` yêu cầu dữ liệu train/test đang ở chuẩn `Target_T3`.
- `model_02.py` và `backtest.py` yêu cầu dữ liệu train/test đang ở chuẩn `Target_Dynamic`.
- `xgboost_quant_model.pkl` bị ghi đè bởi `model_01.py` và `model_02.py`. Trước khi chạy backtest theo Triple Barrier, bắt buộc chạy lại `model_02.py`.

## 6. Pipeline chạy lại từ đầu

### 6.1. Pipeline dữ liệu gốc T+3

Chạy khi muốn tái tạo dữ liệu chuẩn ban đầu cho các thực nghiệm 1, 1a, 1b, 1d, 1e:

```bash
python dataset.py
python data_cleaning.py
python examine_data.py
python feature_split.py
```

Sau pipeline này, các file chuẩn cần có:

- `VN30_Advanced_Quant_Dataset.csv`
- `VN30_Cleaned_Dataset.csv`
- `VN30_Train.csv` với `Target_T3`
- `VN30_Test.csv` với `Target_T3`
- `chart/phan_phoi_nhan_T3.png`

### 6.2. Pipeline baseline T+3

Chạy sau pipeline dữ liệu gốc T+3:

```bash
python model_01.py
```

Output cần có:

- `xgboost_quant_model.pkl`: model baseline T+3.
- `chart/TN1.png`: ma trận nhầm lẫn baseline.

Baseline T+3 ban đầu ghi nhận:

- Win-rate: 36.22%.
- TP: 644 lệnh bắt trúng.
- FP: 1134 lệnh MUA sai/đu đỉnh.

### 6.3. Pipeline Triple Barrier

Chạy khi muốn tái tạo dữ liệu và model tốt nhất hiện tại:

```bash
python dataset.py
python data_cleaning.py
python redefine_dynamic_target_02.py
python feature_split_02.py
python model_02.py
```

Output cần có:

- `VN30_Advanced_Quant_Dataset.csv`
- `VN30_Cleaned_Dataset.csv`
- `VN30_Dynamic_Dataset.csv`
- `VN30_Train.csv` với `Target_Dynamic`
- `VN30_Test.csv` với `Target_Dynamic`
- `xgboost_quant_model.pkl`: model Triple Barrier từ `model_02.py`.
- `chart/TN2.png`

Sau đó chạy backtest:

```bash
python backtest.py
```

Output backtest:

- `chart/BACKTEST.png`

## 7. Hướng dẫn chạy đúng theo từng thực nghiệm

### 7.1. Baseline T+3

Mục tiêu: xây dựng mô hình cơ sở dự đoán cổ phiếu tăng hơn 1.5% trong T+3.

Chạy:

```bash
python dataset.py
python data_cleaning.py
python feature_split.py
python model_01.py
```

File dữ liệu cần tìm sau khi chạy:

- `VN30_Cleaned_Dataset.csv`
- `VN30_Train.csv`
- `VN30_Test.csv`
- `xgboost_quant_model.pkl`
- `chart/TN1.png`

### 7.2. Thực nghiệm 1a: GridSearchCV và đặc trưng vĩ mô

Mục tiêu: dùng GridSearchCV kết hợp biến trung bình chỉ báo của rổ VN30 để tối ưu Precision.

Chạy:

```bash
python dataset.py
python data_cleaning.py
python feature_split.py
python model_01a.py
```

File cần tìm:

- `VN30_Train.csv` và `VN30_Test.csv` ở chuẩn `Target_T3`.
- `chart/TN1a.png`.

Kết quả: thất bại. Win-rate giảm nhẹ từ 36.22% xuống 36.20%. FP tăng từ 1134 lên 1322 lệnh, TP đạt 750 lệnh.

Bài học rút ra:

> Thực nghiệm cho thấy việc để máy tính tự động dò tìm siêu tham số trên dữ liệu tài chính dễ dẫn đến hiện tượng quá khớp với nhiễu. Khi được cấp thêm các biến vĩ mô, mô hình từ bỏ tính phòng thủ ban đầu để trở nên hiếu chiến hơn, dẫn đến việc báo lệnh dễ dãi. Điều này chứng minh rằng trong thị trường chứng khoán đầy rủi ro, một thuật toán phức tạp hơn không đồng nghĩa với một kết quả giao dịch tốt hơn.

### 7.3. Thực nghiệm 1b: Ép ngưỡng xác suất quyết định

Mục tiêu: dùng `predict_proba()` từ model baseline và chỉ báo MUA khi xác suất vượt ngưỡng 0.54.

Chạy:

```bash
python dataset.py
python data_cleaning.py
python feature_split.py
python model_01.py
python model_01b.py
```

Lý do phải chạy `model_01.py` trước: `model_01b.py` cần tải `xgboost_quant_model.pkl` của baseline T+3.

File cần tìm:

- `xgboost_quant_model.pkl`
- `VN30_Test.csv` ở chuẩn `Target_T3`
- `chart/TN1b.png`

Kết quả: thất bại về mặt thanh khoản thực chiến. Win-rate tăng lên 38.55%, FP giảm còn 711 lệnh, nhưng TP giảm mạnh còn 446 lệnh.

Bài học rút ra:

> Thực nghiệm này minh họa rõ nét bài toán đánh đổi giữa độ chính xác và độ phủ. Mặc dù việc nâng ngưỡng quyết định giúp bảo vệ vốn, nó bóp nghẹt thanh khoản của hệ thống, khiến tần suất giao dịch sụt giảm và không đủ để sinh ra lợi nhuận kép bù đắp chi phí cơ hội.

### 7.4. Thực nghiệm 1c: Nâng chuẩn đầu ra lên 3%

Mục tiêu: thay vì tìm nhịp tăng 1.5% trong T+3, nâng mục tiêu lên 3% để tìm các mẫu hình tăng trưởng bùng nổ.

Chạy:

```bash
python dataset.py
python data_cleaning.py
python redefine_target_01c.py
python feature_split.py
python model_01.py
```

Cảnh báo: `redefine_target_01c.py` ghi đè trực tiếp cột `Target_T3` trong `VN30_Cleaned_Dataset.csv`. Nếu muốn quay lại nhãn 1.5%, chạy lại:

```bash
python dataset.py
python data_cleaning.py
python feature_split.py
```

File cần tìm:

- `VN30_Cleaned_Dataset.csv` đã bị cập nhật `Target_T3` theo ngưỡng 3%.
- `VN30_Train.csv`
- `VN30_Test.csv`
- `chart/TN1.png` nếu dùng `model_01.py`.
- `chart/TN1c.png` nếu có phiên bản script/biểu đồ 1c được chạy riêng.

Kết quả: thất bại nặng. Win-rate giảm xuống 28.79%, TP 241 và FP 596.

Bài học rút ra:

> Thực nghiệm nâng mục tiêu lợi nhuận lên 3% để lọc siêu cổ phiếu đã làm giảm Precision xuống mức 28.79%. Điều này phản ánh bản chất cấu trúc của rổ VN30: đây là các cổ phiếu vốn hóa siêu lớn, việc giá bật tăng đột biến hơn 3% chỉ trong 3 ngày là những sự kiện dị biệt, thường do tin tức vĩ mô dẫn dắt chứ không tuân theo quy luật của các chỉ báo kỹ thuật truyền thống.

### 7.5. Thực nghiệm 1d: Cross-Sectional Ranking

Mục tiêu: chuyển các chỉ báo tuyệt đối thành điểm xếp hạng tương đối từ 0.0 đến 1.0 trong cùng một ngày giao dịch, nhằm tìm cổ phiếu mạnh nhất dòng tiền.

Chạy:

```bash
python dataset.py
python data_cleaning.py
python feature_split.py
python model_01d.py
```

File cần tìm:

- `VN30_Train.csv` và `VN30_Test.csv` ở chuẩn `Target_T3`.
- `chart/TN1d.png`.

Kết quả: thất bại. Win-rate giảm xuống 33.74%, FP tăng mạnh lên 2268 lệnh, TP đạt 1155 lệnh.

Bài học rút ra:

> Thực nghiệm này vấp phải một cái bẫy kinh điển trong phân tích dòng tiền: kẻ mạnh nhất trong một thị trường sụp đổ vẫn là kẻ thua cuộc. Việc bình chuẩn hóa xếp hạng đã vô tình xóa sổ hoàn toàn bức tranh toàn cảnh. Mô hình bị ép phải chọn ra những mã đẹp nhất mỗi ngày, kể cả vào những ngày thị trường hoảng loạn bán tháo.

### 7.6. Thực nghiệm 1e: Meta-Labeling

Mục tiêu: triển khai hệ thống hai mô hình. Mô hình thứ nhất phát hiện cơ hội, mô hình thứ hai kiểm duyệt và từ chối các lệnh rủi ro cao.

Chạy:

```bash
python dataset.py
python data_cleaning.py
python feature_split.py
python model_01.py
python model_01e.py
```

Lý do phải chạy `model_01.py` trước: `model_01e.py` tải model baseline từ `xgboost_quant_model.pkl`.

File cần tìm:

- `xgboost_quant_model.pkl` của baseline T+3.
- `VN30_Train.csv` và `VN30_Test.csv` ở chuẩn `Target_T3`.
- `chart/TN1e.png`.

Kết quả: Win-rate cải thiện nhẹ lên 37.84%. FP giảm xuống 662 lệnh, nhưng TP chỉ còn 403 lệnh.

Bài học rút ra:

> Mặc dù việc triển khai hệ thống hai mô hình tốn kém tài nguyên tính toán, Win-rate vẫn không thể vượt mốc 40%. Sự chững lại về mặt hiệu suất này là minh chứng cho việc hệ thống đã chạm đến trần thông tin của khung thời gian cứng T+3. Độ nhiễu siêu ngắn hạn mang nặng tính ngẫu nhiên khiến thuật toán dù phức tạp đến mấy cũng không thể bù đắp được khuyết điểm của dữ liệu gốc.

### 7.7. Thực nghiệm 2: Triple Barrier Method

Mục tiêu: bỏ khung T+3 cứng và mô phỏng quản trị vị thế thực tế bằng 3 bức tường:

- Take Profit: +4%.
- Stop Loss: -2%.
- Max Hold: 15 ngày giao dịch.

Chạy:

```bash
python dataset.py
python data_cleaning.py
python redefine_dynamic_target_02.py
python feature_split_02.py
python model_02.py
```

File cần tìm:

- `VN30_Dynamic_Dataset.csv`
- `VN30_Train.csv` với `Target_Dynamic`
- `VN30_Test.csv` với `Target_Dynamic`
- `xgboost_quant_model.pkl` của mô hình Triple Barrier
- `chart/TN2.png`

Kết quả: bước đột phá. Win-rate tăng lên 42.58%, vượt xa baseline 36.22%. FP giảm còn 677 lệnh, TP đạt 502 lệnh.

Hạn chế: Recall giảm xuống khoảng 13%. Mô hình trở nên rất khắt khe và từ chối phần lớn cơ hội. Nguyên nhân cốt lõi được xác định là hiện tượng đa cộng tuyến giữa các chỉ báo kỹ thuật đầu vào, khiến mô hình chọn giải pháp an toàn là đứng ngoài thị trường.

### 7.8. Thực nghiệm 2a: PCA khử đa cộng tuyến

Mục tiêu: giải quyết tình trạng recall thấp của Triple Barrier bằng PCA, nén dữ liệu thành 15 siêu biến độc lập.

Chạy:

```bash
python dataset.py
python data_cleaning.py
python redefine_dynamic_target_02.py
python model_02a.py
```

Lưu ý: `model_02a.py` đọc trực tiếp `VN30_Dynamic_Dataset.csv`, tự chia train/test bên trong script, không cần chạy `feature_split_02.py`.

File cần tìm:

- `VN30_Dynamic_Dataset.csv`
- `chart/TN2a.png`

Kết quả: PCA tăng Recall lên khoảng 26.5% và TP lên 1030 nhịp, nhưng FP tăng lên 1585 lệnh. Win-rate giảm về 39.39%.

Bài học rút ra:

> Việc sử dụng PCA để khử đa cộng tuyến gây ra tác dụng phụ: các siêu biến toán học làm mất đi ý nghĩa vật lý gốc của các chỉ báo tài chính, ví dụ vùng quá mua của RSI. Dưới góc độ quản trị rủi ro thực chiến, việc hi sinh Precision để đổi lấy Recall là một sự đánh đổi không an toàn.

### 7.9. Thực nghiệm 2b: Chọn lọc đặc trưng theo domain knowledge

Mục tiêu: thay PCA bằng chọn lọc chỉ báo có ý nghĩa tài chính, gồm 4 nhóm: xu hướng, động lượng, biến động và dòng tiền.

Chạy:

```bash
python dataset.py
python data_cleaning.py
python redefine_dynamic_target_02.py
python model_02b.py
```

Lưu ý: `model_02b.py` đọc trực tiếp `VN30_Dynamic_Dataset.csv`, tự chọn feature và tự chia train/test.

File cần tìm:

- `VN30_Dynamic_Dataset.csv`
- `chart/TN2b.png`

Kết quả: thất bại trong việc duy trì tỷ lệ thắng. Win-rate giảm xuống 39.50%, TP 670 và FP 1026.

Bài học rút ra:

> Thực nghiệm chứng minh việc áp đặt định kiến của con người vào chọn lọc biến đầu vào không mang lại hiệu quả cao hơn thuật toán tự học. Khi bị ép nhìn vào các chỉ báo dòng tiền hay biến động, vốn chứa nhiều nhiễu trên VN30, mô hình bị phân tâm khỏi các tín hiệu cốt lõi.

### 7.10. Thực nghiệm 2c: Macro Regime Filter

Mục tiêu: chỉ cho phép mô hình phát tín hiệu MUA khi VN30 mô phỏng đang trong pha uptrend, xác định bằng giá trung bình rổ VN30 lớn hơn SMA 100.

Chạy:

```bash
python dataset.py
python data_cleaning.py
python redefine_dynamic_target_02.py
python model_02c.py
```

Lưu ý: `model_02c.py` đọc trực tiếp `VN30_Dynamic_Dataset.csv`, tự tạo biến `Macro_Uptrend`, tự chia train/test và tự áp bộ lọc.

File cần tìm:

- `VN30_Dynamic_Dataset.csv`
- `chart/TN2c.png`

Kết quả: thất bại. FP giảm xuống 479 lệnh, nhưng TP cũng giảm mạnh còn 315 lệnh. Win-rate về 39.67%.

Bài học rút ra:

> Thực nghiệm minh chứng cho cái bẫy độ trễ kinh điển trong phân tích kỹ thuật. Việc sử dụng SMA 100 làm tấm khiên chắn khiến hệ thống mất đi sự linh hoạt. Mô hình bị ép phải mua đuổi khi xu hướng đã quá rõ ràng và bị cấm giải ngân ở những nhịp điều chỉnh sâu.

## 8. Backtest mô hình Triple Barrier

Backtest chỉ nên chạy sau khi đã tạo đúng model Triple Barrier từ Thực nghiệm 2:

```bash
python dataset.py
python data_cleaning.py
python redefine_dynamic_target_02.py
python feature_split_02.py
python model_02.py
python backtest.py
```

`backtest.py` giả lập các điều kiện:

- Vốn ban đầu: 100,000,000 VND.
- Tỷ trọng mỗi lệnh: 20% tổng tài sản.
- Tối đa 5 mã đang nắm giữ.
- Phí mua: 0.15%.
- Phí bán: 0.25%, đã gồm thuế.
- Take Profit: +4%.
- Stop Loss: -2%.
- Max Hold: 15 ngày.

File đầu vào bắt buộc:

- `VN30_Test.csv` phải là bản được tạo bởi `feature_split_02.py`, tức có `Target_Dynamic`.
- `xgboost_quant_model.pkl` phải là model được tạo bởi `model_02.py`, không phải model T+3 từ `model_01.py`.

Output:

- `chart/BACKTEST.png`
- Báo cáo tổng tài sản cuối kỳ, tổng lợi nhuận ròng, tổng số lệnh và Win-rate thực tế được in ra terminal.

## 9. Bảng tổng hợp kết quả thực nghiệm

Ghi chú mốc so sánh: mô hình cơ sở Baseline T+3 ban đầu ghi nhận Win-rate 36.22%, với 1134 lệnh báo MUA sai và 644 lệnh bắt trúng.

| Thực nghiệm | Ý tưởng | Win-rate | TP | FP | Kết luận |
|---|---:|---:|---:|---:|---|
| Baseline T+3 | XGBoost, Target T+3 > 1.5% | 36.22% | 644 | 1134 | Mốc so sánh ban đầu |
| 1a | GridSearchCV + biến vĩ mô | 36.20% | 750 | 1322 | Thất bại, tăng overfitting và báo lệnh dễ dãi |
| 1b | Ép ngưỡng xác suất | 38.55% | 446 | 711 | Win-rate tăng nhưng thanh khoản bị bóp nghẹt |
| 1c | Target T+3 > 3% | 28.79% | 241 | 596 | Thất bại nặng do sự kiện tăng >3% quá dị biệt với VN30 |
| 1d | Cross-Sectional Ranking | 33.74% | 1155 | 2268 | Thất bại vì mất bối cảnh absolute momentum |
| 1e | Meta-Labeling | 37.84% | 403 | 662 | Cải thiện nhẹ nhưng vẫn chạm trần thông tin T+3 |
| 2 | Triple Barrier | 42.58% | 502 | 677 | Tốt nhất trong các thực nghiệm hiện tại |
| 2a | PCA | 39.39% | 1030 | 1585 | Recall tăng nhưng Precision giảm không an toàn |
| 2b | Domain feature selection | 39.50% | 670 | 1026 | Human bias không vượt thuật toán tự chọn |
| 2c | Macro regime filter | 39.67% | 315 | 479 | Bộ lọc SMA 100 bị trễ và bỏ lỡ vùng đáy |

## 10. Báo cáo chi tiết các thực nghiệm

### 10.1. Mô hình cơ sở: Baseline T+3

Mô hình cơ sở sử dụng XGBoost trên tập dữ liệu đã được lọc Top 15 chỉ báo kỹ thuật, với nhãn `Target_T3 = 1` khi cổ phiếu tăng hơn 1.5% sau 3 ngày giao dịch. Đây là mốc chuẩn để đánh giá toàn bộ các cải tiến phía sau.

Kết quả baseline:

- Win-rate: 36.22%.
- False Positives: 1134 lệnh MUA sai/đu đỉnh.
- True Positives: 644 lệnh bắt trúng.
- File chạy: `model_01.py`.
- Biểu đồ: `chart/TN1.png`.

Ý nghĩa: baseline có khả năng tìm được một phần các nhịp tăng ngắn hạn, nhưng số lệnh sai còn lớn. Đây là vấn đề chính cần tối ưu trong các thực nghiệm sau: tăng Precision mà không bóp nghẹt số lượng cơ hội giao dịch.

### 10.2. Thực nghiệm 1a: Tối ưu siêu tham số và bổ sung đặc trưng vĩ mô

Mục tiêu: sử dụng GridSearchCV kết hợp tạo thêm biến trung bình chỉ báo của toàn rổ VN30. Ý tưởng là giúp mô hình có thêm góc nhìn vĩ mô, thay vì chỉ nhìn từng cổ phiếu riêng lẻ. GridSearchCV được dùng để dò tìm cấu hình XGBoost tốt hơn, ưu tiên tối ưu tỷ lệ lệnh MUA đúng.

Luồng chạy:

```bash
python dataset.py
python data_cleaning.py
python feature_split.py
python model_01a.py
```

Cơ chế chính trong code:

- Đọc `VN30_Train.csv` và `VN30_Test.csv` ở chuẩn `Target_T3`.
- Gộp train/test tạm thời để tạo đồng bộ các biến vĩ mô.
- Tính trung bình theo ngày của 2 chỉ báo đứng đầu trong bộ Top 15.
- Thêm các biến dạng `market_avg_<indicator>`.
- Dùng `GridSearchCV` với các tham số `n_estimators`, `max_depth`, `learning_rate`.
- Chấm điểm theo `precision`.

Kết quả: thất bại. Win-rate gần như đi ngang và giảm nhẹ từ 36.22% xuống 36.20%. Số lệnh MUA sai tăng mạnh từ 1134 lên 1322 lệnh, dù số lệnh bắt trúng tăng lên 750.

Diễn giải: mô hình trở nên hiếu chiến hơn. Việc thêm biến vĩ mô và dò siêu tham số không giúp mô hình phòng thủ tốt hơn, mà làm nó phát tín hiệu MUA dễ dãi hơn. Precision không cải thiện vì phần tăng TP không bù được phần tăng FP.

Bài học rút ra:

> Thực nghiệm cho thấy việc để máy tính tự động dò tìm siêu tham số trên dữ liệu tài chính dễ dẫn đến hiện tượng quá khớp với nhiễu. Khi được cấp thêm các biến vĩ mô, mô hình từ bỏ tính phòng thủ ban đầu để trở nên hiếu chiến hơn, dẫn đến việc báo lệnh dễ dãi. Điều này chứng minh rằng trong thị trường chứng khoán đầy rủi ro, một thuật toán phức tạp hơn không đồng nghĩa với một kết quả giao dịch tốt hơn, đúng với nguyên lý Occam's Razor.

### 10.3. Thực nghiệm 1b: Ép ngưỡng xác suất quyết định

Mục tiêu: thay vì dùng nhãn dự đoán mặc định từ `predict()`, mô hình lấy xác suất bằng `predict_proba()` và chỉ phát lệnh MUA khi xác suất vượt ngưỡng tự tin. Trong code hiện tại, ngưỡng được đặt là `0.54`.

Luồng chạy:

```bash
python dataset.py
python data_cleaning.py
python feature_split.py
python model_01.py
python model_01b.py
```

Cơ chế chính trong code:

- `model_01.py` tạo model baseline và lưu `xgboost_quant_model.pkl`.
- `model_01b.py` tải lại model baseline.
- Lấy xác suất lớp MUA bằng `model.predict_proba(X_test)[:, 1]`.
- Chỉ chuyển thành lệnh MUA nếu xác suất lớn hơn hoặc bằng `0.54`.

Kết quả: thất bại về mặt thanh khoản thực chiến. Win-rate tăng lên 38.55%, FP giảm còn 711 lệnh, nhưng TP giảm nghiêm trọng còn 446 lệnh.

Diễn giải: mô hình bớt sai hơn vì bị siết ngưỡng, nhưng cũng bỏ qua quá nhiều cơ hội đúng. Tỷ lệ thắng tăng không đủ để bù lại sự suy giảm tần suất giao dịch và số nhịp tăng bắt được.

Bài học rút ra:

> Thực nghiệm này minh họa rõ nét bài toán đánh đổi giữa độ chính xác và độ phủ. Mặc dù việc nâng ngưỡng quyết định giúp bảo vệ vốn, nó bóp nghẹt thanh khoản của hệ thống, khiến tần suất giao dịch sụt giảm và không đủ để sinh ra lợi nhuận kép bù đắp chi phí cơ hội.

### 10.4. Thực nghiệm 1c: Nâng chuẩn đầu ra và tìm kiếm siêu cổ phiếu

Mục tiêu: thay vì huấn luyện mô hình tìm các nhịp tăng 1.5% trong T+3, nâng kỳ vọng lên 3% trong T+3 để mô hình chỉ học các mẫu hình tăng trưởng mạnh, rõ ràng hơn và ít nhiễu hơn.

Luồng chạy:

```bash
python dataset.py
python data_cleaning.py
python redefine_target_01c.py
python feature_split.py
python model_01.py
```

Cơ chế chính trong code:

- `redefine_target_01c.py` đọc `VN30_Cleaned_Dataset.csv`.
- Xóa `Target_T3` cũ nếu có.
- Tạo lại `Target_T3 = 1` khi `Future_Return_3D > 0.03`.
- Ghi đè trực tiếp vào `VN30_Cleaned_Dataset.csv`.
- Chạy lại `feature_split.py` để tạo `VN30_Train.csv` và `VN30_Test.csv` theo target mới.

Cảnh báo vận hành: đây là thực nghiệm có ghi đè nhãn trong `VN30_Cleaned_Dataset.csv`. Muốn quay về chuẩn 1.5%, cần chạy lại `dataset.py`, `data_cleaning.py` và `feature_split.py`.

Kết quả: thất bại nặng. Win-rate rơi xuống 28.79%, TP chỉ còn 241 và FP là 596.

Diễn giải: nhãn 3% làm lớp dương trở nên hiếm và khó học hơn. Với rổ VN30, các nhịp tăng trên 3% trong 3 ngày thường không phải mẫu hình kỹ thuật lặp lại ổn định, mà là các sự kiện dị biệt.

Bài học rút ra:

> Thực nghiệm nâng mục tiêu lợi nhuận lên 3% để lọc siêu cổ phiếu đã làm giảm Precision xuống mức 28.79%. Điều này phản ánh bản chất cấu trúc của rổ VN30: đây là các cổ phiếu vốn hóa siêu lớn, việc giá bật tăng đột biến hơn 3% chỉ trong 3 ngày là những sự kiện dị biệt, thường do tin tức vĩ mô dẫn dắt chứ không tuân theo quy luật của các chỉ báo kỹ thuật truyền thống.

### 10.5. Thực nghiệm 1d: Biến đổi dữ liệu theo điểm xếp hạng

Mục tiêu: loại bỏ nhiễu từ xu hướng chung bằng cách không dùng giá trị tuyệt đối của chỉ báo, ví dụ RSI = 60, mà chuyển thành điểm xếp hạng tương đối trong cùng ngày giao dịch. Mỗi cổ phiếu được xếp hạng so với 29 mã còn lại trong rổ VN30, với điểm từ 0.0 đến 1.0.

Luồng chạy:

```bash
python dataset.py
python data_cleaning.py
python feature_split.py
python model_01d.py
```

Cơ chế chính trong code:

- Đọc `VN30_Train.csv` và `VN30_Test.csv` ở chuẩn `Target_T3`.
- Với từng feature, dùng `groupby('time')[col].rank(pct=True)`.
- Huấn luyện lại XGBoost trên không gian feature đã được ranking.

Kết quả: thất bại. Win-rate giảm sâu xuống 33.74%. Mô hình trở nên cực kỳ hiếu chiến, FP tăng lên 2268 lệnh, TP tăng lên 1155 lệnh.

Diễn giải: ranking làm mô hình luôn nhìn thấy cổ phiếu mạnh nhất tương đối trong ngày, nhưng đánh mất thông tin thị trường chung đang khỏe hay yếu. Trong một ngày toàn thị trường xấu, cổ phiếu đứng hạng cao nhất vẫn có thể là một lựa chọn thua lỗ.

Bài học rút ra:

> Thực nghiệm này vấp phải một cái bẫy kinh điển trong phân tích dòng tiền: kẻ mạnh nhất trong một thị trường sụp đổ vẫn là kẻ thua cuộc. Việc bình chuẩn hóa xếp hạng đã vô tình xóa sổ hoàn toàn bức tranh toàn cảnh, hay absolute momentum. Mô hình bị ép phải chọn ra những mã đẹp nhất mỗi ngày, kể cả vào những ngày thị trường hoảng loạn bán tháo.

### 10.6. Thực nghiệm 1e: Hệ thống hai mô hình

Mục tiêu: áp dụng Meta-Labeling theo tư duy của Marcos Lopez de Prado. Mô hình thứ nhất đóng vai trò tìm cơ hội, mô hình thứ hai đóng vai trò kiểm duyệt các lệnh MUA do mô hình thứ nhất đề xuất, từ đó cố gắng loại bỏ các lệnh rủi ro cao.

Luồng chạy:

```bash
python dataset.py
python data_cleaning.py
python feature_split.py
python model_01.py
python model_01e.py
```

Cơ chế chính trong code:

- `model_01.py` tạo model baseline.
- `model_01e.py` tải lại `xgboost_quant_model.pkl`.
- Dùng `cross_val_predict` để mô hình thứ nhất dự đoán trên tập train một cách khách quan hơn.
- Chỉ lấy các dòng mà mô hình thứ nhất báo MUA để tạo tập train cho mô hình thứ hai.
- Mô hình thứ hai học cách phân biệt lệnh MUA nào nên chấp thuận, lệnh nào nên bác bỏ.
- Trên test, mô hình thứ nhất quét toàn thị trường trước, mô hình thứ hai kiểm duyệt từng lệnh MUA.

Kết quả: Win-rate chỉ cải thiện nhẹ lên 37.84%. FP giảm xuống 662 lệnh, nhưng TP cũng giảm xuống 403 lệnh.

Diễn giải: mô hình thứ hai có tác dụng cắt bớt lệnh sai, nhưng cũng cắt luôn nhiều lệnh đúng. Hệ thống hai mô hình phức tạp hơn nhưng không vượt qua được vùng 40% Win-rate.

Bài học rút ra:

> Mặc dù việc triển khai hệ thống hai mô hình tốn kém tài nguyên tính toán, Win-rate vẫn không thể vượt mốc 40%. Sự chững lại về mặt hiệu suất này là minh chứng rõ ràng cho việc hệ thống đã chạm đến trần thông tin của khung thời gian cứng T+3. Độ nhiễu siêu ngắn hạn mang nặng tính ngẫu nhiên khiến thuật toán dù phức tạp đến mấy cũng không thể bù đắp được khuyết điểm của dữ liệu gốc.

### 10.7. Thực nghiệm 2: Thay nhãn cứng bằng Triple Barrier Method

Mục tiêu: loại bỏ tư duy dự đoán khung thời gian cứng T+3 và thay bằng nhãn động mô phỏng quản trị vị thế thực tế. Một lệnh được xem là thành công nếu chạm Take Profit trước Stop Loss trong thời gian nắm giữ tối đa.

Thông số Triple Barrier:

- Take Profit: +4%.
- Stop Loss: -2%.
- Max Hold: 15 ngày giao dịch.
- Risk:Reward: 1:2.

Luồng chạy:

```bash
python dataset.py
python data_cleaning.py
python redefine_dynamic_target_02.py
python feature_split_02.py
python model_02.py
```

Cơ chế chính trong code:

- `redefine_dynamic_target_02.py` đọc `VN30_Cleaned_Dataset.csv`.
- Sắp xếp theo `Ticker` và `time`.
- Với từng cổ phiếu, quét tương lai từ T+1 đến T+15.
- Gán `Target_Dynamic = 1` nếu giá chạm +4% trước.
- Gán `Target_Dynamic = 0` nếu giá chạm -2% trước hoặc hết thời gian mà không đạt mục tiêu.
- Xóa các cột `Target_T3` và `Future_Return_3D` khỏi dataset động.
- `feature_split_02.py` chọn Top 15 feature theo `Target_Dynamic`.
- `model_02.py` huấn luyện XGBoost trên nhãn động Triple Barrier.

Kết quả: bước đột phá. Win-rate tăng lên 42.58%, vượt xa baseline 36.22%. FP giảm còn 677 lệnh, TP đạt 502 lệnh.

Hạn chế tồn tại: Recall giảm xuống khoảng 13%. Mô hình trở nên rất khắt khe và từ chối phần lớn cơ hội trên thị trường. Nguyên nhân cốt lõi được xác định là hiện tượng đa cộng tuyến giữa 15 chỉ báo kỹ thuật đầu vào, khiến mô hình chọn giải pháp an toàn là đứng ngoài thị trường.

Diễn giải: thay đổi cách gán nhãn tạo ra cải thiện lớn hơn mọi tinh chỉnh mô hình trước đó. Điều này cho thấy vấn đề chính không nằm ở XGBoost, mà nằm ở định nghĩa mục tiêu học. Triple Barrier gần hơn với hành vi giao dịch thực tế nên tín hiệu học được có chất lượng cao hơn.

### 10.8. Thực nghiệm 2a: Khử đa cộng tuyến bằng PCA

Mục tiêu: xử lý tình trạng mô hình Triple Barrier quá thận trọng và Recall chỉ khoảng 13%. PCA được dùng để nén không gian chỉ báo kỹ thuật thành 15 thành phần chính độc lập hơn, kỳ vọng giảm đa cộng tuyến và giải phóng thanh khoản.

Luồng chạy:

```bash
python dataset.py
python data_cleaning.py
python redefine_dynamic_target_02.py
python model_02a.py
```

Cơ chế chính trong code:

- Đọc trực tiếp `VN30_Dynamic_Dataset.csv`.
- Tự chia train/test theo mốc `2025-01-01`.
- Chuẩn hóa feature bằng `StandardScaler`.
- Fit scaler trên train và chỉ transform test để tránh data leakage.
- Dùng `PCA(n_components=15)`.
- Huấn luyện XGBoost trên 15 thành phần PCA.

Kết quả: PCA giải phóng thanh khoản và tăng Recall lên khoảng 26.5%. TP tăng lên 1030, nhưng FP cũng tăng mạnh lên 1585. Win-rate giảm về 39.39%.

Diễn giải: PCA giúp mô hình báo MUA nhiều hơn nhưng làm mất nghĩa tài chính trực tiếp của chỉ báo. Khi không còn hiểu rõ vùng quá mua, quá bán, động lượng hay biến động theo nghĩa gốc, mô hình tăng độ phủ bằng cách chấp nhận nhiều tín hiệu nhiễu hơn.

Bài học rút ra:

> Việc sử dụng PCA để khử đa cộng tuyến gây ra tác dụng phụ: các siêu biến toán học làm mất đi ý nghĩa vật lý gốc của các chỉ báo tài chính, ví dụ vùng quá mua của RSI. Dưới góc độ quản trị rủi ro thực chiến, việc hi sinh Precision để đổi lấy Recall là một sự đánh đổi không an toàn.

### 10.9. Thực nghiệm 2b: Chọn lọc đặc trưng trực giao theo domain knowledge

Mục tiêu: khắc phục nhược điểm mất ý nghĩa tài chính của PCA. Thay vì nén toán học, thực nghiệm áp đặt tư duy con người để chọn một bộ chỉ báo cân bằng từ 4 nhóm: xu hướng, động lượng, biến động và dòng tiền.

Luồng chạy:

```bash
python dataset.py
python data_cleaning.py
python redefine_dynamic_target_02.py
python model_02b.py
```

Cơ chế chính trong code:

- Đọc trực tiếp `VN30_Dynamic_Dataset.csv`.
- Tự phân nhóm feature bằng tên cột.
- Nhóm xu hướng: `macd`, `sma`, `ema`, `trend`, `adx`.
- Nhóm động lượng: `rsi`, `stoch`, `cci`, `mfi`, `mom`, `roc`.
- Nhóm biến động: `bb`, `atr`, `std`, `band`, `kc`.
- Nhóm dòng tiền/khối lượng: `vol`, `obv`, `cmf`, `vwap`, `adi`.
- Lấy tối đa 4 chỉ báo đại diện mỗi nhóm rồi huấn luyện XGBoost.

Kết quả: thất bại trong việc duy trì tỷ lệ thắng. Win-rate giảm xuống 39.50%, TP đạt 670 và FP tăng lên 1026.

Diễn giải: chọn feature theo trực giác tài chính giúp giữ nghĩa của chỉ báo, nhưng không đảm bảo tối ưu trên dữ liệu VN30. Một số nhóm như dòng tiền và biến động có thể chứa nhiều nhiễu trong khung dữ liệu này, làm mô hình phân tâm khỏi tín hiệu cốt lõi.

Bài học rút ra:

> Thực nghiệm chứng minh việc áp đặt định kiến của con người vào chọn lọc biến đầu vào không mang lại hiệu quả cao hơn thuật toán tự học. Khi bị ép nhìn vào các chỉ báo dòng tiền hay biến động, vốn chứa nhiều nhiễu trên VN30, mô hình bị phân tâm khỏi các tín hiệu cốt lõi.

### 10.10. Thực nghiệm 2c: Tích hợp bộ lọc vĩ mô

Mục tiêu: vượt ngưỡng Win-rate của mô hình Triple Barrier bằng cách thêm lăng kính thị trường chung. Mô hình chỉ được phép khớp lệnh MUA khi chỉ số VN30 mô phỏng đang ở pha Uptrend, được xác định bằng điều kiện giá trung bình rổ VN30 lớn hơn SMA 100.

Luồng chạy:

```bash
python dataset.py
python data_cleaning.py
python redefine_dynamic_target_02.py
python model_02c.py
```

Cơ chế chính trong code:

- Đọc trực tiếp `VN30_Dynamic_Dataset.csv`.
- Tạo chỉ số thị trường mô phỏng bằng trung bình giá đóng cửa toàn rổ theo từng ngày.
- Tính `SMA_100` của chỉ số mô phỏng.
- Tạo biến `Macro_Uptrend = 1` nếu `Market_Index > SMA_100`.
- Huấn luyện mô hình kỹ thuật như bình thường.
- Khi dự đoán test, chỉ giữ lệnh MUA nếu `ai_raw_predictions & macro_shield_test` bằng 1.

Kết quả: thất bại. FP giảm mạnh xuống 479 lệnh, nhưng TP cũng rơi xuống 315 lệnh. Win-rate giảm về 39.67%.

Diễn giải: bộ lọc vĩ mô làm đúng nhiệm vụ phòng thủ nhưng quá trễ. SMA 100 chỉ xác nhận xu hướng sau khi thị trường đã hồi phục rõ, vì vậy mô hình bị cấm mua ở các vùng đáy hoảng loạn và dễ bị cho phép mua khi xu hướng đã muộn.

Bài học rút ra:

> Thực nghiệm minh chứng cho cái bẫy độ trễ kinh điển trong phân tích kỹ thuật. Việc sử dụng SMA 100 làm tấm khiên chắn khiến hệ thống mất đi sự linh hoạt. Mô hình bị ép phải mua đuổi khi xu hướng đã quá rõ ràng, thường là đỉnh ngắn hạn, và bị cấm giải ngân ở những nhịp điều chỉnh sâu, tức vùng đáy hoảng loạn.

## 11. Kết luận toàn tập

Sau 9 vòng thực nghiệm bao quát các hướng tối ưu từ siêu tham số, xếp hạng tương đối, hệ thống hai mô hình, nén chiều không gian bằng PCA cho đến bộ lọc vĩ mô, kết luận hiện tại là:

> Thực nghiệm 2 với dữ liệu nguyên bản và nhãn Triple Barrier đạt Win-rate 42.58% cùng Risk:Reward 1:2. Đây là cấu hình tốt nhất trong phạm vi các thực nghiệm hiện tại trên tập dữ liệu OHLCV của rổ VN30. Thay vì tiếp tục vắt kiệt dữ liệu và tăng nguy cơ overfitting, mô hình Triple Barrier được dùng để tiến hành backtest kiểm chứng khả năng sinh lời thực tế.

Nói cách khác, thay đổi có giá trị nhất không phải là làm mô hình phức tạp hơn, mà là định nghĩa lại nhãn mục tiêu sao cho gần với hành vi giao dịch thực tế hơn. Các cải tiến sau Thực nghiệm 2 đều cho thấy một dạng đánh đổi không mong muốn: tăng số lệnh nhưng giảm Precision, hoặc giảm FP nhưng đồng thời bóp nghẹt TP.

Kết luận thực dụng cho giai đoạn tiếp theo:

- Dùng mô hình Triple Barrier từ Thực nghiệm 2 làm mô hình chính.
- Không tiếp tục tối ưu chỉ để tăng chỉ số trên ma trận nhầm lẫn nếu không cải thiện backtest.
- Đánh giá mô hình bằng lợi nhuận ròng, drawdown, số lệnh, phí giao dịch và độ ổn định theo thời gian.
- Nếu mở rộng nghiên cứu, ưu tiên thêm nguồn dữ liệu mới có thông tin thật sự khác biệt như dữ liệu thị trường phái sinh, tin tức, dòng tiền tổ chức hoặc biến vĩ mô chất lượng cao.

## 12. Cảnh báo vận hành

- Không chạy lẫn nhánh T+3 và nhánh Triple Barrier mà không tạo lại `VN30_Train.csv` và `VN30_Test.csv`.
- `redefine_target_01c.py` ghi đè `VN30_Cleaned_Dataset.csv`. Muốn quay lại target 1.5%, chạy lại `dataset.py` và `data_cleaning.py`.
- `feature_split.py` và `feature_split_02.py` cùng ghi đè `VN30_Train.csv` và `VN30_Test.csv`.
- `model_01.py` và `model_02.py` cùng ghi đè `xgboost_quant_model.pkl`.
- Trước khi chạy `model_01b.py` hoặc `model_01e.py`, hãy chắc chắn `xgboost_quant_model.pkl` là model baseline từ `model_01.py`.
- Trước khi chạy `backtest.py`, hãy chắc chắn `xgboost_quant_model.pkl` là model Triple Barrier từ `model_02.py`.
- Dữ liệu tải qua `vnstock` có thể thay đổi theo thời điểm chạy, vì vậy số dòng và kết quả có thể lệch nhẹ nếu chạy lại vào ngày khác hoặc dùng phiên bản thư viện khác.
