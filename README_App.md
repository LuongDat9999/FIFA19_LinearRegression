# ⚽ FIFA 19 Player Analysis & Prediction App

Ứng dụng Streamlit để phân tích và dự đoán chỉ số cầu thủ FIFA 19 sử dụng Machine Learning.

## 🚀 Tính năng chính

### 🏠 Trang chủ
- Tổng quan về ứng dụng
- Thống kê cơ bản về dữ liệu FIFA 19
- Giới thiệu các tính năng và mô hình

### 🔮 Dự đoán Overall
- Dự đoán chỉ số Overall của cầu thủ dựa trên các thuộc tính
- Sử dụng cả mô hình hồi quy tuyến tính thủ công và sklearn
- Hiển thị độ chính xác của mô hình (R² Score, RMSE)
- Giao diện thân thiện với slider để nhập thông số

### 📊 So sánh Cầu thủ
- So sánh nhiều cầu thủ cùng lúc
- Biểu đồ cột và biểu đồ radar
- Chọn linh hoạt các thuộc tính để so sánh
- Bảng dữ liệu chi tiết

### 📈 Phân tích Dữ liệu
- Phân phối các thuộc tính quan trọng
- Ma trận tương quan giữa các thuộc tính
- Biểu đồ phân tích chi tiết
- Thống kê mô tả dữ liệu

## 🛠️ Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- pip hoặc conda

### Cài đặt thư viện
```bash
pip install -r requirements.txt
```

## 🚀 Chạy ứng dụng

### Bước 1: Chuẩn bị dữ liệu
Đảm bảo file `processed_data.csv` đã được tạo từ notebook `Fifa.ipynb`

### Bước 2: Chạy ứng dụng
```bash
streamlit run app.py
```

Ứng dụng sẽ mở tại: `http://localhost:8501`

## 📊 Dữ liệu sử dụng

- **Nguồn**: FIFA 19 Dataset
- **Số lượng cầu thủ**: 18,129
- **Số thuộc tính**: 55
- **Thuộc tính chính**:
  - Age (Tuổi)
  - Potential (Tiềm năng)
  - International Reputation (Danh tiếng quốc tế)
  - Skill Moves (Kỹ năng rê bóng)
  - Reactions (Khả năng phản ứng)
  - Overall (Chỉ số tổng quát)

## 🎯 Mô hình Machine Learning

### Linear Regression
- **Mô hình thủ công**: Sử dụng công thức toán học để tính hệ số hồi quy
- **Mô hình sklearn**: Sử dụng thư viện scikit-learn
- **Độ chính xác**: ~91% (R² Score)
- **Thuộc tính quan trọng nhất**:
  1. Skill Moves
  2. International Reputation
  3. Age
  4. Potential

## 🔧 Cấu trúc code

```
app.py
├── Cấu hình Streamlit
├── Hàm xử lý dữ liệu
├── Hàm hồi quy tuyến tính
├── Trang chủ
├── Trang dự đoán
├── Trang so sánh cầu thủ
└── Trang phân tích dữ liệu
```

## 📱 Giao diện

- **Responsive design**: Tương thích với mọi kích thước màn hình
- **Sidebar navigation**: Điều hướng dễ dàng giữa các trang
- **Interactive widgets**: Slider, multiselect, button
- **Visualization**: Biểu đồ matplotlib và seaborn
- **Real-time updates**: Cập nhật kết quả theo thời gian thực

---

⭐ Nếu dự án này hữu ích, hãy cho một star!
