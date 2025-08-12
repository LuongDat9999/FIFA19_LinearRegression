# FIFA19_LinearRegression

Dự án hồi quy tuyến tính (Linear Regression) phân tích dữ liệu cầu thủ trong FIFA 19 nhằm dự đoán **chỉ số tổng thể** (Overall Rating) hoặc các đặc tính quan trọng khác của cầu thủ.

---

## Giới thiệu
Dự án sử dụng mô hình hồi quy tuyến tính để xây dựng mô hình dự đoán trên bộ dữ liệu FIFA 19, nhằm hiểu rõ mối quan hệ giữa các thuộc tính cầu thủ và hiệu suất đánh giá tổng thể.

---

## Mục tiêu
- Khám phá và tiền xử lý dữ liệu cầu thủ (tuổi, kỹ năng, chỉ số vật lý...).
- Huấn luyện mô hình **Linear Regression** để dự đoán giá trị thực của chỉ số tổng thể (Overall).
- Đánh giá hiệu suất mô hình & trực quan hóa hiệu quả dự đoán.

---

## Tính năng
- Notebook `Fifa.ipynb`:  
  - Tiền xử lý dữ liệu;  
  - Phân tích & trực quan;  
  - Huấn luyện mô hình hồi quy;  
  - Đánh giá đường hồi quy (R², MSE,...).
- Báo cáo chi tiết (`Report_Detail.pdf`) mô tả phương pháp, kết quả và đánh giá.

---
## Yêu cầu hệ thống
- Python 3.x  
- Jupyter Notebook  
- Thư viện: pandas, numpy, matplotlib, scikit-learn (các version mới)

---
## Phương pháp & Quy trình
- Tiền xử lý dữ liệu: làm sạch, loại bỏ giá trị thiếu, định dạng lại kiểu dữ liệu.
- Khám phá dữ liệu: phân tích phân phối, tương quan giữa các biến.
- Huấn luyện mô hình: sử dụng Linear Regression đơn giản – chọn biến đầu vào phù hợp.
- Đánh giá hiệu suất: R² (độ phù hợp), MSE (Sai số bình phương trung bình), và trực quan hóa kết quả.

  
