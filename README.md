# ♻️ AI phân loại rác thông minh (Không sử dụng DeepLearning)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)  
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)](https://scikit-learn.org/stable/)  
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green)](https://opencv.org/)  
[![Gradio](https://img.shields.io/badge/Gradio-Web%20UI-red)](https://gradio.app/)  

Ứng dụng AI sử dụng **HOG (Histogram of Oriented Gradients)** và **SVM (Support Vector Machine)** để **phân loại rác** thành các nhóm khác nhau (**Vô cơ, Hữu cơ, Tái chế**). Sau khi nhận diện loại rác, hệ thống còn đưa ra **hướng dẫn xử lý** phù hợp để giảm thiểu ô nhiễm môi trường.  

---

## 🚀 Tính năng chính  
- 📸 Nhận diện rác từ **ảnh tải lên** hoặc **chụp trực tiếp qua webcam**.  
- 🧠 Mô hình **SVM + HOG** gọn nhẹ, dự đoán nhanh, dễ triển khai.  
- ♻️ Phân loại rác thành 3 nhóm chính:  
  - **Organic (Hữu cơ)** 🌿  
  - **Recyclable (Tái chế)** ♻️  
  - **Inorganic (Vô cơ)** 🗑️  
- 📘 Hiển thị **hướng dẫn xử lý rác**: tái chế, ủ phân, bỏ đúng nơi quy định, xử lý rác nguy hại…  
- 🌐 Giao diện web trực quan bằng **Gradio**.  

---

## 🛠️ Công nghệ sử dụng  
- [Python](https://www.python.org/)  
- [OpenCV](https://opencv.org/) – Xử lý ảnh  
- [scikit-learn](https://scikit-learn.org/stable/) – Huấn luyện & dự đoán  
- [scikit-image](https://scikit-image.org/) – Trích xuất đặc trưng HOG  
- [Gradio](https://gradio.app/) – Giao diện web  

---

## 📂 Cấu trúc dự án  
- train_model.py # Huấn luyện mô hình và lưu file waste_classifier.pkl
- app.py # Chạy giao diện Gradio với mô hình đã huấn luyện
- dataset/ # Dataset phân loại rác (chia thư mục con theo loại)
- waste_classifier.pkl # Mô hình bạn đã huấn luyện
- requirements.txt # Thư viện cần thiết
- README.md # Giới thiệu dự án


---

### ▶️ Cách chạy  
```bash
1️⃣ Clone repo  
git clone https://github.com/your-username/waste-classifier.git
cd waste-classifier

2️⃣ Cài đặt thư viện
pip install -r requirements.txt

3️⃣ (Tuỳ chọn) Huấn luyện mô hình lại (Đề xuất sử dụng GPU hoặc gg colab pro để train)
python train_model.py

4️⃣ Chạy ứng dụng Gradio
python app.py

5️⃣ Mở trình duyệt tại:
👉 http://127.0.0.1:7860/
```
📊 Kết quả mô hình

Mô hình sử dụng SVM (linear kernel)

Độ chính xác trung bình đạt: ~90% (tuỳ dataset)

Ví dụ kết quả dự đoán trên Gradio:

Ảnh đầu vào	Kết quả dự đoán	Nhóm rác	Hướng dẫn xử lý
🥩 thịt	meat	Organic 🌿	Bỏ vào thùng rác hữu cơ
📦 hộp giấy	cardboard	Recyclable ♻️	Làm sạch, ép phẳng và bỏ vào thùng tái chế
💡 bóng đèn	light_bulb	Inorganic 🗑️	Mang đến điểm thu gom rác thải nguy hại
💡 Ứng dụng thực tế

Giáo dục cộng đồng về phân loại rác đúng cách.

Hỗ trợ trong hệ thống thùng rác thông minh.

Tích hợp vào ứng dụng di động để người dùng dễ dàng sử dụng.
