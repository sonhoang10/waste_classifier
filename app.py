#Giao diện gradio
# 1. Import thư viện
import os
import cv2
import joblib
import gradio as gr
import numpy as np
from skimage.feature import hog

# Dữ liệu về các loại rác và cách xử lý
type_and_method = {
    # Nhóm Vô cơ (Inorganic)
    "build_matierial": ["Inorganic (Vô cơ)", "Vật liệu xây dựng như gạch, vữa, bê tông có thể được tái chế để san lấp. Kim loại nên được bán cho cơ sở phế liệu."],
    "clothing": ["Inorganic (Vô cơ)", "Quần áo còn tốt: Quyên góp từ thiện.\nQuần áo hỏng: Tái chế thành giẻ lau, vật liệu cách âm hoặc bỏ vào thùng rác vô cơ."],
    "cloth_dphuc": ["Inorganic (Vô cơ)", "Đồng phục cũ/hỏng có thể được tái chế. Nếu không, bỏ vào thùng rác vô cơ."],
    "electronic": ["Inorganic (Vô cơ)", "Rác thải điện tử nguy hại. Không vứt vào rác thường. Mang đến các điểm thu hồi rác thải điện tử tại siêu thị, trung tâm bảo hành."],
    "light_bulb": ["Inorganic (Vô cơ)", "Bóng đèn huỳnh quang/compact chứa thủy ngân, cần được xử lý riêng như chất thải nguy hại. Bóng LED có thể tái chế kim loại và nhựa."],
    "medical_trash": ["Inorganic (Vô cơ)", "Chất thải y tế cực kỳ nguy hại. Phải được thu gom và xử lý theo quy định của cơ sở y tế."],
    "Rubber_wheels": ["Inorganic (Vô cơ)", "Lốp xe có thể được tái chế thành cao su tái sinh để làm thảm, sân chơi. Liên hệ các cơ sở tái chế chuyên nghiệp."],
    "table_chair": ["Inorganic (Vô cơ)", "Nếu còn sử dụng được, hãy thanh lý hoặc quyên góp. Nếu hỏng, phân loại các bộ phận (gỗ, kim loại, nhựa) để tái chế hoặc bỏ đi."],
    "Fmask_Tbrush": ["Inorganic (Vô cơ)", "Khẩu trang, bàn chải đánh răng là rác thải vô cơ, không tái chế được. Bỏ vào thùng rác vô cơ."],
    "pen": ["Inorganic (Vô cơ)", "Bút viết là rác thải vô cơ, không tái chế được. Bỏ vào thùng rác vô cơ."],
    "plastic_non_re": ["Inorganic (Vô cơ)", "Các loại nhựa không thể tái chế (nhựa dùng một lần bẩn, đồ chơi phức hợp). Bỏ vào thùng rác vô cơ."],
    "Thermos": ["Inorganic (Vô cơ)", "Bình giữ nhiệt hỏng là rác thải vô cơ. Bỏ vào thùng rác vô cơ."],

    # Nhóm Hữu cơ (Organic)
    "oil_cook": ["Organic (Hữu cơ)", "Không đổ trực tiếp xuống cống. Gom lại vào chai/lọ kín và mang đến các điểm thu gom dầu ăn đã qua sử dụng để tái chế thành nhiên liệu sinh học."],
    "flower": ["Organic (Hữu cơ)", "Có thể ủ làm phân compost tại nhà hoặc bỏ vào thùng rác hữu cơ."],
    "fruit": ["Organic (Hữu cơ)", "Vỏ và phần thừa của trái cây là rác hữu cơ. Có thể ủ làm phân compost hoặc bỏ vào thùng rác hữu cơ."],
    "meat": ["Organic (Hữu cơ)", "Thịt và xương thừa là rác hữu cơ. Bỏ vào thùng rác hữu cơ."],
    "pencil": ["Organic (Hữu cơ)", "Vỏ bút chì (gỗ) có thể phân hủy hữu cơ. Bỏ vào thùng rác hữu cơ."],
    "plant": ["Organic (Hữu cơ)", "Lá cây, cành cây nhỏ là rác hữu cơ. Có thể ủ làm phân compost hoặc bỏ vào thùng rác hữu cơ."],

    # Nhóm Tái chế (Recyclable)
    "milk_box": ["Recyclable (Tái chế)", "Xả sạch, làm khô và dẹp lại. Đem đến các điểm thu gom vỏ hộp giấy chuyên dụng (ví dụ: của Tetra Pak) để tái chế."],
    "cardboard": ["Recyclable (Tái chế)", "Làm sạch, làm phẳng và bỏ vào thùng rác tái chế hoặc bán cho cơ sở phế liệu."],
    "glass": ["Recyclable (Tái chế)", "Làm sạch và bỏ vào thùng rác tái chế. Cẩn thận để không gây nguy hiểm."],
    "metal": ["Recyclable (Tái chế)", "Làm sạch và bỏ vào thùng rác tái chế hoặc bán cho cơ sở phế liệu."],
    "paper": ["Recyclable (Tái chế)", "Giữ khô ráo, không dính dầu mỡ. Bỏ vào thùng rác tái chế hoặc bán cho cơ sở phế liệu."],
    "Plastic": ["Recyclable (Tái chế)", "Làm sạch, loại bỏ nhãn nếu có thể. Bỏ vào thùng rác tái chế các loại nhựa có ký hiệu tái chế (PET, HDPE,...)."]
}

# 2. Load model, scaler, và label_map
MODEL_PATH = "waste_classifier.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy tệp model tại: {MODEL_PATH}")
svm_model, scaler, label_map = joblib.load(MODEL_PATH)


# 3. HOG (trích xuất đặc trưng)
def extract_hog_features(image_path, size=(128, 128)): #Thông số resize của ảnh (càng cao càng chính xác)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    img = cv2.resize(img, size)
    features = hog(img,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys',
                   visualize=False,
                   feature_vector=True)
    return features

# 4. Hàm dự đoán
def predict_image(image):
    features = extract_hog_features(image)
    features = scaler.transform([features])  # Chuẩn hóa như lúc train

    prediction = svm_model.predict(features)[0]
    label_name = label_map[prediction]

    waste_type, guide = type_and_method.get(label_name, ["Không rõ", "Chưa có hướng dẫn."])

    # Thêm biểu tượng cho đẹp
    if "hữu cơ" in waste_type.lower():
        waste_type_display = waste_type + " 🌿"
    elif "tái chế" in waste_type.lower():
        waste_type_display = waste_type + " ♻️"
    else:
        waste_type_display = waste_type + " 🗑️"

    return image, label_name, waste_type_display, guide


# 5. Xây dựng giao diện Gradio
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(
        type="pil",
        label="📸 Tải ảnh hoặc chụp bằng webcam",
        sources=["upload", "clipboard", "webcam"]
    ),
    outputs=[
        gr.Image(label="📷 Ảnh bạn đã đưa vào"),
        gr.Textbox(label="🔖 Tên loại rác (dự đoán)"),
        gr.Textbox(label="♻️ Nhóm rác", interactive=False),
        gr.Textbox(label="📘 Hướng dẫn xử lý", lines=5, interactive=False)
    ],
    title="♻️ Hệ thống phân loại rác thông minh",
    description="Tải ảnh hoặc chụp ảnh rác để AI nhận diện loại rác và đề xuất cách xử lý phù hợp.",
    theme="soft",
    allow_flagging="never"
)

# Run
if __name__ == "__main__":
    demo.launch()
