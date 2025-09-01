# 1. Import thư viện
import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm  # hiển thị tiến trình

# 2. Download dataset từ gg drive
# dataset = https://drive.google.com/file/d/1kEd4pP0ArDlHfxQ6AxlaeMcMrGwjfz3d/view?usp=sharing
# Đảm bảo thư mục "dataset" nằm cùng cấp với tệp train_model.py
dataset_path = "dataset/"

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

# 4. Get dữ liệu (duyệt sâu vào từng folder con) và gán nhãn
def load_dataset(root_dir):
    X, y, labels = [], [], []
    label_map = {}
    label_id = 0

    class_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    for class_name in class_dirs:
        label_map[label_id] = class_name
        class_path = os.path.join(root_dir, class_name)
        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        print(f"🔹 Lớp: {class_name} ({len(files)} ảnh)")
        for file in tqdm(files, desc=f"  -> Xử lý {class_name}", leave=False):
            file_path = os.path.join(class_path, file)
            try:
                feat = extract_hog_features(file_path)
                X.append(feat)
                y.append(label_id)
            except Exception as e:
                print(f"⚠️ Lỗi với {file_path}: {e}")

        label_id += 1

    return np.array(X), np.array(y), label_map

# 5. Tải dữ liệu
print("\n📥 Đang load dataset và trích xuất đặc trưng HOG...")
X, y, label_map = load_dataset(dataset_path)

# 6. Chuẩn hóa đặc trưng
print("\n⚙️  Đang chuẩn hóa dữ liệu...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Chia tập train/test
print("\n📊 Đang chia dữ liệu train/test...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 8. Train SVM
print("\n🧠 Đang huấn luyện mô hình SVM...")
svm_model = SVC(kernel='linear', probability=True, C=1.0)
svm_model.fit(X_train, y_train)

# 9. Lưu model
print("\n💾 Đang lưu mô hình...")
joblib.dump((svm_model, scaler, label_map), "waste_classifier.pkl")
print("✅ Đã lưu mô hình vào waste_classifier.pkl")

# 10. Đánh giá mô hình
print("\n📈 Đang đánh giá mô hình...")
y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
target_names = [label_map[i] for i in sorted(set(y))]
report = classification_report(y_test, y_pred, target_names=target_names)

print(f"\n🎯 Độ chính xác: {accuracy:.4f}")
print("\n📋 Báo cáo phân loại:\n")
print(report)