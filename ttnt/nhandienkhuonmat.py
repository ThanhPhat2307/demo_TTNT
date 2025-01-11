import cv2
import numpy as np
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Hàm huấn luyện AdaBoost với bộ phân loại DecisionTree
def train_adaboost(X_train, y_train):
    # Sử dụng cây quyết định với độ sâu nhỏ để làm bộ phân loại yếu
    base_estimator = DecisionTreeClassifier(max_depth=1)
    # Khởi tạo mô hình AdaBoost với tham số 'estimator' thay vì 'base_estimator'
    ada_boost = AdaBoostClassifier(estimator=base_estimator, n_estimators=50)
    # Huấn luyện mô hình
    ada_boost.fit(X_train, y_train)
    return ada_boost

# Hàm sử dụng OpenCV để phát hiện khuôn mặt
def detect_faces(image, face_cascade):
    # Chuyển ảnh về thang độ xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Phát hiện khuôn mặt trong ảnh
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Tải dữ liệu khuôn mặt từ OpenCV (hệ thống Haar Cascades)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Hàm tiền xử lý dữ liệu từ ảnh
def extract_face_features(image_path, face_cascade):
    image = cv2.imread(image_path)
    if image is None:
        return None, None  # Nếu không đọc được ảnh, trả về None

    faces = detect_faces(image, face_cascade)
    if len(faces) == 0:
        return None, None  # Nếu không phát hiện khuôn mặt, trả về None

    features = []
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]  # Cắt khuôn mặt
        face = cv2.resize(face, (50, 50))  # Resize về kích thước cố định
        features.append(face.flatten())  # Chuyển thành vector đặc trưng 1 chiều

    return np.array(features), np.ones(len(features))  # Trả về các đặc trưng của khuôn mặt và nhãn là 1

# Đọc ảnh từ thư mục và trích xuất đặc trưng
def load_data_from_folder(folder_path, face_cascade):
    X = []
    y = []

    # Duyệt qua các ảnh trong thư mục
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        features, labels = extract_face_features(file_path, face_cascade)
        if features is not None:
            X.append(features)
            y.append(labels)

    # Chuyển đổi thành mảng numpy
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

# Đọc ảnh có khuôn mặt từ thư mục
face_folder = 'mat'  # Thay thế bằng đường dẫn tới thư mục chứa ảnh có khuôn mặt
X_faces, y_faces = load_data_from_folder(face_folder, face_cascade)

# Giả sử bạn có một thư mục ảnh không có khuôn mặt, bạn có thể tạo dữ liệu âm (0)
# Cách làm đơn giản là lấy ảnh ngẫu nhiên từ thư mục khác không chứa khuôn mặt
non_face_folder = 'khongcomat'  # Thay thế bằng thư mục ảnh không chứa khuôn mặt
X_non_faces, y_non_faces = load_data_from_folder(non_face_folder, face_cascade)

# Ghép dữ liệu khuôn mặt và không khuôn mặt lại
X = np.vstack([X_faces, X_non_faces])
y = np.hstack([y_faces, np.zeros(len(y_non_faces))])  # Nhãn 0 cho không khuôn mặt

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Huấn luyện mô hình AdaBoost
model = train_adaboost(X_train, y_train)

# Kiểm tra mô hình trên tập kiểm tra
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))

# Đảm bảo bạn cung cấp đúng đường dẫn ảnh để thử nghiệm nhận diện khuôn mặt
image_path = 'phat_2.jpg'  # Thay thế với đường dẫn tới ảnh của bạn
image = cv2.imread(image_path)

# Kiểm tra nếu ảnh có được tải thành công không
if image is None:
    print(f"Lỗi: Không thể đọc ảnh từ đường dẫn {image_path}. Kiểm tra lại đường dẫn ảnh.")
else:
    # Phát hiện khuôn mặt trong ảnh
    faces = detect_faces(image, face_cascade)

    # Vẽ hình chữ nhật quanh các khuôn mặt phát hiện được
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Hiển thị ảnh kết quả
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Tắt trục
    plt.show()

    # Lưu kết quả ra file (tùy chọn)
    cv2.imwrite('detected_faces.jpg', image)
