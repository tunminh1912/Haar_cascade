import cv2
import numpy as np
import os
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()

# Kiểm tra xem mô hình đã lưu có tồn tại không
if os.path.exists('recognizer/trainningData.yml'):
    recognizer.read('recognizer/trainningData.yml')

# Đường dẫn tệp lưu danh sách ID đã huấn luyện
trained_ids_file = 'recognizer/trained_ids.txt'

# Tạo thư mục "recognizer" nếu chưa tồn tại
if not os.path.exists('recognizer'):
    os.makedirs('recognizer')

# Đọc danh sách các ID đã được huấn luyện
if os.path.exists(trained_ids_file):
    with open(trained_ids_file, 'r') as f:
        trained_ids = set(map(int, f.read().splitlines()))
else:
    trained_ids = set()

path = 'dataSet'

def getImageWithId(path, trained_ids):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    array_faces = []
    array_IDs = []

    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')

        Id = int(os.path.basename(imagePath).split('.')[1])  # Sửa lại để hoạt động trên mọi hệ điều hành
        if Id in trained_ids:
            print(f"ID {Id} đã được huấn luyện, bỏ qua...")
            continue

        array_faces.append(faceNp)
        array_IDs.append(Id)

        cv2.imshow('trainning', faceNp)
        cv2.waitKey(30)

    return array_faces, array_IDs

array_faces, array_IDs = getImageWithId(path, trained_ids)

# Nếu có hình ảnh mới, huấn luyện mô hình
if array_faces:
    recognizer.update(array_faces, np.array(array_IDs))

    # Lưu lại danh sách các ID đã được huấn luyện
    trained_ids.update(array_IDs)
    with open(trained_ids_file, 'w') as f:
        f.write('\n'.join(map(str, trained_ids)))

    # Lưu mô hình đã huấn luyện
    recognizer.save('recognizer/trainningData.yml')
else:
    print("Không có hình ảnh mới để huấn luyện.")

cv2.destroyAllWindows()
