import cv2
import numpy as np
import os
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()

# Kiểm tra xem mô hình đã lưu có tồn tại không
if os.path.exists('recognizer/trainningData.yml'):
    recognizer.read('recognizer/trainningData.yml')

path = 'dataSet'

def getImageWithId(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    # print(imagePaths)
    array_faces = []
    array_IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')
        print(faceNp)

        Id = int(imagePath.split('\\')[1].split('.')[1])
        array_faces.append(faceNp)
        array_IDs.append(Id)

        cv2.imshow('trainning', faceNp)
        cv2.waitKey(30)

    return array_faces, array_IDs

array_faces, array_IDs = getImageWithId(path)

# Huấn luyện với các hình ảnh mới
recognizer.update(array_faces, np.array(array_IDs))

if not os.path.exists('recognizer'):
    os.makedirs('recognizer')

recognizer.save('recognizer/trainningData.yml')

cv2.destroyAllWindows()
