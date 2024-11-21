import cv2
import numpy as np
import sqlite3
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import os
import time
import threading

# Tải cascade cho nhận diện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

localtime = time.asctime( time.localtime(time.time()) )

def getProfile(id):
    conn = sqlite3.connect('./data.db')
    cursor = conn.cursor()
    query = "SELECT * FROM people WHERE ID=?"
    cursor.execute(query, (id,))
    profile = cursor.fetchone()
    conn.close()
    return profile

def toggle_camera_display():
    recognizer.read(r'C:\Users\Admin\PycharmProjects\NhanDienKhuonMat\.venv\recognizer\trainningData.yml')
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)
        update_frame()
    else:
        cap.release()
        cap = None
        if info_label.winfo_exists():  # Check if the info_label still exists
            info_label.config(text="")  # Clear the info when the camera is off
        camera_label.configure(image='')  # Clear the image from the camera label

def update_frame():
    global cap
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    user_info = "Unknown"  # Default user info display

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y: y + h, x: x + w]
        id, confidence = recognizer.predict(roi_gray)

        if confidence < 40:
            profile = getProfile(id)
            if profile is not None:
                user_info = (
                    f"MSSV: {profile[0]}\n"
                    f"Tên: {profile[1]}\n"
                    f"Ngày sinh: {profile[2]}\n"
                    f"Khoa: {profile[3]}\n"
                    f"Thời gian: {localtime}"
                )
                cv2.putText(frame, profile[1], (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tk = ImageTk.PhotoImage(img)
    camera_label.imgtk = img_tk
    camera_label.configure(image=img_tk)

    # Update user info in info_label
    info_label.config(text=user_info, anchor="w", justify="left")

    camera_label.after(50, update_frame)

def run_train_data():
    # Kiểm tra xem mô hình đã lưu có tồn tại không
    if os.path.exists('recognizer/trainningData.yml'):
        recognizer.read('recognizer/trainningData.yml')

    trained_ids_file = 'recognizer/trained_ids.txt' # Đường dẫn tệp lưu danh sách ID đã huấn luyện

    if not os.path.exists('recognizer'):    # Tạo thư mục "recognizer" nếu chưa tồn tại
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


def insertOrupdate(id_value, name_value, birthday_value, major_value, localtime):
    conn = sqlite3.connect('./data.db')
    try:
        # Kiểm tra bản ghi có tồn tại không
        query = "SELECT * FROM People WHERE ID=?"
        cursor = conn.execute(query, (id_value,))
        isRecordExist = len(cursor.fetchall()) > 0

        if not isRecordExist:
            # Thêm mới
            query = """
                INSERT INTO People (ID, Name, Birthday, Major, localtime)
                VALUES (?, ?, ?, ?, ?)
            """
            conn.execute(query, (id_value, name_value, birthday_value, major_value, localtime))
        else:
            # Cập nhật
            query = """
                UPDATE People 
                SET Name = ?, Birthday = ?, Major = ?, localtime = ?
                WHERE ID = ?
            """
            conn.execute(query, (name_value, birthday_value, major_value, localtime, id_value))

        conn.commit()
        print("Dữ liệu đã được lưu thành công!")
    except sqlite3.Error as e:
        print(f"Lỗi SQLite: {e}")
    finally:
        conn.close()


def get_data_func():
    right_frame = tk.Frame(window)
    right_frame.grid(row=1, column=1, columnspan=2, padx=20, pady=20, sticky="n")

    # Label và Entry cho mã số sinh viên
    tk.Label(right_frame, text="Mã số sinh viên:", font=("Arial", 12)).grid(row=2, column=1, padx=10, pady=5,
                                                                            sticky="w")
    id_entry = tk.Entry(right_frame, font=("Arial", 12))
    id_entry.grid(row=2, column=2, padx=10, pady=5)

    # Label và Entry cho họ và tên
    tk.Label(right_frame, text="Họ và tên:", font=("Arial", 12)).grid(row=3, column=1, padx=10, pady=5, sticky="w")
    name_entry = tk.Entry(right_frame, font=("Arial", 12))
    name_entry.grid(row=3, column=2, padx=10, pady=5)

    # Label và Entry cho ngày sinh
    tk.Label(right_frame, text="Ngày sinh (dd/mm/yyyy):", font=("Arial", 12)).grid(row=4, column=1, padx=10, pady=5,
                                                                                   sticky="w")
    birthday_entry = tk.Entry(right_frame, font=("Arial", 12))
    birthday_entry.grid(row=4, column=2, padx=10, pady=5)

    # Label và Entry cho ngành học
    tk.Label(right_frame, text="Ngành học:", font=("Arial", 12)).grid(row=5, column=1, padx=10, pady=5, sticky="w")
    major_entry = tk.Entry(right_frame, font=("Arial", 12))
    major_entry.grid(row=5, column=2, padx=10, pady=5)

    # Nút Camera
    def on_save():
        # Lấy giá trị từ Entry
        id_value = id_entry.get()
        name_value = name_entry.get()
        birthday_value = birthday_entry.get()
        major_value = major_entry.get()
        localtime = time.asctime(time.localtime(time.time()))

        # Gọi hàm lưu dữ liệu
        if id_value and name_value and birthday_value and major_value:
            insertOrupdate(id_value, name_value, birthday_value, major_value, localtime)

            right_frame.destroy() # Xóa các widget cũ trong right_frame

            # Chạy quá trình lấy ảnh từ camera trong một luồng riêng biệt
            def capture_images():
                sampleNum = 0
                cap = cv2.VideoCapture(0)

                while True:
                    ret, frame = cap.read()

                    if not ret:
                        break

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Chuyển ảnh thành ảnh xám

                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                        if not os.path.exists('dataSet'):
                            os.makedirs('dataSet')  # Tạo thư mục nếu chưa có

                        sampleNum += 1
                        cv2.imwrite(f'dataSet/User.{id_value}.{sampleNum}.jpg', gray[y:y + h, x:x + w])

                    # Chuyển đổi ảnh từ OpenCV (BGR) sang ảnh RGB của Tkinter
                    cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(cv_image)
                    img_tk = ImageTk.PhotoImage(img)

                    # Cập nhật ảnh lên camera_label
                    camera_label.img_tk = img_tk  # Tham chiếu để giữ ảnh trong bộ nhớ
                    camera_label.config(image=img_tk)

                    if sampleNum >= 100:
                        camera_label.destroy()
                        break

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()

            # Khởi chạy một luồng mới cho việc capture ảnh
            capture_thread = threading.Thread(target=capture_images)
            capture_thread.start()

        else:
            print("Vui lòng điền đầy đủ thông tin!")

    get_camera = tk.Button(right_frame, text="Camera", bg="#90EE90", font=("Arial", 16), command=on_save)
    get_camera.grid(row=6, column=2, padx=10, pady=5)


def upload_image():
    recognizer.read(r'C:\Users\Admin\PycharmProjects\NhanDienKhuonMat\.venv\recognizer\trainningData.yml')
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
    )
    if not file_path:
        return  # Người dùng không chọn ảnh

    # Đọc ảnh bằng OpenCV
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    user_info = "Unknown"  # Thông tin mặc định

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y: y + h, x: x + w]
        id, confidence = recognizer.predict(roi_gray)

        if confidence < 40:  # Ngưỡng nhận diện
            profile = getProfile(id)
            if profile is not None:
                user_info = (
                    f"MSSV: {profile[0]}\n"
                    f"Tên: {profile[1]}\n"
                    f"Ngày sinh: {profile[2]}\n"
                    f"Khoa: {profile[3]}\n"
                    f"Thời gian: {localtime}"
                )
                cv2.putText(image, profile[1], (x + 10, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(image, "Unknown", (x + 10, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Resize ảnh về kích thước cố định, ví dụ 400x400
    fixed_size = (600, 400)
    resized_image = cv2.resize(image, fixed_size)

    # Chuyển đổi ảnh sang định dạng Tkinter
    img = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    img_tk = ImageTk.PhotoImage(img)

    # Hiển thị ảnh trên label
    camera_label.config(image=img_tk)
    camera_label.image = img_tk

    # Hiển thị thông tin người dùng
    if info_label.winfo_exists():  # Check if info_label exists before updating it
        info_label.config(text=user_info, anchor="w", justify="left")



window = tk.Tk()
window.title("Xử Lý Ảnh và Thị Giác Máy Tính")
window.geometry("1024x700")

title_label = tk.Label(window, text="XỬ LÝ ẢNH VÀ THỊ GIÁC MÁY TÍNH", bg="#E79F73", fg="white", font=("Arial", 18))
title_label.grid(row=0, column=0, columnspan=2, pady=10, sticky="ew")

left_frame = tk.Frame(window)
left_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")

nhom_label = tk.Label(left_frame, text="NHÓM 07", bg="#7ECFF1", fg="white", font=("Arial", 14))
nhom_label.pack(pady=5, anchor="w")

get_data_button = tk.Button(left_frame, text="Get data", bg="#90EE90", font=("Arial", 16), command=get_data_func)
get_data_button.pack(pady=10, anchor="w")

train_data_button = tk.Button(left_frame, text="Train data", bg="#90EE90", font=("Arial", 16), command=run_train_data)
train_data_button.pack(pady=10, anchor="w")

camera_button = tk.Button(left_frame, text="Camera", bg="#90EE90", font=("Arial", 16), command=toggle_camera_display)
camera_button.pack(pady=10, anchor="w")

upload_button = tk.Button(left_frame, text="Upload", bg="palegreen", font=("Arial", 16), command=upload_image)
upload_button.pack(pady=10, anchor="w")

# Add user info label below the camera label
info_label = tk.Label(left_frame, text="", font=("Arial", 12))
info_label.pack(pady=10, anchor="w")

camera_label = tk.Label(window)
camera_label.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")

cap = None
fontface = cv2.FONT_HERSHEY_SIMPLEX

window.mainloop()
