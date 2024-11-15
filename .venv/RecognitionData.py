import cv2
import numpy as np
import sqlite3
from PIL import Image, ImageTk
import tkinter as tk

# Tải cascade cho nhận diện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read(r'C:\Users\Admin\PycharmProjects\NhanDienKhuonMat\.venv\recognizer\trainningData.yml')


def getProfile(id):
    conn = sqlite3.connect('./data.db')
    cursor = conn.cursor()
    query = "SELECT * FROM people WHERE ID=?"
    cursor.execute(query, (id,))
    profile = cursor.fetchone()
    conn.close()
    return profile


def toggle_camera_display():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)
        update_frame()
    else:
        cap.release()
        cap = None
        camera_label.configure(image='')


def update_frame():
    global cap
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y: y + h, x: x + w]
        # Nhận diện khuôn mặt
        id, confidence = recognizer.predict(roi_gray)
        if confidence < 40:
            profile = getProfile(id)
            if profile is not None:
                cv2.putText(frame, " " + str(profile[1]), (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tk = ImageTk.PhotoImage(img)
    camera_label.imgtk = img_tk
    camera_label.configure(image=img_tk)

    camera_label.after(10, update_frame)


window = tk.Tk()
window.title("Xử Lý Ảnh và Thị Giác Máy Tính")
window.geometry("800x600")

title_label = tk.Label(window, text="XỬ LÝ ẢNH VÀ THỊ GIÁC MÁY TÍNH", bg="#E79F73", fg="white", font=("Arial", 18))
title_label.grid(row=0, column=0, columnspan=2, pady=10, sticky="ew")

left_frame = tk.Frame(window)
left_frame.grid(row=1, column=0, padx=20, pady=20, sticky="n")

nhom_label = tk.Label(left_frame, text="NHÓM 07", bg="#7ECFF1", fg="white", font=("Arial", 14))
nhom_label.pack(pady=5)

camera_button = tk.Button(left_frame, text="Camera", bg="#90EE90", font=("Arial", 16), command=toggle_camera_display)
camera_button.pack(pady=10)

upload_button = tk.Button(left_frame, text="Upload", bg="palegreen", font=("Arial", 16),
                          command=lambda: None)
upload_button.pack(pady=10)

camera_label = tk.Label(window)
camera_label.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")

cap = None
fontface = cv2.FONT_HERSHEY_SIMPLEX

window.mainloop()
