import cv2
import numpy as np
import sqlite3
import os   #truy cay cac thu muc
import time;

localtime = time.asctime( time.localtime(time.time()) )

def insertOrupdate(id, name, birthday, major, localtime):
    conn = sqlite3.connect('./data.db')

    query = "SELECT * FROM People WHERE ID=?"
    cursor = conn.execute(query, (id,))

    isRecordExist = 0

    for row in cursor:
        isRecordExist = 1

    if(isRecordExist == 0):
        query = "INSERT INTO People(ID,Name,Birthday,Major,localtime) VALUES("+str(id)+",'"+str(name)+"','"+str(birthday)+"','"+str(major)+"','"+str(localtime)+"')"
    else :
        query = "UPDATE People SET Name='"+str(name)+"',Birthday='"+str(birthday)+"',Major='"+str(major)+"',localtime='"+str(localtime)+"' WHERE ID="+str(id)+""

    conn.execute(query)
    conn.commit()
    conn.close()

#load tv
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

#insert to db
print("Thời gian hiện tại là :", localtime)
id = input("Nhập mã số sinh viên: ")
name = input("Nhập họ và tên: ")
birthday = input("Nhập ngày sinh: ")
major = input("Nhập ngành học: ")

insertOrupdate(id,name,birthday,major,localtime)

sampleNum = 0

while(True):
    ret, frame = cap.read()

    #chuyen thanh anh xam de train model
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)

        if not os.path.exists('dataSet'):
            os.makedirs('dataSet')  #tao folder moi

        sampleNum += 1
        #luu User.1.1
        cv2.imwrite('dataSet/User.'+str(id)+'.'+str(sampleNum)+'.jpg', gray[y: y+h, x: x+w])

    cv2.imshow('frame',frame)
    cv2.waitKey(10)

    if sampleNum > 200:
        break

cap.release()
cv2.destroyAllWindows()