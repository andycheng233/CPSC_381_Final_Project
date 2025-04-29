import cv2
import numpy as np

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Micro-Expression Detection", frame)
