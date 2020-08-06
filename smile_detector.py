import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('C:\\Users\\honey\\PycharmProjects\\smile_detection-master\\haarcascade_frontalface_default.xml')
smileCascade = cv2.CascadeClassifier('C:\\Users\\honey\\PycharmProjects\\smile_detection-master\\haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier("C:\\Users\\honey\\PycharmProjects\\smile_detection-master\\haarcascade_eye.xml")
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    '''gray,
        scaleFactor=1.3,
        minNeighbors=5,      
        minSize=(30,30)
    )'''

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ex, ey + eh), (0, 255, 0), 2)

        smile = smileCascade.detectMultiScale(roi_gray, 1.3, 5)
        '''roi_gray,
            scaleFactor= 1.5,
            minNeighbors=15,
            minSize=(25, 25),
            )'''

        for i in smile:
            if len(smile) > 1:
                cv2.putText(img, "Smiling", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('video', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
