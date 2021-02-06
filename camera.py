import cv2
import numpy as np
from scipy import stats
import time

cap = cv2.VideoCapture(1)
kernel = np.ones((5,5),np.uint8)
clahe = cv2.createCLAHE(clipLimit = 2) 

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    gray = cv2.medianBlur(gray, 7)
    gray = clahe.apply(gray)
    #gray = cv2.filter2D(gray, cv2.CV_8UC3, g_kernel)
    ret2, th3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gray = cv2.bitwise_and(gray, gray, mask = th3)
    th4 = cv2.adaptiveThreshold(gray,20,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
    th4 = cv2.erode(th4,kernel,iterations = 1)
    th4 = cv2.dilate(th4, kernel)
    th4 = cv2.erode(th4,kernel,iterations = 1)
    img3 = cv2.hconcat([gray, th4])
    cv2.imshow('frame', img3)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()