import cv2
import numpy as np
from collections import deque

cap = cv2.VideoCapture(1)

def nothing(x):
    pass

pts = deque(maxlen=1200)
cv2.namedWindow('bar frame')
cv2.createTrackbar('H_Upper', 'bar frame', 82, 100, nothing)
cv2.createTrackbar('S_Upper', 'bar frame', 182, 255, nothing)
cv2.createTrackbar('V_Upper', 'bar frame', 169, 255, nothing)
cv2.createTrackbar('H_Lower', 'bar frame', 23, 100, nothing)
cv2.createTrackbar('S_Lower', 'bar frame', 32, 255, nothing)
cv2.createTrackbar('V_Lower', 'bar frame', 0, 255, nothing)


#lower_green = np.array([40, 100, 50])
#upper_green = np.array([80, 255, 255])


while(1):

    _, frame = cap.read()
    frame=cv2.flip(frame,1)


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hu = cv2.getTrackbarPos('H_Upper', 'bar frame')
    su = cv2.getTrackbarPos('S_Upper', 'bar frame')
    vu = cv2.getTrackbarPos('V_Upper', 'bar frame')
    hl = cv2.getTrackbarPos('H_Lower', 'bar frame')
    sl = cv2.getTrackbarPos('S_Lower', 'bar frame')
    vl = cv2.getTrackbarPos('V_Lower', 'bar frame')
    lower_green = np.array([hl, sl, vl])
    upper_green = np.array([hu, su, vu])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    res = cv2.bitwise_not(frame.copy(),frame.copy(), mask= mask)
    center = None
    if len(cnts) > 0:
        for c in cnts:
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 0), 5)
                #cv2.circle(res, center, 5, (0, 0, 255), -1)
                pts.appendleft(center)

#        c = max(cnts, key=cv2.contourArea)


    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(1200 / float(i + 1)) * 1)
        cv2.line(res, pts[i - 1], pts[i], (0, 255, 255), thickness)

    cv2.imshow("original frame", frame)
    cv2.imshow("tracking",mask)
    cv2.imshow("bar frame",frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
