import cv2
import matplotlib.pyplot as plt
import util

cap = cv2.VideoCapture(0)


array = []
while True:
    req,frame= cap.read()
    x,y,c = frame.shape
    frame = cv2.flip(frame,1)
    frame = util.Objectdetect(frame)
    cv2.imshow("frame",frame)
    # cv2.setWindowProperty("frame", cv2.WND_PROP_TOPMOST, 1)

    if (cv2.waitKey(1) & 0xFF ) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
