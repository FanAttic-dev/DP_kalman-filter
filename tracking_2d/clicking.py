import numpy as np
import cv2
from constants import colors
from kalman import KalmanFilter


def on_mouse_click(event, x, y, flags, param):
    global frame, mouse_x, mouse_y
    mouse_x, mouse_y = None, None
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = x, y
        frame = get_blank_frame()
        cv2.circle(frame, (x, y), radius=10,
                   color=colors["white"], thickness=-1)
    elif event == cv2.EVENT_MBUTTONDOWN:
        frame = get_blank_frame()


def get_blank_frame():
    # h, w, d = 1000, 6500, 3
    h, w, d = 512, 512, 3
    return np.zeros((h, w, d), np.uint8)


frame = get_blank_frame()
mouse_x, mouse_y = 0, 0

window_name = "Frame"
window_flags = cv2.WINDOW_AUTOSIZE  # cv2.WINDOW_NORMAL
cv2.namedWindow(window_name, window_flags)
cv2.setMouseCallback(window_name, on_mouse_click)

kf = KalmanFilter()  # TODO

while True:
    cv2.imshow(window_name, frame)

    key = cv2.waitKey(20)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
