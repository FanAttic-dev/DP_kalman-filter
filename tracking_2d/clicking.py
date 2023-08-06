import numpy as np
import cv2
from constants import colors
from kalman import KalmanFilter

H, W, D = 512, 512, 3
CIRCLE_RADIUS = 10


def on_mouse_click(event, x, y, flags, param):
    global frame, mouse_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pos = (x, y)
    elif event == cv2.EVENT_MBUTTONDOWN:
        mouse_pos = None


def get_blank_frame():
    return np.zeros((H, W, D), np.uint8)


def draw_pred():
    pred_x, pred_y = kf.pos
    cv2.rectangle(
        frame,
        (int(pred_x - CIRCLE_RADIUS), int(pred_y - CIRCLE_RADIUS)),
        (int(pred_x + CIRCLE_RADIUS), int(pred_y + CIRCLE_RADIUS)),
        colors["red"],
        thickness=2
    )


def draw_meas():
    cv2.circle(
        frame,
        mouse_pos,
        radius=CIRCLE_RADIUS,
        color=colors["white"],
        thickness=-1
    )


frame = get_blank_frame()
mouse_pos = (0, 0)

window_name = "Frame"
window_flags = cv2.WINDOW_AUTOSIZE  # cv2.WINDOW_NORMAL
cv2.namedWindow(window_name, window_flags)
cv2.setMouseCallback(window_name, on_mouse_click)

kf = KalmanFilter(dt=0.1, acc_x=0, acc_y=0, std_acc=0.1, std_measurement=50)

while True:
    cv2.imshow(window_name, frame)
    frame = get_blank_frame()
    measurement_available = mouse_pos is not None

    kf.predict(decelerate=(not measurement_available))

    draw_pred()

    if measurement_available:
        kf.update(*mouse_pos)
        draw_meas()

    key = cv2.waitKey(0)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
