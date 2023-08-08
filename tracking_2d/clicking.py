import numpy as np
import cv2
from constants import colors
from kalman import KalmanFilterAcc, KalmanFilterVel

H, W, D = 512, 512, 3
# H, W, D = 1000, 6500, 3
CIRCLE_RADIUS = 10


def on_mouse_click(event, x, y, flags, param):
    global frame, mouse_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pos = (x, y)
    elif event == cv2.EVENT_MBUTTONDOWN:
        mouse_pos = None


def on_std_meas_trackbar_change(std_meas):
    std_meas /= 10
    print(f"std_meas: {std_meas}")
    kf.set_R(std_meas)


def on_std_acc_trackbar_change(std_acc):
    std_acc /= 100
    print(f"std_acc: {std_acc}")
    kf.set_Q(std_acc)


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


# kf = KalmanFilterAcc(dt=0.1, std_acc=0.1, std_measurement=50)
kf = KalmanFilterVel(dt=0.1, std_acc=0.1, std_measurement=50)

frame = get_blank_frame()
mouse_pos = (0, 0)

window_name = "Frame"
# window_flags = cv2.WINDOW_NORMAL
window_flags = cv2.WINDOW_AUTOSIZE
cv2.namedWindow(window_name, window_flags)
cv2.setMouseCallback(window_name, on_mouse_click)
cv2.createTrackbar("std meas", window_name, 100,
                   100, on_std_meas_trackbar_change)
cv2.createTrackbar("std acc", window_name, 10,
                   100, on_std_acc_trackbar_change)


while True:
    cv2.imshow(window_name, frame)
    frame = get_blank_frame()
    measurement_available = mouse_pos is not None

    # kf.decelerate = not measurement_available
    kf.predict()

    draw_pred()
    kf.print()

    if measurement_available:
        kf.update(*mouse_pos)
        draw_meas()

    key = cv2.waitKey(0)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
