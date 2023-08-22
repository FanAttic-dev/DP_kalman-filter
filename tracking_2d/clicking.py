import numpy as np
import cv2
from constants import colors
from kalman import KalmanFilterVel
from dynamics import Dynamics

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
    model.std_meas = std_meas
    model.init_R()


def on_std_acc_trackbar_change(std_acc):
    std_acc /= 100
    print(f"std_acc: {std_acc}")
    model.std_acc = std_acc
    model.init_Q()


def on_alpha_trackbar_change(alpha):
    alpha /= 100
    print(f"alpha: {alpha}")
    model.set_alpha(alpha)


def on_dt_trackbar_change(dt):
    dt /= 100
    print(f"dt: {dt}")
    model.set_dt(dt)


def get_blank_frame():
    return np.zeros((H, W, D), np.uint8)


def draw_pred():
    pred_x, pred_y = model.pos
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


# model = KalmanFilterAcc(dt=0.1, std_acc=0.1, std_meas=10)
# model = KalmanFilterVelCtrlAcc(dt=0.1, std_acc=0.1, std_meas=10)
model = KalmanFilterVel(dt=0.1, std_acc=0.1, std_meas=50)
# model = Dynamics(dt=0.1, alpha=0.01)

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
# cv2.createTrackbar("alpha", window_name, 10,
#                    100, on_alpha_trackbar_change)
# cv2.createTrackbar("dt", window_name, 10,
#                    100, on_dt_trackbar_change)


while True:
    cv2.imshow(window_name, frame)
    frame = get_blank_frame()
    measurement_available = mouse_pos is not None

    model.set_decelerating((not measurement_available))
    model.predict()

    draw_pred()
    model.print()

    if measurement_available:
        model.update(*mouse_pos)
        draw_meas()

    key = cv2.waitKey(0)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
