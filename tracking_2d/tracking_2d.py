from constants import colors
import detector
from kalman import KalmanFilterAcc
# from kalman_control import KalmanFilter

import cv2

cap = cv2.VideoCapture("./tracking_2d/video/video_randomball.avi")
kf = KalmanFilterAcc(dt=0.1, std_acc=1, std_measurement=10)

i = 0
is_paused = False
while True:
    ret, frame = cap.read()
    if not ret:
        break

    circle = detector.detect(frame, False)

    x_pred, y_pred = kf.predict(decelerate=is_paused)
    cv2.rectangle(
        frame,
        (int(x_pred - circle["radius"]), int(y_pred - circle["radius"])),
        (int(x_pred + circle["radius"]), int(y_pred + circle["radius"])),
        colors["blue"],
        thickness=2
    )

    if circle is not None and not is_paused:
        # Detected circle
        cv2.circle(frame, (circle["x"], circle["y"]),
                   radius=circle["radius"], color=colors['red'], thickness=2)

        kf.update(circle["x"], circle["y"])

    cv2.imshow('image', frame)
    kf.print()

    key = cv2.waitKey(100)
    if key == ord('q'):
        break
    elif key == ord('n'):
        is_paused = not is_paused

    i += 1


cap.release()
cv2.destroyAllWindows()
