import detector
# from kalman import KalmanFilter
from kalman_control import KalmanFilter

import cv2


colors = {
    "yellow": (0, 191, 255),
    "blue": (255, 0, 0),
    "red": (0, 0, 255)
}

cap = cv2.VideoCapture("./tracking_2d/video/video_randomball.avi")
kf = KalmanFilter(dt=0.1, acc_x=1, acc_y=1,
                  std_acc=1, std_measurement=2)

i = 0
pause_interval = 100
pause_length = 50
pause_counter = 0
is_paused = False
while True:
    ret, frame = cap.read()
    if not ret:
        break

    circle = detector.detect(frame, False)

    x_pred, y_pred = kf.predict()
    cv2.rectangle(frame, (int(x_pred - circle["radius"]), int(y_pred - circle["radius"])),
                  (int(x_pred + circle["radius"]), int(y_pred + circle["radius"])), colors["blue"], thickness=2)

    if i % pause_interval == 0 or pause_counter > pause_length:
        is_paused = not is_paused
        pause_counter = 0 if is_paused else 1

    if circle is not None and not is_paused:
        # Detected circle
        cv2.circle(frame, (circle["x"], circle["y"]),
                   radius=circle["radius"], color=colors['red'], thickness=2)

        kf.update(circle["x"], circle["y"])

    if is_paused:
        pause_counter += 1

    cv2.imshow('image', frame)

    print(f"X velocity: {kf.x[1]}")

    key = cv2.waitKey(100)
    if key == ord('q'):
        break

    i += 1


cap.release()
cv2.destroyAllWindows()
