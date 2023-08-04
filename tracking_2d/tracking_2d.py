import detector
# from kalman import KalmanFilter
from kalman_control import KalmanFilter

import cv2


colors = {
    "yellow": (0, 191, 255),
    "blue": (255, 0, 0)
}

cap = cv2.VideoCapture("./tracking_2d/video/video_randomball.avi")
kf = KalmanFilter(dt=0.1, acc_x=0.1, acc_y=0.1,
                  std_acc=1, std_measurement=10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    circle = detector.detect(frame, False)

    if circle is not None:
        # Detected circle
        cv2.circle(frame, (circle["x"], circle["y"]),
                   radius=circle["radius"], color=colors['yellow'], thickness=2)

        x_pred, y_pred = kf.predict()

        cv2.rectangle(frame, (int(x_pred - circle["radius"]), int(y_pred - circle["radius"])),
                      (int(x_pred + circle["radius"]), int(y_pred + circle["radius"])), colors["blue"], thickness=2)

        kf.update(circle["x"], circle["y"])

    cv2.imshow('image', frame)

    key = cv2.waitKey(100)
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
