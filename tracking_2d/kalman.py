import numpy as np


class KalmanFilter():
    def __init__(self, dt, acc_x, acc_y, std_acc, std_measurement):
        self.dt = dt

        self.x = np.matrix([
            [0],  # x
            [0],  # x'
            [acc_x],  # x''
            [0],  # y
            [0],  # y'
            [acc_y]  # y''
        ])

        self.A = np.matrix([
            [1, dt, dt**2 / 2, 0, 0, 0],
            [0, 1, dt, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, dt, dt**2 / 2],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1]
        ])

        self.H = np.matrix([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])

        self.Q = np.matrix([
            [dt**4 / 4, dt**3 / 2, dt**2 / 2, 0, 0, 0],
            [dt**3 / 2, dt**2, dt, 0, 0, 0],
            [dt**2 / 2, dt, 1, 0, 0, 0],
            [0, 0, 0, dt**4 / 4, dt**3 / 2, dt**2 / 2],
            [0, 0, 0, dt**3 / 2, dt**2, dt],
            [0, 0, 0, dt**2 / 2, dt, 1]
        ]) * std_acc**2

        self.P = np.eye(self.A.shape[1])

        self.R = std_measurement**2

    def predict(self):
        self.x = np.dot(self.A, self.x)

        # P = A * P * A' + Q
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        return self.x[0].item(), self.x[3].item()

    def update(self, x_meas, y_meas):
        # K = P * H' * inv(H * P * H' + R)
        HPH = np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(HPH + self.R)))

        z = np.matrix([
            [x_meas],
            [y_meas]
        ])
        self.x = self.x + np.dot(K, z - np.dot(self.H, self.x))

        I = np.eye(self.H.shape[1])
        I_KH = I - np.dot(K, self.H)
        KRK = np.dot(K, np.dot(self.R, K.T))
        # P = (I - K * H) * P * (I - K * H)' + K * R * K
        self.P = np.dot(I_KH, np.dot(self.P, I_KH.T)) + KRK
        return K
