import numpy as np
import matplotlib.pyplot as plt

# Source: https://github.com/RahmadSadli/Kalman-Filter/blob/master/KalmanFilter.py


class KalmanFilter():
    def __init__(self, dt, std_acc, std_measurement):
        self.dt = dt

        self.x = np.matrix([
            [0],
            [0]
        ])

        self.A = np.matrix([
            [1, dt],
            [0, 1]
        ])

        self.B = np.matrix([
            [dt**2 / 2],
            [dt]
        ])

        self.H = np.matrix([
            [1, 0]
        ])

        self.Q = np.matrix([
            [dt**4 / 4, dt**3 / 2],
            [dt**3 / 2, dt**2]
        ]) * std_acc**2

        self.P = np.eye(self.A.shape[0])

        self.R = std_measurement**2

    def predict(self, u):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)

        # P = A * P * A' + Q
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        return self.x

    def update(self, z):
        # K = P * H' * inv(H * P * H' + R)
        HPH = np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(HPH + self.R)))

        self.x = self.x + np.dot(K, z - np.dot(self.H, self.x))

        I = np.eye(self.H.shape[1])
        I_KH = I - np.dot(K, self.H)
        KRK = np.dot(K, np.dot(self.R, K.T))
        # P = (I - K * H) * P * (I - K * H)' + K * R * K
        self.P = np.dot(I_KH, np.dot(self.P, I_KH.T)) + KRK


def main():
    dt = 0.1
    n = 100
    t = np.linspace(0, n, n)

    real_track = dt * (t**2 - t)

    acc = 20
    std_acc = 10  # [m/s^2]
    std_measurement = 50  # [m]

    np.random.seed(42)
    measurements = real_track + np.random.normal(0, 50, size=n)

    kf = KalmanFilter(dt, std_acc, std_measurement)

    predictions = []
    for z in measurements:
        x_ = kf.predict(u=acc)
        predictions.append(x_[0].item())

        kf.update(z)

    fig, ax = plt.subplots()
    ax.plot(t, real_track, label="True", color="green")
    ax.plot(t, measurements, label="Measured", color="red")
    ax.plot(t, predictions, label="Predicted", color="blue")
    ax.legend()
    plt.show()


main()
