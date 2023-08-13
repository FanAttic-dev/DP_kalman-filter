import numpy as np
import matplotlib.pyplot as plt

# Source: https://github.com/RahmadSadli/Kalman-Filter/blob/master/KalmanFilter.py


class KalmanFilter():
    def __init__(self, dt, acc, std_acc, std_meas):
        self.dt = dt

        self.x = np.matrix([
            [0],
            [0],
            [acc]
        ])

        self.A = np.matrix([
            [1, dt, dt**2 / 2],
            [0, 1, dt],
            [0, 0, 1]
        ])

        self.H = np.matrix([
            [1, 0, 0]
        ])

        self.Q = np.matrix([
            [dt**4 / 4, dt**3 / 2, dt**2 / 2],
            [dt**3 / 2, dt**2, dt],
            [dt**2 / 2, dt, 1]
        ]) * std_acc**2

        self.P = np.eye(self.A.shape[0])

        self.R = std_meas**2

    def predict(self):
        self.x = np.dot(self.A, self.x)

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
        return K


def main():
    dt = 0.1
    n = 100
    t = np.linspace(0, n, n)

    real_track = dt * (t**2 - t)

    acc = 2
    std_acc = .1  # [m/s^2]
    std_meas = 20  # [m]

    np.random.seed(42)
    measurements = real_track + np.random.normal(0, 50, size=n)

    kf = KalmanFilter(dt, acc, std_acc, std_meas)

    predictions = []
    kalman_gains = []
    for z in measurements:
        x_ = kf.predict()
        predictions.append(x_[0].item())

        K = kf.update(z)
        kalman_gains.append(K[2].item())

    fig, ax = plt.subplots(2, 1)
    fig.set_size_inches(10, 8)
    ax[0].plot(t, real_track, label="True", color="green")
    ax[0].plot(t, measurements, label="Measured", color="red")
    ax[0].plot(t, predictions, label="Predicted", color="blue")
    ax[0].legend()
    ax[1].set_title("Kalman gain")
    ax[1].plot(t, kalman_gains)
    plt.show()


main()
