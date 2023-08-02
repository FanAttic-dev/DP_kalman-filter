import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

# Source: https://www.kalmanfilter.net/kalman1d_pn.html

n = 10

measurement_sigma = 0.1

np.random.seed(42)
true = np.linspace(50, 55, n)
Z = true + np.random.normal(0, measurement_sigma, size=n)

# Kalman filter


def get_K(p, r):
    return p / (p + r)


def get_confidence_interval(P, C):
    alpha = 1 - C
    q = 1 - (alpha / 2)
    dof = n-1
    t = st.t.ppf(q, dof)
    ci = t * P / np.sqrt(n)
    return ci


x = 10  # Initial guess
p = 100**2
r = measurement_sigma**2  # measurement variance
q = 0.1
X = np.zeros(n)
P = np.zeros(n)
K = np.zeros(n)
for i, z in enumerate(Z):  # measure
    # update
    k = get_K(p, r)
    x_ = x + k * (z - x)
    p_ = (1 - k) * p

    # predict
    x = x_
    p = p_ + q

    X[i] = x
    P[i] = p
    K[i] = k

    print(f"Iteration {i}: z: {z}, x: {x}, p: {p}")

# Plotting
fig, ax = plt.subplots(2, 1)
fig.set_size_inches(10, 8)
ax[0].plot(np.arange(len(Z)), Z,
           '-*', label='Measurements', color="red")
ax[0].plot(np.arange(len(Z)), true,
           '-d', label="True values", color='green')
ax[0].plot(np.arange(len(Z)), X,
           '-o', label="Estimated values", color='blue')
ci = get_confidence_interval(P, 0.95)
ax[0].fill_between(np.arange(len(Z)), (X - ci), (X + ci), color='b', alpha=.1)
ax[0].legend()

ax[1].plot(K)
ax[1].set_title("Kalman gain")
plt.show()
