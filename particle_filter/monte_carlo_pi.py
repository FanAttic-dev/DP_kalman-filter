import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform


N = 20000
r = 1
area = (2*r)**2

pts = uniform(-1, 1, (N, 2))

dist = np.linalg.norm(pts, axis=1)
in_circle = dist <= 1

pts_in_circle = np.count_nonzero(in_circle)
pi = 4 * (pts_in_circle / N)

print(f'mean pi(N={N})= {pi:.4f}')
print(f'err  pi(N={N})= {abs(np.pi-pi):.4f}')

# plot results
plt.scatter(pts[in_circle, 0], pts[in_circle, 1],
            marker=',', edgecolor='k', s=1)
plt.scatter(pts[~in_circle, 0], pts[~in_circle, 1],
            marker=',', edgecolor='r', s=1)
plt.axis('equal')
plt.show()
