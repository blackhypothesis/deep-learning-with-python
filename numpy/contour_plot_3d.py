from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

def fun(x, y):
    return np.sin(np.sqrt(x**2 + y**2))

x = np.linspace(-10, 10, 40)
y = np.linspace(-10, 10, 40)
X, Y = np.meshgrid(x, y)
Z = fun(X, Y)

fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cmap='cool', alpha=0.8)

ax.set_title('3D Contour Plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()