import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

batchSize = 50
real_w1 = 7
real_w2 = -8
real_b = 30
var = 120

np.random.seed(0)
x1 = np.random.randint(0, 100, size=batchSize)
x2 = np.random.randint(0, 100, size=batchSize)
y = real_w1 * x1 + real_w2 * x2 + real_b + np.random.randint(-1 * var, var, size=batchSize)

ep = 0.001
i = 0
w1 = 1
w2 = 1
b = 1
alpha = 0.0001
J0 = 1
while i < 1000:
    i += 1
    loss = w1 * x1 + w2 * x2 + b - y
    J = 0.5 * np.dot(loss, loss) / len(x1)
    temp1 = np.dot(np.transpose(x1), loss) / len(x1)
    temp2 = np.dot(np.transpose(x2), loss) / len(x2)
    temp3 = np.mean(loss)
    w1 -= alpha * temp1
    w2 -= alpha * temp2
    b -= temp3
    if abs(J0 - J) < ep:
        break
    print('No.%d: J=%.5f, w1=%.2f, w2=%.2f, b=%.2f' % (i, J, w1, w2, b))
    J0 = J

print('The estimate line is Y=%.1fX1+%.1fX2+%.1f' % (w1, w2, b))
print('The real line is     Y=%.1fX1+%.1fX2+%.1f' % (real_w1, real_w2, real_b))

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x1, x2, y)
p1 = np.arange(0, 100, 1)
p2 = np.arange(0, 100, 1)
p1, p2 = np.meshgrid(p1, p2)
q = w1 * p1 + w2 * p2 + b
surf = ax.plot_surface(p1, p2, q, rstride=1, cstride=1, cmap=cm.jet,  linewidth=0, antialiased=True)
plt.show()
