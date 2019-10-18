import numpy as np
import matplotlib.pyplot as plt

batchSize = 20
real_w = 12
real_b = 9
var = 1

np.random.seed(0)
x = np.random.randint(0, 100, size=batchSize)
y = real_w * x + real_b + np.random.randint(-1 * var, var, size=batchSize)

ep = 0.005
i = 0
w = 1
b = 1
alpha = 0.0001
J0 = 1
while i < 1000:
    i += 1
    loss = w * x + b - y
    J = 0.5 * np.dot(loss, loss) / len(x)
    temp1 = np.dot(np.transpose(x), loss) / len(x)
    temp2 = np.mean(loss)
    w -= alpha * temp1
    b -= alpha * temp2 * 10000
    if abs(J0 - J) < ep:
        break
    print('No.%d: J=%.5f, w=%.2f, b=%.2f' % (i, J, w, b))
    J0 = J

print('The estimate line is Y=%.1fX+%.1f' % (w, b))
print('The real line is     Y=%.1fX+%.1f' % (real_w, real_b))
p = np.linspace(0, 100, 50)
q = w * p + b
plt.plot(p, q, color='red')
plt.scatter(x, y)
plt.show()
