import numpy as np
import matplotlib.pyplot as plt

batchSize = 50  #生成随机点个数
real_w = -5     #设定weight值
real_b = 60     #设定bias值
var = 20        #设定随机点的偏差值

np.random.seed(1)       #设定用于生成随机数的seed，特定的seed值对应于特定的随机数列
x = np.random.randint(0, 100, size=batchSize)   #生成0到100的batchSize个随机整数
y = real_w * x + real_b + np.random.randint(-1 * var, var, size=batchSize)  #生成y

ep = 0.001  #设定结束循环所需的阈值，也即两个循环内Cost的变化值
i = 0       #循环计数
w = 1       #初始化weight值
b = 1       #初始化bias值
alpha = 0.0001  #设定learning rate
J0 = 1      #用于记录上个循环的Cost值
while i < 1000:
    i += 1
    loss = w * x + b - y
    J = 0.5 * np.dot(loss, loss) / len(x)           #计算Cost值
    temp1 = np.dot(np.transpose(x), loss) / len(x)  #Cost方程对weight偏导
    temp2 = np.mean(loss)                           #Cost方程对bias偏导
    w -= alpha * temp1          #更新weight值
    b -= alpha * temp2 * 10000  #更新bias值
    if abs(J0 - J) < ep:        #判断是否达到循环结束条件
        break
    print('No.%d: J=%.5f, w=%.2f, b=%.2f' % (i, J, w, b))
    J0 = J      #记录该循环的Cost值

print('The estimate line is Y=%.1fX+%.1f' % (w, b))
print('The real line is     Y=%.1fX+%.1f' % (real_w, real_b))
p = np.linspace(0, 100, 50)
q = w * p + b
plt.plot(p, q, color='red')     #画拟合出的直线
plt.scatter(x, y)       #画生成的原始随机点
plt.show()
