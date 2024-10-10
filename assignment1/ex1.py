import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1-y2)**2);

mean = [ 3 , 3 ]
cov = [ [ 1 , 0 ] , [ 0 , 1 ] ]
a = np.random.multivariate_normal (mean, cov , 500).T
mean = [ -3 , -3]
cov = [ [ 2 , 0 ] , [ 0 , 5 ] ]
b = np.random.multivariate_normal(mean, cov , 500). T
c = np.concatenate((a, b) , axis=1 )
c = c.T

idtab = ['a'] * 500 + ['b'] * 500
df = pd.DataFrame(c)
df.insert(2, 'id', idtab, True)
np.random.shuffle(c)
c = c.T
x = c[0]
y = c[1]
id = c[2]

n = 1000
index1 = np.random.randint(0, n - 1)
rx1 = x[index1]
ry1 = y[index1]

rx1path = []
ry1path = []

rx2path = []
ry2path = []

alpha = 0.1
index2 = np.random.randint(0, n - 1)
rx2 = x[index2]
ry2 = y[index2]

def pp1_3(rx1, ry1, rx2, ry2, x, y):
    iter_count = 100
    for iter in range(iter_count):
        print(f"before r1: {rx1}, {ry1}");
        print(f"before r2: {rx2}, {ry2}");
        rx1path.append(rx1)
        ry1path.append(ry1)
        rx2path.append(rx2)
        ry2path.append(ry2)
        for i in range(1000):
            if distance(x[i], y[i], rx1, ry1) < distance(x[i], y[i], rx2, ry2):
                rx1 = (1 - alpha) * rx1 + alpha * x[i];
                ry1 = (1 - alpha) * ry1 + alpha * y[i];
            else: 
                rx2 = (1 - alpha) * rx2 + alpha * x[i];
                ry2 = (1 - alpha) * ry2 + alpha * y[i];
        print(f"after r1: {rx1}, {ry1}");
        print(f"after r2: {rx2}, {ry2}");

iter_count = 100
for iter in range(iter_count):
    print(f"before r1: {rx1}, {ry1}");
    print(f"before r2: {rx2}, {ry2}");
    rx1path.append(rx1)
    ry1path.append(ry1)
    rx2path.append(rx2)
    ry2path.append(ry2)
    dx1 = 0
    dx2 = 0
    dy1 = 0
    dy2 = 0
    for i in range(1000):
        if distance(x[i], y[i], rx1, ry1) < distance(x[i], y[i], rx2, ry2):
            dx1 += x[i] - rx1
            dy1 += y[i] - ry1
        else: 
            dx2 += x[i] - rx2
            dy2 += y[i] - ry2
    rx1 += alpha / n * dx1
    ry1 += alpha / n * dy1

    rx2 += alpha / n * dx2
    ry2 += alpha / n * dy2
    print(f"after r1: {rx1}, {ry1}");
    print(f"after r2: {rx2}, {ry2}");

plt.plot( x , y , 'x' )
plt.plot( a[0] , a[1] , 'x', color='g')
plt.plot( b[0] , b[1] , 'x', color='c')

plt.plot(rx1path, ry1path, 'o', color='r') 
plt.plot(rx2path, ry2path, 'o', color='b') 
plt.plot([rx1, rx2], [ry1, ry2], 'o', color='y') 
plt.axis('equal')
plt.show()
