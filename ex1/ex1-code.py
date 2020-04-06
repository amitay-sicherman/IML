import numpy as np
import matplotlib.pyplot as plt
import math

row = 1000
col = 100000
data = np.random.binomial(1, 0.25, (col, row))
epsilons = (0.5, 0.25, 0.1, 0.01, 0.001)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr


###############################
###########  11-15  ###########
###############################

def get_orthogonal_matrix(dim):
    H = np.random.randn(dim, dim)
    Q, R = qr(H)
    return Q


def plot_3d(x_y_z):
    '''
    plot points in 3D
    :param x_y_z: the points. numpy array with shape: 3 X num_samples (first dimension for x, y, z
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_y_z[0], x_y_z[1], x_y_z[2], s=1, marker='.', depthshade=False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def plot_2d(x_y):
    '''
    plot points in 2D
    :param x_y_z: the points. numpy array with shape: 2 X num_samples (first dimension for x, y
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_y[0], x_y[1], s=1, marker='.')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def conditional(x_y_z):
    temp = x_y_z[0:2, :]
    temp2 = x_y_z[2, :]
    # cond=temp2>-0.4 and temp2<0.1
    return temp[:, np.logical_and(temp2 > -0.4, temp2 < 0.1)]


# q11
mean = [0, 0, 0]
cov = np.eye(3)
x_y_z = np.random.multivariate_normal(mean, cov, 50000).T
plot_3d(x_y_z);

# q12
s = np.diag([0.1, 0.5, 2])
newData = np.matmul(s, x_y_z)
newCov = np.matmul(np.matmul(s, cov), s.transpose())
print("scale cove:")
print(newCov)
plot_3d(newData)

# q13
ortoM = get_orthogonal_matrix(3)
ortoData = np.matmul(ortoM, newData)
plot_3d(ortoData)
print("orto cove:")
print(np.matmul(np.matmul(ortoM, newCov), ortoM.transpose()))

# q14
plot_2d(x_y_z[0:2, :])

# q15
plot_2d(conditional(x_y_z))

###############################
###########   16    ###########
###############################

# part a:
av = np.zeros(row)
for i in range(5):
    for m in range(1, row):
        av[m] = np.mean(data[i, 1:m])
    plt.plot(av)
plt.show()

# part b:
# Chebyshev
i = 1
upperBoundChebyshev = np.zeros(row)
fig = plt.figure()
ax = fig.add_subplot(111)

for epsilon in epsilons:
    var = np.var(data[i, :])
    for m in range(1, row):
        upperBoundChebyshev[m] = var / (m * epsilon * epsilon)
    plt.plot(upperBoundChebyshev[5:, ], label='epsilon=' + str(epsilon))
legend = ax.legend()
plt.ylim([0, 5])
plt.title("Chebyshev upper bound")

plt.show()

# Hoeffding
i = 1
upperBoundHoeffding = np.zeros(row)
fig = plt.figure()
ax = fig.add_subplot(111)

for epsilon in epsilons:
    var = np.var(data[i, :])
    for m in range(1, row):
        upperBoundHoeffding[m] = 2 * math.exp(-2 * m * epsilon * epsilon)

    plt.plot(upperBoundHoeffding[1:, ], label='epsilon=' + str(epsilon))
legend = ax.legend()
plt.title("Hoeffding upper bound")
plt.show()

# part c
fig = plt.figure()
ax = fig.add_subplot(111)
pres = np.zeros(row)
for epsilon in epsilons:
    print(epsilon)
    for m in range(1, row):
        temp = np.mean(data[:, 0:m], axis=1)
        pres[m] = (col - np.count_nonzero(temp > 0.25 + epsilon) - np.count_nonzero(temp < 0.25 - epsilon)) / col
    plt.plot(pres[1:], label='epsilon=' + str(epsilon))
legend = ax.legend()
plt.title("16 C")

plt.show()
