import numpy as np
import matplotlib.pyplot as plt
import math
row=1000
col=100000
data = np.random.binomial(1, 0.25, (col,row))
epsilons=(0.5, 0.25, 0.1, 0.01, 0.001)

# part a:
av=np.zeros(row)
for i in range(5):
    for m in range(1,row):
        av[m]=np.mean(data[i,1:m])
    plt.plot(av)
plt.show()

# part b:
# Chebyshev
i=1
upperBoundChebyshev=np.zeros(row)
fig = plt.figure()
ax = fig.add_subplot(111)

for epsilon in epsilons:
    var=np.var(data[i,:])
    for m in range(1,row):

        upperBoundChebyshev[m]=var/(m*epsilon*epsilon)
    plt.plot(upperBoundChebyshev[5:,],label='epsilon='+str(epsilon))
legend = ax.legend()
plt.ylim([0,5])
plt.title("Chebyshev upper bound")

plt.show()

# Hoeffding
i=1
upperBoundHoeffding=np.zeros(row)
fig = plt.figure()
ax = fig.add_subplot(111)

for epsilon in epsilons:
    var=np.var(data[i,:])
    for m in range(1,row):

        upperBoundHoeffding[m]=2*math.exp(-2*m*epsilon*epsilon)

    plt.plot(upperBoundHoeffding[1:,],label='epsilon='+str(epsilon))
legend = ax.legend()
plt.title("Hoeffding upper bound")
plt.show()

# part c
fig = plt.figure()
ax = fig.add_subplot(111)
pres=np.zeros(row)
for epsilon in epsilons:
    print(epsilon)
    for m in range(1,row):
        temp=np.mean(data[:,0:m],axis=1)
        pres[m]=(col-np.count_nonzero(temp > 0.25+epsilon) -np.count_nonzero(temp < 0.25-epsilon))/col
    plt.plot(pres[1:],label='epsilon='+str(epsilon))
legend = ax.legend()
plt.title("16 C")

plt.show()
