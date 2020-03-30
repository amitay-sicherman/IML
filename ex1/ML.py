import numpy as np
c = np.array([[5, 5], [-1, 7]])
cct=np.matmul(c,np.transpose(c))
print(cct)
d,v=np.linalg.eig(cct)
d=np.diag(d)
u=c*v*np.linalg.inv(d)


print(u)


