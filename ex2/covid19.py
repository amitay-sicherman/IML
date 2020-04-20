import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from numpy.linalg import inv,pinv
from sklearn.linear_model import LinearRegression
import math

def fit_linear_regression(X:np.array, y:np.array):
    s = np.linalg.svd(X,full_matrices=False,compute_uv=False)
    w=np.matmul(np.linalg.pinv(X),y)
    return w,s

def fix_col(col,min_val,max_val):
    a = np.array(col.values.tolist())
    b=np.where(a >= max_val, max_val, a)
    return(np.where(b <= min_val, min_val, b))

def load_data(path):
    X=pd.read_csv(path)
    X['log detected']=np.log(X['detected'])

    X['day_num']=fix_col( X['day_num'],1,365*2)
    X['log detected']=fix_col( X['log detected'],0,10)
    X["b"]=1
    y=X['log detected']
    y2=X['detected']
    X=X.drop(columns=['date','detected','log detected'])
    return X,y,y2

X,y,y2=load_data('/content/drive/My Drive/Colab Notebooks/covid19_israel.csv')

w,s=fit_linear_regression(X,y)
print(f'{X.columns[0]}:{w[0]}')
print(f'{X.columns[1]}:{w[1]}')
xscale=np.linspace(0,40,200)
X_line=pd.DataFrame({'x_line':xscale})
X_line['b']=1

fig, ax = plt.subplots()
ax.plot(X['day_num'],y, '.')
ax.plot(X_line['x_line'],np.matmul(X_line.to_numpy(),w))
plt.title('Log Detected VS Days')
plt.xlabel("days")
plt.ylabel("log detected")
ax.legend(['Measuring points',f'linear fit: Y={w[0]:3.2f}X +{w[1]:3.2f}'])

fig, ax = plt.subplots()
ax.plot(X['day_num'],y2, '.')
ax.plot(X_line['x_line'],np.exp(np.matmul(X_line.to_numpy(),w)))
plt.title('Detected VS Days')
plt.xlabel("days")
plt.ylabel("Detected")
ax.legend(['Measuring points',f'linear fit: Y=e^{math.exp(w[0]):5.2f}X +{math.exp(w[1]):5.2f}'])