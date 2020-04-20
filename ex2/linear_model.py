import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from numpy.linalg import inv,pinv
from sklearn.linear_model import LinearRegression

def fit_linear_regression(X:np.array, y:np.array):
    s = np.linalg.svd(X,full_matrices=False,compute_uv=False)
    w=np.matmul(np.linalg.pinv(X),y)
    return w,s

def predict(X:np.array, w:np.array):
    return np.matmul(X,w)

def mse(y_hat:np.array,y:np.array):
    return np.mean(np.power(y_hat-y,2))

def fix_col(col,min_val,max_val):
    a = np.array(col.values.tolist())
    b=np.where(a >= max_val, max_val, a)
    return(np.where(b <= min_val, min_val, b))

def load_data(path,get_cols=False):
    X=pd.read_csv(path)
    X.fillna(0, inplace = True)

    y=X['price']

    X['bedrooms']=fix_col( X['bedrooms'],0.5,15)
    X['bathrooms']=fix_col( X['bathrooms'],1,15)
    X['sqft_living']=fix_col( X['sqft_living'],5,1000000)
    X['sqft_lot']=fix_col( X['sqft_lot'],5,1000000)
    X['floors']=fix_col( X['floors'],1,10)
    X['waterfront']=fix_col( X['waterfront'],0,1)
    X['view']=fix_col( X['view'],0,4)
    X['condition']=fix_col( X['condition'],1,10)
    X['sqft_above']=fix_col( X['sqft_above'],5,1000000)
    X['sqft_basement']=fix_col( X['sqft_basement'],5,1000000)
    X['yr_built']=fix_col( X['yr_built'],1900,2020)
    X['yr_renovated']=fix_col( X['yr_renovated'],1900,2020)
    X['yr_built']=fix_col( X['yr_built'],1900,2020)
    X['sqft_living15']=fix_col( X['sqft_living15'],5,10000)
    X['sqft_lot15']=fix_col( X['sqft_lot15'],5,100000)
    X['balance']=1

    a = np.array(X['yr_built'].values.tolist())
    b=np.where(a >= 2020, 2020, a)
    X['yr_built'] = np.where(b <= 1900, 1900, a)


    # categorical features
    zip_df=pd.get_dummies(X["zipcode"])
    grade_df=pd.get_dummies(X["grade"])
    X=X.drop(columns=['id','date','price','zipcode','grade','lat','long'])
    non_car_cols=X.columns

    X=X.join(zip_df)
    X=X.drop(columns=[0.0])
    X=X.join(grade_df)
    if get_cols:
        return X,y,non_car_cols
    return X,y


def plot_singular_values(s:np.array):
    fig, ax = plt.subplots()
    s=np.sort(s)
    ax.plot(s[::-1], '.')
    plt.title("singular_values")
    plt.ylabel("Value")
    plt.xlabel("Index")
    plt.ylim([0,50])

#Q15
X,y=load_data('/content/drive/My Drive/Colab Notebooks/kc_house_data.csv')
w,s=fit_linear_regression(X.to_numpy(),y.to_numpy())
plot_singular_values(s)

#Q16:
X_df,y_df,cols=load_data('/content/drive/My Drive/Colab Notebooks/kc_house_data.csv',True)
X=X_df.to_numpy()
y=y_df.to_numpy()
train, test=train_test_split(range(0, X.shape[0]))
res=[]
for p in range(1,1000):
    end=int(0.001*p*X.shape[0])
    X_train, X_test, y_train, y_test = train_test_split(X[0:end,],y[0:end])
    w,s=fit_linear_regression(X_train,y_train)
    res.append(mse(predict(X_test,w),y_test))
fig, ax = plt.subplots()
plt.plot(res)
plt.title("MSE value VS data % taken")
plt.xlabel("% from data")
plt.ylabel("MSE value")

fig, ax = plt.subplots()
plt.plot(np.linspace(0,100,999),res)
plt.title("MSE value VS data % taken")
plt.xlabel("% from data")
plt.ylabel("MSE value")


def feature_evaluation(df_non_categorical:pd.DataFrame,response:pd.DataFrame):
    for col in df_non_categorical.columns:
        fig, ax = plt.subplots()

        ax.plot(df_non_categorical[col].to_numpy(),y, '.')
        plt.title("Correlation between " + col+ " and Price"+"(cor="+
                  "{:.2f}".format(np.cov(df_non_categorical[col].to_numpy(),y)[1,0]/(np.std(df_non_categorical[col].to_numpy())*np.std(y)))+")")

        plt.ylabel("Price")
        plt.xlabel(str(col))

        plt.savefig('/content/drive/My Drive/Colab Notebooks/'+str(col))

# Q17
feature_evaluation(X_df[cols],y_df)