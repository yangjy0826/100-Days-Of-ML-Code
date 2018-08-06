import pandas as pd
import numpy as np

df = pd.read_csv('50_Startups.csv')
# print(df)
X = df.iloc[:, :4].values
Y = df.iloc[:, 4].values
# print(X)
# print(Y)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# integer encode
le = LabelEncoder()  # label encoder
X[:, 3] = le.fit_transform(X[:, 3])
# binary encode
ohe = OneHotEncoder()  # onehot encoder
X1 = X[:, 3].reshape(-1, 1)
X1 = ohe.fit_transform(X1).toarray()  # Expected 2D array,所以把一个shape是(50,)的矩阵转换成(50,1)
# print(X1)
# print(X[:, :3])
# print(X1.shape)
# print(X[:, :3].shape)
X = np.hstack((X[:, :3], X1))  # OneHot后的最后一列与之前两列拼接起来
# print(X)

X = X[:, :5]  # Avoid dummmy variables trap
# print(X)

###################################################

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
# print(X_train)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)
# print(regressor.coef_)
# print(regressor.intercept_)

###################################################

Y_pred = regressor.predict(X_test)
# print(Y_pred)
