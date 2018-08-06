import pandas as pd
df = pd.read_csv('studentscores.csv')
# print(df)
X = df.iloc[:, :1].values
# X = df.iloc[:, 0].values 不能是这句，因为这样得到的是1D的向量，而后面的regressor.fit函数必须是2D的输入
Y = df.iloc[:, 1].values
# print(X)
# print(Y)

# from sklearn.preprocessing import Imputer
# imp = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
# imp.fit(X)
# X[:, 1:] = imp.transform(X)
# print(X)

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
# print(X_train)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)
# print(regressor.coef_)
# print(regressor.intercept_)

Y_pred = regressor.predict(X_test)
# print(Y_pred)

import matplotlib.pyplot as plt
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color ='blue')
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_test, regressor.predict(X_test), color ='blue')
plt.show()