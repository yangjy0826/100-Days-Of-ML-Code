import pandas as pd
import numpy as np

df = pd.read_csv('Social_Network_Ads.csv')
# print(df)
X = df.iloc[:, 2:4].values
Y = df.iloc[:, 4].values
# print(X)
# print(Y)

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
# print(X_train)

# feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
# print(X_train)
# print(X_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr = lr.fit(X_train, Y_train)

Y_pred = lr.predict(X_test)
# print(Y_pred)

from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, Y_pred)
# print(confusion_matrix(Y_test, Y_pred))

# import matplotlib.pyplot as plt
# # print(X_train.shape)
# # print(Y_train)
# # for i in range(len(lr.predict(X_train))):
# #     if lr.predict(X_train)[i] == 0:
# #         str = 'blue'
# #     else:
# #         str = 'red'
# plt.scatter(X_train[:, 0], X_train[:, 1], color = str)
# plt.plot(X_train, lr.predict(X_train), color ='green')
# plt.show()
# # plt.scatter(X_test, Y_test, color = 'red')
# # plt.plot(X_test, lr.predict(X_test), color ='blue')
# # plt.show()
