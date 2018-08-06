import numpy as np
import pandas as pd

df = pd.read_csv('Data.csv')
X = df.iloc[:, :3].values
Y = df.iloc[:, 3].values
# print(X)
# print(Y)
# print(X[:, 1:])

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
imp.fit(X[:, 1:])
X[:, 1:] = imp.transform(X[:, 1:])
# print(X)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# print(le.fit(X[:,0]))
# 上一行得到的结果是：LabelEncoder()
# print(list(le.classes_))
# 上一行得到的结果是：['France', 'Germany', 'Spain']
# print(le.transform(list(le.classes_)))
# 上一行得到的结果是：[0 1 2]
# print(list(le.inverse_transform([0, 1, 2])))
# 上一行得到的结果是：['France', 'Germany', 'Spain']
X[:, 0] = le.fit_transform(X[:, 0])
# print(X)

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
# print(X_train)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
# print(X_train)
# print(X_test)
