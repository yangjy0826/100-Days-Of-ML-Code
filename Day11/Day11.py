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


from sklearn.neighbors import KNeighborsClassifier
k =5  # k is the number of nearest neighbor
neigh = KNeighborsClassifier(n_neighbors=k)
neigh.fit(X_train, Y_train)

Y_pred = neigh.predict(X_test)
# print(Y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
# print(cm)
