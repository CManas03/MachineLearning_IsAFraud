import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import os
from sklearn.neighbors import KNeighborsClassifier

train_identity = pd.read_csv('nstrain.csv')
train_identity.head()
train_identity.drop(axis='columns',labels='had_id',inplace=True)
# First we split the data into X and y
X = train_identity.loc[:,train_identity.columns!='isFraud']
y = train_identity['isFraud']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=145)
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# KNN Classifier
# The K-nearest Neighbors (KNN) algorithm is a type of supervised machine learning algorithm used for classification, regression as well as outlier detection. It is extremely easy to implement in its most basic form but can perform fairly complex tasks. It is a lazy learning algorithm since it doesn't have a specialized training phase.
# KNN also doesn't assume anything about the underlying data characteristics, it doesn't expect the data to fit into some type of distribution, such as uniform, or to be linearly separable. This means it is a non-parametric learning algorithm. This is an extremely useful feature since most of the real-world data doesn't really follow any theoretical assumption.


knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
