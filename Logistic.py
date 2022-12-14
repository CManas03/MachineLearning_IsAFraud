import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import classification_report, confusion_matrix

test_identity = pd.read_csv('nstest.csv')
train_identity = pd.read_csv('nstrain.csv')

# Logistic Regression
# Logistic Regression is a generalized Linear Regression in the sense that we don’t output the weighted sum of inputs directly, but we pass it through a function that can map any real value between 0 and 1.
# First we split the dataset into X and y.

train_identity.drop(axis='columns',labels='had_id',inplace=True)
test_identity.drop(axis='columns',labels='had_id',inplace=True)
#Separate the target variable and rest of the variables.
X = train_identity.loc[:,train_identity.columns!='isFraud']
y = train_identity['isFraud']

# Train Test Split
# from sklearn.model_selection import train_test_split
# # implementing train-test-split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=66)

# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression(solver = 'saga')
logisticRegr.fit(X, y)
predictions = logisticRegr.predict(test_identity)
pd.DataFrame(predictions).to_csv('mylgroutput1.csv')

# # Use score method to get accuracy of model
# score = logisticRegr.score(X_test, y_test)
# print(score)

# # Cross validation 
# lgr_cv_score = cross_val_score(logisticRegr, X, y, cv=5, scoring='roc_auc')

# print(lgr_cv_score)
