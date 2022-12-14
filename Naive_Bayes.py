import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

train_identity = pd.read_csv('nstrain.csv')

train_identity.drop(axis='columns',labels='had_id',inplace=True)
#Separate the target variable and rest of the variables.
X = train_identity.loc[:,train_identity.columns!='isFraud']
y = train_identity['isFraud']

#Train Test Split
# implementing train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=66)

# Naive Bayes
# The Naive Bayes method makes the assumption that the predictors contribute equally and independently to selecting the output class. Although the Naive Bayes model’s assumption that all predictors are independent of one another is unfeasible in real-world circumstances, this assumption produces a satisfactory outcome in the majority of instances.

# Gaussian Naive Bayes
# This classifier is employed when the predictor values are continuous and are expected to follow a Gaussian distribution.

nb = GaussianNB()
nb.fit(X_train, y_train)
print("Naive Bayes score: ",nb.score(X_test, y_test))
