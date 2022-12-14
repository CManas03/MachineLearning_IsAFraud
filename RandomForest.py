import numpy as np # linear algebra
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

train_identity = pd.read_csv('nstrain')
test_identity = pd.read_csv('nstest')
train_identity.drop(axis='columns',labels='had_id',inplace=True)
test_identity.drop(axis='columns',labels='had_id',inplace=True)

# Random forests algorithms are used for classification and regression. The random forest is an ensemble learning method, composed of multiple decision trees. By averaging out the impact of several decision trees, random forests tend to improve prediction.

# Random forests tend to shine in scenarios where a model has a large number of features that individually have weak predicative power but much stronger power collectively.

X = train_identity.loc[:,train_identity.columns!='isFraud']
y = train_identity['isFraud']

# implementing train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=66)

from sklearn.model_selection import GridSearchCV
param = {'n_estimators': [100, 200, 300, 400, 500], 'max_depth': [2, 3, 4, 7, 9]}
rfc = RandomForestClassifier()
clf_rfc_cv = GridSearchCV(rfc, param, cv=5,scoring='roc_auc', n_jobs=-1)
clf_rfc_cv.fit(X_train,y_train)

print("tuned hpyerparameters :(best parameters) ",clf_rfc_cv.best_params_)
print("accuracy :",clf_rfc_cv.best_score_)

rfc = RandomForestClassifier(n_estimators=400,max_depth=9,bootstrap="False")
rfc.fit(X,y)
y_pred = rfc.predict(test_identity)
# save output to csv file
pd.DataFrame(y_pred).to_csv('myrfoutput2.csv')
pd.DataFrame(rfc_predict).to_csv('myrfoutput1.csv')

rfc_cv_score = cross_val_score(rfc, X, y, cv=5, scoring='roc_auc')

#More scores
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())

