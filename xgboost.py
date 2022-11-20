import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

train_identity = pd.read_csv('mycsvfile3_new.csv')
test_identity = pd.read_csv('mycsvfile2_new.csv')
train_identity.shape
train_identity["isFraud"].plot(kind='hist',edgecolor='black')

train_identity.drop(axis='columns',labels='had_id',inplace=True)
test_identity.drop(axis='columns',labels='had_id',inplace=True)

data_dmatrix = xgb.DMatrix(data=X,label=y)
train_identity.head()

#Separate the target variable and rest of the variables.
X = train_identity.loc[:,train_identity.columns!='isFraud']
y = train_identity['isFraud']

#Now we will convert the dataset into an optimized data structure called Dmatrix that XGBoost supports and gives it acclaimed performance and efficiency gains.
data_dmatrix = xgb.DMatrix(data=X,label=y)

xgb_cl = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.6, eval_metric='auc',
              gamma=1, gpu_id=-1, importance_type='gain',
              interaction_constraints='', learning_rate=0.02, max_delta_step=0,
              max_depth=12, min_child_weight=100,
              monotone_constraints='()', n_estimators=900, n_jobs=8,
              num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
              scale_pos_weight=1, subsample=1.0, tree_method='gpu_hist',
              use_label_encoder=False, validate_parameters=1, verbosity=None)

xgb_cl.fit(X,y)
preds = xgb_cl.predict(test_identity)
#print(accuracy_score(y_test, preds))

pd.DataFrame(preds).to_csv('myoutputxgboost9.csv')

train_pred = xgb_cl.predict(X)
print(accuracy_score(y, train_pred))
