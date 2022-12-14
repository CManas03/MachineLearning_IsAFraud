import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import pickle
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import StratifiedKFold



# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


train_identity = pd.read_csv('nstrain.csv')
test_identity = pd.read_csv('nstest.csv')
train_identity.drop(axis='columns',labels='had_id',inplace=True)
test_identity.drop(axis='columns',labels='had_id',inplace=True)
#Separate the target variable and rest of the variables.
X = train_identity.loc[:,train_identity.columns!='isFraud']
y = train_identity['isFraud']
data_dmatrix = xgb.DMatrix(data=X,label=y)



# Boosting is a sequential technique which works on the principle of an ensemble. It combines a set of weak learners and delivers improved prediction accuracy.

# default base learners of XGBoost: tree ensembles. The tree ensemble model is a set of classification and regression trees (CART). Trees are grown one after another ,and attempts to reduce the misclassification rate are made in subsequent iterations.


xgb_cl = xgb.XGBClassifier(
            n_estimators = 800,
            max_depth = 8,
            learning_rate = 0.02,
            subsample = 0.8,
            colsample_bytree = 0.8,
            missing = -1,
            random_state = 0,
            tree_method='gpu_hist',
            gpu_id=-1)


'''xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.6, eval_metric='auc',
              gamma=1, gpu_id=-1, importance_type='gain',
              interaction_constraints='', learning_rate=0.1, max_delta_step=0,
              max_depth=9, min_child_weight=100,
              monotone_constraints='()', n_estimators=150, n_jobs=8,
              num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
              scale_pos_weight=1, subsample=1.0, tree_method='exact',
              use_label_encoder=False, validate_parameters=1, verbosity=None)'''
              
              
xgb_cl.fit(X,y)
preds = xgb_cl.predict(test_identity)

pd.DataFrame(preds).to_csv('myoutputxgboost41.csv')

####Attempt to do GridSearchCV####

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=145)

# xgb_cl = xgb.XGBClassifier()

'''
brute force scan for all parameters, here are the tricks
usually max_depth is 6,7,8
learning rate is around 0.05, but small changes may make big diff
tuning min_child_weight subsample colsample_bytree can have 
much fun of fighting against overfit 
n_estimators is how many round of boosting
finally, ensemble xgboost with multiple seeds may reduce variance
'''

# parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
#               'objective':['binary:logistic'],
#               'learning_rate': [0.05], #so called `eta` value
#               'max_depth': [6],
#               'min_child_weight': [11],
#               'silent': [1],
#               'subsample': [0.8],
#               'colsample_bytree': [0.7],
#               'n_estimators': [5], #number of trees, change it to 1000 for better results
#               'missing':[-999],
#               'seed': [1337]}

# from sklearn.grid_search import GridSearchCV
# from sklearn.model_selection import StratifiedKFold

# clf = GridSearchCV(xgb_model, parameters, n_jobs=5, 
#                    cv=StratifiedKFold(train_identity['isFraud'], n_folds=5, shuffle=True), 
#                    scoring='roc_auc',
#                    verbose=2, refit=True)

# from sklearn.metrics import accuracy_score
# xgb_cl.fit(X_train,y_train)
# preds = xgb_cl.predict(X_test)
# print(accuracy_score(y_test, preds))
