import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re

train_identity = pd.read_csv('mycsvfile_te.csv')
test_identity = pd.read_csv('mycsvfile_tr.csv')

train_identity.drop(axis='columns',labels='had_id',inplace=True)
test_identity.drop(axis='columns',labels='had_id',inplace=True)

'''LightGBM grows tree vertically while other tree based learning algorithms grow trees horizontally. 
It means that LightGBM grows tree leaf-wise while other algorithms grow level-wise. It will choose 
the leaf with max delta loss to grow. When growing the same leaf, leaf-wise algorithm can reduce more 
loss than a level-wise algorithm.'''

'''The key difference in speed is because XGBoost split the tree nodes one level at a time and LightGBM does that one node at a time.'''

#importing lgbm
import lightgbm as lgb

#create classifier instance.
clf = lgb.LGBMClassifier()


# Change columns names ([LightGBM] Do not support special JSON characters in feature name.)
new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in train_identity.columns}
new_n_list = list(new_names.values())
# [LightGBM] Feature appears more than one time.
new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col for i, (col, new_col) in enumerate(new_names.items())}
train_identity = train_identity.rename(columns=new_names)

# same thing for test
new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in test_identity.columns}
new_n_list = list(new_names.values())
# [LightGBM] Feature appears more than one time.
new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col for i, (col, new_col) in enumerate(new_names.items())}
test_identity = test_identity.rename(columns=new_names)

#Separate the target variable and rest of the variables.
X = train_identity.loc[:,train_identity.columns!='isFraud']
y = train_identity['isFraud']

#splitting train data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 45)

# first we run it on the default parameters of lgbm classifer, later we will do hyperparameter tuning.
clf.fit(X_train,y_train)

# predict the results
y_pred=clf.predict(X_test)

# view accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred, y_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

# comparing train-set and test-set accuracy to check for overfitting.
y_pred_train = clf.predict(X_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

max_depth_list = [12,20,20,20,21]
n_estimators_list = [1000,1200,1200,1500,1500]
num_leaves_list = [31,31,40,45,50]
subsample_list = [0.2,0.2,0.4,0.6,0.7] 

preds_list_test = []
preds_list_train = []

from sklearn.metrics import accuracy_score

# testing for all those array values.
for i in range(5):
    clf_new = lgb.LGBMClassifier(boosting_type='dart',metric='binary_logloss',application='binary',
                                 max_depth=max_depth_list[i],n_estimators=n_estimators_list[i],num_leaves=num_leaves_list[i],learning_rate=0.1,
                                 feature_fraction=0.8,subsample=subsample_list[i])
    
    clf_new.fit(X_train,y_train)
    y_pred=clf_new.predict(X_test)
    
    accuracy=accuracy_score(y_pred, y_test)
    ac_test = accuracy_score(y_test, y_pred)
    print('iteration : ',i+1)
    print('LightGBM Model accuracy score: {0:0.4f}'.format(ac_test))
    preds_list_test.append(ac_test)
    y_pred_train = clf_new.predict(X_train)
    ac_train = accuracy_score(y_train, y_pred_train)
    print('Training-set accuracy score: {0:0.4f}'. format(ac_train))
    preds_list_train.append(ac_train)
    
    clf_new.fit(X,y)
    y_pred=clf_new.predict(test_identity)
    savestring = 'mylgbroutput'+str(i+4)+'.csv'
    pd.DataFrame(y_pred).to_csv(savestring)

# first we run it on the default parameters of lgbm classifer, later we will do hyperparameter tuning.
clf_new.fit(X_train,y_train)

# predict the results
y_pred=clf_new.predict(X_test)

# view accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred, y_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

# comparing train-set and test-set accuracy to check for overfitting.
y_pred_train = clf_new.predict(X_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

# This is better accuracy, now lets train the classifier with the whole of train data and make a prediction for the whole of test data.
clf_new = lgb.LGBMClassifier(boosting_type='dart',metric='binary_logloss',application='binary',
                             max_depth=20,n_estimators=1500,num_leaves=45,learning_rate=0.1,
                             feature_fraction=0.8,subsample=0.6,lambda_l1=0.5)

# fit model on whole train data
clf_new.fit(X,y)

# predict the results on test data
y_pred=clf_new.predict_proba(test_identity)

# save output to csv file
pd.DataFrame(y_pred).to_csv('mylgbmoutput12.csv')