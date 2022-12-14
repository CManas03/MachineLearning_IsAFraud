import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import optuna
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from optuna.integration import LightGBMPruningCallback
import lightgbm as lgb
import re
from sklearn.model_selection import train_test_split

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

train_identity = pd.read_csv('nstrain.csv')
test_identity = pd.read_csv('nstest.csv')

train_identity.drop(axis='columns',labels='had_id',inplace=True)
test_identity.drop(axis='columns',labels='had_id',inplace=True)

# The size of dataset is increasing rapidly. It is become very difficult for traditional data science algorithms to give accurate results. Light GBM is prefixed as Light because of its high speed. Light GBM can handle the large size of data and takes lower memory to run.

# LightGBM grows tree vertically while other tree based learning algorithms grow trees horizontally. It means that LightGBM grows tree leaf-wise while other algorithms grow level-wise. It will choose the leaf with max delta loss to grow. When growing the same leaf, leaf-wise algorithm can reduce more loss than a level-wise algorithm.

# The key difference in speed is because XGBoost split the tree nodes one level at a time and LightGBM does that one node at a time.


### Some control parameters:
    
# max_depth : It describes the maximum depth of tree. This parameter is used to handle model overfitting. If you feel that your model is overfitted, you should to lower max_depth.

# min_data_in_leaf : It is the minimum number of the records a leaf may have. The default value is 20, optimum value. It is also used to deal with overfitting.

# feature_fraction: Used when your boosting is random forest. 0.8 feature fraction means LightGBM will select 80% of parameters randomly in each iteration for building trees.

# bagging_fraction : specifies the fraction of data to be used for each iteration and is generally used to speed up the training and avoid overfitting.

# early_stopping_round : This parameter can help you speed up your analysis. Model will stop training if one metric of one validation data doesn’t improve in last early_stopping_round rounds. This will reduce excessive iterations.

# lambda : lambda specifies regularization. Typical value ranges from 0 to 1.

# min_gain_to_split : This parameter will describe the minimum gain to make a split. It can used to control number of useful splits in tree.

# min_gain_to_split : This parameter will describe the minimum gain to make a split. It can used to control number of useful splits in tree.


#create classifier instance.
clf = lgb.LGBMClassifier()

#Since lightgbm does not support special json characters in feature names, we use a simple code snippet to rename our columns.



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

#First we train the classifier by using only the train data, we use split the train data into "train" and "test" chunks

#Separate the target variable and rest of the variables.
X = train_identity.loc[:,train_identity.columns!='isFraud']
y = train_identity['isFraud']

#splitting train data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 45)

clf_new = lgb.LGBMClassifier(boosting_type='dart',metric='binary_logloss',application='binary',
                            max_depth=12,n_estimators=1500,num_leaves=4096,learning_rate=0.1,
                            feature_fraction=0.8,subsample=0.6,lambda_l1=0.5)


def objective(trial, X, y):
    param_grid = {
        #"device_type": trial.suggest_categorical("device_type", ['gpu']),
        "n_estimators": trial.suggest_categorical("n_estimators", [1500]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100, step=40),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1000, 10000, step=1000),
        "lambda_l1": trial.suggest_int("lambda_l1", 0.5, 2),
        "lambda_l2": trial.suggest_int("lambda_l2", 0.5, 2),
#         "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
#         "bagging_fraction": trial.suggest_float(
#             "bagging_fraction", 0.2, 0.95, step=0.4
#         ),
#         "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
#         "feature_fraction": trial.suggest_float(
#             "feature_fraction", 0.2, 0.95, step=0.4
#         ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1121218)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgb.LGBMClassifier(objective="binary", **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="binary_logloss",
            early_stopping_rounds=100,
            callbacks=[
                LightGBMPruningCallback(trial, "binary_logloss")
            ],  # Add a pruning callback
        )
        preds = model.predict_proba(X_test)
        cv_scores[idx] = log_loss(y_test, preds)

    return np.mean(cv_scores)

import warnings
warnings.filterwarnings('ignore')

study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")
func = lambda trial: objective(trial, X, y)
study.optimize(func, n_trials=5)

print(f"\tBest value (rmse): {study.best_value:.5f}")
print(f"\tBest params:")

for key, value in study.best_params.items():
    print(f"\t\t{key}: {value}")
    
    '''
    Best value (rmse): 0.05312
	Best params:
		n_estimators: 1500
		learning_rate: 0.09525516304951938
		num_leaves: 60
		max_depth: 11
		min_data_in_leaf: 1000
		lambda_l1: 1
		lambda_l2: 1
    '''
