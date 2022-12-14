import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train_identity = pd.read_csv('nstrain.csv')
train_identity.drop(axis='columns',labels='had_id',inplace=True)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn import metrics


#The hypothesis for perceptron is the "unit step" function applied over theta_x, 
#where theta_x is the array of weights that we assign

def unit_func(z):
    return 1.0 if (z>0) else 0.0 
#We take the dot product of the input features and the parameters theta and then apply the unit function
#to get prediction. If the prediction is wrong or if we find a misclassified point then we update the 
#parameters theta.

#Let's now define a function to instantiate our perceptron model.
def perceptron(x,y,alpha,epochs):
    
    m,n = x.shape
    # m and n give us number of training points and features respectively.
    
    #array to store params
    theta = np.zeros((n+1,1))
    
    #array to store misclassified points
    misclassifications = []
    
    #training loop
    for epoch in range(epochs):
        
        #store how many misses.
        misses = 0
        
        for xid,xele in enumerate(x):
            
            #Insering additional 1 for bias.
            xele = np.insert(xele, 0, 1).reshape(-1,1)
            
            #Calculating y_hat/y_pred
            y_hat = unit_func(np.dot(xele.T, theta))
            
            # Updating if the example is misclassified.
            if (np.squeeze(y_hat) - y[xid]) != 0:
                theta += alpha*((y[xid] - y_hat)*xele)
                
                # Incrementing misses 1.
                misses += 1
        
        # Appending number of misclassified examples
        # at every iteration.
        misclassifications.append(misses)
        
    return theta, misclassifications

#Now that we have theta, theta.x = 0 will give us our decision boundary.

#The limitations are : 
#we have lots of features and we dont know if our data is linearly seperable or not. 
#perceptron can take varying amounts of time and computation.

#pocket algorithm.
#def pocket_perceptron

# In order to look at the performance of our model, we use k-fold cross validation. However, it can result in misleading results and potentially fail when used on classification problems with a severe class imbalance (just like ours xD). Instead, the technique must be modified to stratify the sampling by the class label, called stratified k-fold cross-validation.

# Specifically, we can split a dataset randomly, although in such a way that maintains the same class distribution in each subset. This is called stratification or stratified sampling and the target variable (y), the class, is used to control the sampling process.

X = train_identity.loc[:,train_identity.columns!='isFraud']
y = train_identity['isFraud']

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
# enumerate the splits and summarize the distributions
for train_ix, test_ix in kfold.split(X, y):
	# select rows
	train_X, test_X = X.iloc[train_ix], X.iloc[test_ix]
	train_y, test_y = y.iloc[train_ix], y.iloc[test_ix]
	# summarize train and test composition
	train_0, train_1 = len(train_y[train_y==0]), len(train_y[train_y==1])
	test_0, test_1 = len(test_y[test_y==0]), len(test_y[test_y==1])
	print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))

# F1 Score
# In order to calculate the F1 score, we need to calculate two metrics which are precision and recall.

# Precision (P) = TP/(TP + FP)\ Recall (R) = TP/(TP + FN)

# TP = True positives - If model predicts Positive class correctly then its True Positive.\ FP = False positives - If model predicts Positive class incorrectly then its False Positive.\ FN = False negatives - If model predicts Negative class incorrectly then its False Negative.

# F1 = 2PR/(P + R)\ Upon simplifying the formula for F1, we get:\ F1 = TP/(TP+(FP+FN)/2)\ If we consider FP+FN as net falses (NF) we can write it as:\ F1 = TP/(TP + (NF/2))

# Using the above formula, we can directly calculate the F1 score for each label without having to calculate the precision and recall.\ Once we have the F1 scores for each label, we can calculate the macro average and the weighted average i,e. the overall F1 score.

X = X.to_numpy()
y = y.to_numpy()

theta,missclassfics = perceptron(X,y,0.1,10)

m,n = X.shape

X = np.c_[np.ones(m),X]

count=0
y_pred=np.zeros((m,1))
for element in (np.dot(X,theta)):
    if (element>0):
        y_pred[count]=1
    else:
        y_pred[count]=0
        
train_X = train_X.to_numpy()
train_y = train_y.to_numpy()

theta,missclassfics = perceptron(train_X,train_y,0.1,10)

theta.shape
m,n = train_X.shape
train_X = np.c_[np.ones(m),train_X] 
count=0
y_pred_train=np.zeros((m,1))
for element in (np.dot(train_X,theta)):
    if (element>0):
        y_pred_train[count]=1
    else:
        y_pred_train[count]=0
        
def get_y_pred(m,X,theta):
    
    count=0
    y_pred=np.zeros((m,1))
    for element in (np.dot(X,theta)):
        if (element>0):
            y_pred[count]=1
        else:
            y_pred[count]=0
            
    return y_pred

X = train_identity.loc[:,train_identity.columns!='isFraud']
y = train_identity['isFraud']

count=0
for train_ix, test_ix in kfold.split(X, y):
    
    count=count+1
    print("Iteration number : ",count)
    # select rows
    train_X, test_X = X.iloc[train_ix], X.iloc[test_ix]
    train_y, test_y = y.iloc[train_ix], y.iloc[test_ix]
    
    
    m,n = train_X.shape
    m_test,n_test = test_X.shape
    
    train_X = train_X.to_numpy() 
    train_y = train_y.to_numpy()
    
    test_X = test_X.to_numpy()
    test_y = test_y.to_numpy()
    
    #training the model.
    theta,missclassfics = perceptron(train_X,train_y,0.1,10)
    train_X = np.c_[np.ones(m),train_X]
    y_pred_train = get_y_pred(m,train_X,theta)
    train_f1 = f1_score(train_y, y_pred_train, average='weighted')
    
    #testing the model.
    theta,missclassfics = perceptron(test_X,test_y,0.1,10)
    test_X = np.c_[np.ones(m_test),test_X] 
    y_pred_test = get_y_pred(m_test,test_X,theta)
    test_f1 = f1_score(test_y, y_pred_test, average='weighted')
    
    print("Training score : ",train_f1)
    print("Testing score : ",test_f1)
    
# Support Vector Machines
# Now using the weights obtained by the perceptron as the initial weights for SVM.

X = train_identity.loc[:,train_identity.columns!='isFraud']
y = train_identity['isFraud']
weights = {0:1.0, 1:100.0}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
# enumerate the splits and summarize the distributions
for train_ix, test_ix in kfold.split(X, y):
    # select rows
    train_X, test_X = X.iloc[train_ix], X.iloc[test_ix]
    train_y, test_y = y.iloc[train_ix], y.iloc[test_ix]
    
    svclassifier = SVC(kernel='rbf',gamma='scale',class_weight=weights)
    svclassifier.fit(train_X, train_y)
    
    y_pred = svclassifier.predict(test_X)
    print("Accuracy:",metrics.accuracy_score(test_y, y_pred))
