import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt 
import seaborn as sns 
import sklearn.preprocessing as preprocessing 

train_identity = pd.read_csv('train.csv')
test_identity = pd.read_csv('test.csv')

train_identity.head()

# check null values
train_identity.isna().sum()

# lets look at the distribution and metrics of the numerical values
train_identity.describe()

train_identity["dist2"].unique().shape
train_identity["dist1"].unique().shape

train_identity["dist2"].describe()

# remove less rows nulls
for i in train_identity.columns:
    if (train_identity[i].isna().sum()<10000):
        train_identity.drop(axis="rows", labels=train_identity.index[train_identity[i].isna()], inplace=True)

train_identity["addr1"].fillna(train_identity["addr1"].mean(),inplace=True)
train_identity["addr2"].fillna(train_identity["addr2"].mean(),inplace=True)

test_identity["addr1"].fillna(test_identity["addr1"].mean(),inplace=True)
test_identity["addr2"].fillna(test_identity["addr2"].mean(),inplace=True)

'''for columns dist1 and dist2, given in the dataset distribution - distances between (not limited) billing address, mailing address, zip code, IP address, phone area, etc.” We can see that dist2 has lot more null values than dist1. So out of both the distances it's better to remove the "dist2" column since it has 90%+ null values.
'''
train_identity.drop(axis="columns",labels="dist2",inplace=True)
test_identity.drop(axis="columns",labels="dist2",inplace=True)

# D1-D15: timedelta, such as days between previous transaction, etc. D7 has 90%+ null values out of these so we can drop it.
train_identity["D7"].describe()

sns.boxplot(x=train_identity["D10"])
sns.boxplot(x=train_identity["D15"])

test_df = train_identity[["D1","D2","D3","D4","D5","D6","D7","D8","D9","D10","D11","D12","D13","D14","D15"]]

test_df.corr()
plt.figure(figsize=(15,15))
sns.heatmap(test_df.corr(), vmin=-1, cmap="coolwarm", annot=True)

# replacing with median for D10 and D15.
train_identity["D10"].fillna(train_identity["D10"].median(),inplace=True)
train_identity["D15"].fillna(train_identity["D15"].median(),inplace=True)

# replacing with median for D10 and D15.
test_identity["D10"].fillna(test_identity["D10"].median(),inplace=True)
test_identity["D15"].fillna(test_identity["D15"].median(),inplace=True)

train_identity.drop(axis="columns",labels="D2",inplace=True)
test_identity.drop(axis="columns",labels="D2",inplace=True)

train_identity.drop(axis="columns",labels=["D6","D12","D7"],inplace=True)
test_identity.drop(axis="columns",labels=["D6","D12","D7"],inplace=True)

train_identity.drop(axis="columns",labels=["D13","D14"],inplace=True)
test_identity.drop(axis="columns",labels=["D13","D14"],inplace=True)

train_identity.drop(axis="columns",labels=["D8","D9"],inplace=True)
test_identity.drop(axis="columns",labels=["D8","D9"],inplace=True)

test_df = train_identity[["D1","D3","D4","D5","D10","D11","D15"]]
test_df.corr()
plt.figure(figsize=(7,7))
sns.heatmap(test_df.corr(), vmin=-1, cmap="coolwarm", annot=True)

'''This is how our correlation matrix looks like now, We can also analyse further by looking at the correlation between each of these columns with "isFraud" and see if we can drop any more columns.'''
test_df = train_identity[["D1","D3","D4","D5","D10","D11","D15","isFraud"]]
test_df.corr()
plt.figure(figsize=(8,8))
sns.heatmap(test_df.corr(), vmin=-1, cmap="coolwarm", annot=True)

train_identity.drop(axis="columns",labels=["D3","D11"],inplace=True)
test_identity.drop(axis="columns",labels=["D3","D11"],inplace=True)

train_identity.drop(axis="columns",labels="D5",inplace=True)
test_identity.drop(axis="columns",labels="D5",inplace=True)

# REGRESSION IMPUTATION
'''When we have multiple variables with missing values, we can't just directly use Regression Imputation
 to impute one of them as the predictors contain missing data themselves. But then, how can we impute one variable without imputing another?'''

'''We can avoid this Catch-22 situation by initially imputing all the variables with missing values using some trivial methods
like Simple Random Imputation (we impute the missing data with random observed values of the variable) which is later followed by 
Regression Imputation of each of the variables iteratively.'''

def random_imputation(df, feature):

    number_missing = df[feature].isnull().sum()
    observed_values = df.loc[df[feature].notnull(), feature]
    df.loc[df[feature].isnull(), feature + '_imp'] = np.random.choice(observed_values, number_missing, replace = True)
    
    return df

from sklearn import linear_model

train_identity_cat=train_identity.select_dtypes(include=['object'])
test_identity_cat=test_identity.select_dtypes(include=['object'])

# temporarily I am replacing the null values with mode for D4.
train_identity["D4"].fillna(train_identity["D4"].mode(),inplace=True)
test_identity["D4"].fillna(test_identity["D4"].mode(),inplace=True)

# Now we look at the columns M1-M9. These columns tell us about matches, such as names on card and address, etc.
train_identity[["M1","M2","M3","M4","M5","M6","M7","M8","M9"]].isna().sum()

match_df = train_identity[["M1","M2","M3","M4","M5","M6","M7","M8","M9","isFraud"]]
column_names_to_one_hot = ["M1", "M2","M3","M4","M5","M6","M7","M8","M9"]
match_df = pd.get_dummies(match_df, columns=column_names_to_one_hot)

match_df.corr()
plt.figure(figsize=(20,20))
sns.heatmap(match_df.corr(), vmin=-1, cmap="coolwarm", annot=True)
plt.show()

#since correlation is less than 0.9, we are keeping all the columns intact.

'''Lets look at the features of type Vxxx, other than the ones which have no null values. Now, we can
 put all the remaining ones into different classes based on common number of null values.'''

# V1-V11 have the same number of null values - 205480, lets anaylse these.
Vxxx_df = train_identity[["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","isFraud"]]
plt.figure(figsize=(20,20))
sns.heatmap(Vxxx_df.corr(), vmin=-1, cmap="coolwarm", annot=True)
plt.show()

train_identity.drop(axis="columns",labels=["V4","V11"],inplace=True)
test_identity.drop(axis="columns",labels=["V4","V11"],inplace=True)

# V12-V34 have the same number of null values - 56226
temp_array =[]
tempstr = "V"
for i in range(12,35):
    temp_array.append(tempstr+str(i))
temp_array.append("isFraud")
Vxxx_df = train_identity[temp_array]
plt.figure(figsize=(30,30))
sns.heatmap(Vxxx_df.corr(), vmin=-1, cmap="coolwarm", annot=True)
plt.show()

train_identity.drop(axis="columns",labels=["V13","V15","V16","V34","V31","V32","V18","V21","V22","V28","V30"],inplace=True)
test_identity.drop(axis="columns",labels=["V13","V15","V16","V34","V31","V32","V18","V21","V22","V28","V30"],inplace=True)

for i in ["V12","V14","V17","V19","V20","V23","V24","V25","V26","V27","V29","V33"]:
    train_identity[i].fillna(train_identity[i].median(),inplace=True)
    test_identity[i].fillna(test_identity[i].median(),inplace=True)

# Moving on to the next group, V35-V52 - 124356 null values.
temp_array =[]
tempstr = "V"
for i in range(35,53):
    temp_array.append(tempstr+str(i))
temp_array.append("isFraud")
Vxxx_df = train_identity[temp_array]
plt.figure(figsize=(30,30))
sns.heatmap(Vxxx_df.corr(), vmin=-1, cmap="coolwarm", annot=True)
plt.show()

# we can drop the columns - V36,V37,V39,V42,V44,V49,V50,V51
train_identity.drop(axis="columns",labels=["V36","V37","V39","V42","V44","V44","V49","V50","V51"],inplace=True)
test_identity.drop(axis="columns",labels=["V36","V37","V39","V42","V44","V44","V49","V50","V51"],inplace=True)

for i in ["V35","V38","V40","V41","V43","V45","V46","V47","V48","V52"]:
    train_identity[i].fillna(train_identity[i].median(),inplace=True)
    test_identity[i].fillna(test_identity[i].median(),inplace=True)

# The next group is V53-V74 with 56835 null values.
temp_array =[]
tempstr = "V"
for i in range(53,75):
    temp_array.append(tempstr+str(i))
temp_array.append("isFraud")
Vxxx_df = train_identity[temp_array]
plt.figure(figsize=(30,30))
sns.heatmap(Vxxx_df.corr(), vmin=-1, cmap="coolwarm", annot=True)
plt.show()

# we can drop the columns - V54,V57,V58,V60,V63,V64,V70,V71,V72,V73
train_identity.drop(axis="columns",labels=["V54","V57","V58","V71","V72","V73","V60","V63","V64","V70"],inplace=True)
test_identity.drop(axis="columns",labels=["V54","V57","V58","V71","V72","V73","V60","V63","V64","V70"],inplace=True)

temp_array.remove("isFraud")
labels=["V54","V57","V58","V71","V72","V73","V60","V63","V64","V70"]
for i in temp_array:
    if i not in labels:
        train_identity[i].fillna(train_identity[i].median(),inplace=True)
        test_identity[i].fillna(test_identity[i].median(),inplace=True)

# Next we have V75-V94 with 65701 null values.
temp_array =[]
tempstr = "V"
for i in range(75,95):
    temp_array.append(tempstr+str(i))
temp_array.append("isFraud")
Vxxx_df = train_identity[temp_array]
plt.figure(figsize=(30,30))
sns.heatmap(Vxxx_df.corr(), vmin=-1, cmap="coolwarm", annot=True)
plt.show()

# we can drop the columns - V76,V79,V81,V82,V85,V91,V92,V93,V94
train_identity.drop(axis="columns",labels=["V76","V79","V81","V82","V85","V91","V92","V94","V93"],inplace=True)
test_identity.drop(axis="columns",labels=["V76","V79","V81","V82","V85","V91","V92","V94","V93"],inplace=True)

temp_array.remove("isFraud")
labels=["V76","V79","V81","V82","V85","V91","V92","V94","V93"]
for i in temp_array:
    if i not in labels:
        train_identity[i].fillna(train_identity[i].median(),inplace=True)
        test_identity[i].fillna(train_identity[i].median(),inplace=True)

temp_array.remove("isFraud")
labels=["V76","V79","V81","V82","V85","V91","V92","V94","V93"]
for i in temp_array:
    if i not in labels:
        train_identity[i].fillna(train_identity[i].median(),inplace=True)
        test_identity[i].fillna(train_identity[i].median(),inplace=True)

# Within these columns, we drop the ones which have greater than 75% null values.
thresh = 433584*0.75
for i in train_identity.columns[97:342]:
    # we are iterating through all the remaining columns of the type Vxxx.
    if(train_identity[i].isna().sum()>thresh):
        train_identity.drop(axis="columns",labels=i,inplace=True)
        test_identity.drop(axis="columns",labels=i,inplace=True)

# Now we look at some of the categorical columns.
train_identity["DeviceType"].value_counts(dropna=False).plot(kind='barh')

mobile_fraud=[]
desktop_fraud=[]
for i in train_identity.index:
    if (train_identity["DeviceType"][i]=='mobile' and train_identity["isFraud"][i]==1):
        mobile_fraud.append(i)
    elif(train_identity["DeviceType"][i]=='desktop' and train_identity["isFraud"][i]==1):
        desktop_fraud.append(i)
langs = ['mobile_fraud','desktop_fraud']
students = [len(mobile_fraud),len(desktop_fraud)]
fig=plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(langs,students)
plt.show()

# Percentage of fraud transactions done through mobile vs desktop would give us a better idea
students = [(len(mobile_fraud)/40939)*100,(len(desktop_fraud)/63158)*100]
fig=plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(langs,students)
plt.show()

train_identity["DeviceType"].fillna('unknown',inplace=True)
test_identity["DeviceType"].fillna('unknown',inplace=True)

# in deviceInfo - Since this is a lot of unique values and there are also many null values in this column, we can group most of these columns and make a new column for the NULL values.
def setDevice(df):
    df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()
    
    df['device_name'] = df['DeviceInfo'].str.split('/', expand=True)[0]

    df.loc[df['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
    df.loc[df['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
    df.loc[df['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
    df.loc[df['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
    df.loc[df['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
    df.loc[df['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
    df.loc[df['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
    df.loc[df['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

    df.loc[df.device_name.isin(df.device_name.value_counts()[df.device_name.value_counts() < 200].index), 'device_name'] = "Others"
    df['had_id'] = 1
    
    return df
train_identity = setDevice(train_identity)
test_identity = setDevice(test_identity)

train_identity["device_name"].isna().sum()
train_identity.drop(axis="columns",labels="DeviceInfo",inplace=True)
test_identity.drop(axis="columns",labels="DeviceInfo",inplace=True)

# Lets take all the id columns seperately
temp_array =[]
tempstr = "id"
for i in range(1,39):
    if(i<10):
        temp_array.append(tempstr+"_"+str(0)+str(i))
    else:
        temp_array.append(tempstr+"_"+str(i))
#temp_array.append("isFraud")
id_df = train_identity[temp_array]

# we can drop the columns with more than 90% null values.
percent_missing = id_df.isnull().sum() * 100 / len(id_df)
nullpercent_df = pd.DataFrame({'column_name': id_df.columns,
                                 'percent_missing': percent_missing})
nullpercent_df.sort_values('percent_missing', inplace=True)

for i in range(0,len(nullpercent_df["column_name"])):
    if (nullpercent_df["percent_missing"][i]>90):
        print(nullpercent_df["column_name"][i])
        train_identity.drop(axis="columns",labels=nullpercent_df["column_name"][i],inplace=True)
        test_identity.drop(axis="columns",labels=nullpercent_df["column_name"][i],inplace=True)

train_identity["id_12"].fillna('Unknown_status',inplace=True)
test_identity["id_12"].fillna('Unknown_status',inplace=True)

train_identity["id_15"].fillna('NotFound',inplace=True)
test_identity["id_15"].fillna('NotFound',inplace=True)

train_identity["id_16"].fillna('Unknown_status',inplace=True)
test_identity["id_16"].fillna('Unknown_status',inplace=True)

train_identity["id_28"].fillna('Unknown_status',inplace=True)
test_identity["id_28"].fillna('Unknown_status',inplace=True)

train_identity["id_29"].fillna('Unknown_status',inplace=True)
test_identity["id_29"].fillna('Unknown_status',inplace=True)

train_identity["id_30"].fillna('Unknown_status',inplace=True)
test_identity["id_30"].fillna('Unknown_status',inplace=True)

train_identity["id_31"].fillna('Unknown_status',inplace=True)
test_identity["id_31"].fillna('Unknown_status',inplace=True)

train_identity["id_33"].fillna('Unknown_status',inplace=True)
test_identity["id_33"].fillna('Unknown_status',inplace=True)

# test_identity does not have a match status of -1 so we can drop from the train one later as it has only 3 such instances.
train_identity["id_34"].fillna('Unknown_status',inplace=True)
test_identity["id_34"].fillna('Unknown_status',inplace=True)

train_identity["id_35"].fillna('Unknown_status',inplace=True)
test_identity["id_35"].fillna('Unknown_status',inplace=True)

train_identity["id_36"].fillna('Unknown_status',inplace=True)
test_identity["id_36"].fillna('Unknown_status',inplace=True)

train_identity["id_37"].fillna('Unknown_status',inplace=True)
test_identity["id_37"].fillna('Unknown_status',inplace=True)

train_identity["id_38"].fillna('Unknown_status',inplace=True)
test_identity["id_38"].fillna('Unknown_status',inplace=True)

# histograms for skew.
plt.hist(train_identity["id_01"],bins=100)
plt.show()

plt.hist(train_identity["id_02"],bins=100)
plt.show()

plt.hist(train_identity["id_03"],bins=50)
plt.show()

temp_array = ["id_01","id_02","id_03","id_04","id_05","id_06","id_09","id_10","id_11","id_12","id_13","id_14","id_15","id_16","id_17","id_19","id_20","id_28","id_29","id_30","id_31","id_32","id_33","id_34","id_35","id_36","id_37","id_38"]
temp_array.append("isFraud")
id_df = train_identity[temp_array]
id_df.corr()
plt.figure(figsize=(20,20))
sns.heatmap(id_df.corr(), vmin=-1, cmap="coolwarm", annot=True)

train_identity.drop(axis='columns',labels=["id_05","id_11"],inplace=True)
test_identity.drop(axis='columns',labels=["id_05","id_11"],inplace=True)

train_identity['id_02' + '_imp'] = train_identity['id_02']
train_identity = random_imputation(train_identity, 'id_02')
test_identity['id_02' + '_imp'] = test_identity['id_02']
test_identity = random_imputation(test_identity, 'id_02')

anal_columns = ["D4","dist1","V1","V2","V3","V5","V6","V7","V8","V9","V10","id_01","id_02","id_03","id_04","id_06","id_09","id_10","id_13","id_14","id_17","id_19","id_20","id_32"]

for feature in anal_columns:
    train_identity[feature + '_imp'] = train_identity[feature]
    test_identity[feature + '_imp'] = test_identity[feature]
    train_identity = random_imputation(train_identity, feature)
    test_identity = random_imputation(test_identity,feature)


# REGRESSION IMPUTATION IMPLEMENTATION
deter_data = pd.DataFrame(columns = ["Det" + name for name in anal_columns])
deter_data1 = pd.DataFrame(columns = ["Det" + name for name in anal_columns])
        
for feature in anal_columns:
    deter_data["Det" + feature] = train_identity[feature + "_imp"]
    deter_data1["Det" + feature] = test_identity[feature + "_imp"]
    parameters = list(set(train_identity.columns) - set(anal_columns) - {feature + '_imp'}-set(train_identity_cat.columns))
    parameters.remove('device_name')
    
    for i in train_identity.columns[train_identity.isna().any()]:
        if i in (train_identity_cat.columns):
            continue
        elif i in set(anal_columns):
            continue
        else:
            parameters.remove(i)
            
    parameters1 = list(set(test_identity.columns) - set(anal_columns) - {feature + '_imp'}-set(train_identity_cat.columns))
    parameters1.remove('device_name')
    
    for i in test_identity.columns[test_identity.isna().any()]:
        if i in (train_identity_cat.columns):
            continue
        elif i in set(anal_columns):
            continue
        else:
            parameters1.remove(i)
            
    
    #Create a Linear Regression model to estimate the missing data
    model = linear_model.LinearRegression()
    model.fit(X = train_identity[parameters], y = train_identity[feature + '_imp'])

    #observe that I preserve the index of the missing data from the original dataframe
    deter_data.loc[train_identity[feature].isnull(), "Det" + feature] = model.predict(train_identity[parameters])[train_identity[feature].isnull()]
    parameters.remove('isFraud')
    model.fit(X = test_identity[parameters1], y = test_identity[feature + '_imp'])
    deter_data1.loc[test_identity[feature].isnull(), "Det" + feature] = model.predict(test_identity[parameters1])[test_identity[feature].isnull()]

mno.matrix(deter_data)
mno.matrix(deter_data1)

# With this we have successfully implemented regression imputation to fill all the remaining continous columns which had moderate amount of null values.
#now we fill the new data in those columns and remove the "_imp" ones.
for feature in anal_columns:
    train_identity[feature] = train_identity[feature + '_imp']
    test_identity[feature] = test_identity[feature + '_imp']
    train_identity.drop(axis="columns",labels=feature+'_imp',inplace=True)
    test_identity.drop(axis="columns",labels=feature+'_imp',inplace=True)

# Let's look at the categorical columns which have null values in them. Consider P_emaildomain and R_emaildomain which give us info about purchaser and recipient email domain respectively.
train_identity["P_emaildomain"].value_counts()

# We can fill the null values with a none/unknown which denotes that email was not required/provided during the transaction.
train_identity["P_emaildomain"].fillna("none",inplace=True)
test_identity["P_emaildomain"].fillna("none",inplace=True)
train_identity["R_emaildomain"].fillna("none",inplace=True)
test_identity["R_emaildomain"].fillna("none",inplace=True)

# M columns
# we can replace the null values with unknown.
for i in range(1,10):
    train_identity["M"+str(i)].fillna("unknown",inplace=True)
    test_identity["M"+str(i)].fillna("unknown",inplace=True)

# some more histograms
plt.hist(train_identity["TransactionAmt"],bins=100)
plt.show()

testskew = np.log(train_identity["TransactionAmt"])
plt.hist(testskew,bins=100)
plt.show()

# Now we do encoding (label or one-hot) depending on the number of unique values in the categorical column.
train_identity_cat=train_identity.select_dtypes(include=['object'])
from sklearn.preprocessing import LabelEncoder

for i in train_identity_cat.columns:
    if(train_identity[i].nunique() > 5):
        label_encoder = LabelEncoder()
        train_identity[i] = label_encoder.fit_transform(train_identity[i])
        test_identity[i] = label_encoder.fit_transform(test_identity[i])
    else:
        train_identity = pd.get_dummies(train_identity, columns = [i])
        test_identity = pd.get_dummies(test_identity, columns = [i])

# more skew
test_df = train_identity[["card1","card2","card3","card5"]]
plt.hist = test_df.hist(bins=50)
plt.tight_layout(pad=0.7, w_pad=0.5, h_pad=1.0)
plt.show()

temp_arr = []
for i in range(1,15):
    temp_arr.append("C"+str(i))
    
test_df = train_identity[temp_arr]

plt.hist = test_df.hist(bins=100)
plt.tight_layout(pad=0.7, w_pad=0.5, h_pad=1.0,)
plt.show()

# SAMPLING
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

train_identity["isFraud"].plot(kind='hist',edgecolor='black')

# simple imputer for the above mentioned columns.
from sklearn.impute import SimpleImputer

for column in newlist:
    imputer = SimpleImputer(missing_values = np.NaN, strategy = 'mean')
    test_identity[column] = imputer.fit_transform(test_identity[column].values.reshape(-1,1))[:,0] 

# we need all the columns other than isFraud in X and isFraud data in Y.
Y = train_identity["isFraud"]
X = train_identity.loc[:,train_identity.columns!="isFraud"]
rus = RandomUnderSampler(random_state=0,sampling_strategy=0.05)
rus.fit(X, Y)
X, Y= rus.fit_resample(X, Y)
ros = RandomOverSampler(random_state=0,sampling_strategy=0.8)
old = X
old["isFraud"] = Y
X_resampled, Y_resampled = ros.fit_resample(X, Y)
train_identity = X_resampled
train_identity["isFraud"] = Y_resampled.tolist()

old["isFraud"].plot(kind='hist',edgecolor='black')

train_identity["isFraud"].plot(kind='hist',edgecolor='black')

test_identity.to_csv('nstest.csv',index=False)
train_identity.to_csv('nstrain.csv',index=False)



