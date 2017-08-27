# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 21:17:50 2017

@author: wuyuhang
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,LogisticRegression, Lasso, Ridge,ElasticNet, RidgeClassifier, LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.model_selection import GridSearchCV,train_test_split, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error,roc_curve,auc
from sklearn.preprocessing import normalize
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
import os
import xgboost as xgb

#setting for running xgboost in windows
#path='C:\Program Files\mingw-w64\x86_64-7.1.0-posix-seh-rt_v5-rev2\mingw64\bin'
#os.environ['PATH'] = path + ';' + os.environ['PATH']

# read in data
dataset=pd.read_excel('Business-Track-Application-Datasets.xlsx',sheetname='Training Data')
dataset=dataset.iloc[:,1:]
# Visualization
dataset.hist(figsize=(40,30),bins=50)
plt.show()

#Data Explorary Analysis

#clean Missing value
print('the missing matrix is \n',dataset.isnull().sum())
print('the missing values in column "FanSatisfaction" have',dataset.loc[dataset.FanSatisfaction==0,'FanSatisfaction'].count())
print('the missing values in column "TotRevSpend" have',dataset.loc[dataset.TotRevSpend==0,'TotRevSpend'].count())
dataset_NoMissing=dataset[dataset.TotRevSpend!=0].copy()
dataset_NoMissing=dataset_NoMissing[dataset_NoMissing.FanSatisfaction!=0].copy()
#Visualization
dataset_NoMissing.hist(figsize=(40,30),bins=50)
plt.show()

#Correlation Matrix
corr_matrix=dataset_NoMissing.corr()
corr_matrix['TotRevSpend'].sort_values(ascending=False)
'''
TotRevSpend        1.000000
YrsInDatabase      0.435460
GamesWatched       0.390067
Income             0.371446
FanSatisfaction    0.178089
FanComplaints     -0.135933
DistToArena       -0.551067
Name: TotRevSpend, dtype: float64
'''
pd.plotting.scatter_matrix(dataset_NoMissing[dataset_NoMissing.columns[1:]],figsize=(20,15)) 

# check outlier
data=pd.DataFrame(columns=[dataset_NoMissing.columns[1:]])
for i in data.columns:
    data[i]=normalize(dataset_NoMissing[i].reshape(1,-1)).tolist()[0]
#visualization
data.boxplot(figsize=(30,15))
for i,d in enumerate(data):
    y = data[d]
    x = np.random.normal(i+1, 0.04, len(y))
    plt.plot(x, y,mec='k', ms=7, marker="o", linestyle="None")
    
#Output Clean Data
dataset_NoMissing.to_csv('cleanData.csv')

#moldeing
#read in clean data
dataset_NM=pd.read_csv('cleanData.csv')

#linear regression
y=dataset_NM['TotRevSpend']
x=dataset_NM.iloc[:,2:]

#split data into training dataset and validation dataset
train_x,validate_x,train_y,validate_y= train_test_split(x,y, train_size=0.8,random_state=42)

#linear regression
regr=LinearRegression(normalize=True)
regr.fit(train_x,train_y)
# The coefficients
print('Linear Coefficients: \n', regr.coef_)
# The intercept
print('Linear Intercept: \n',regr.intercept_)

#Lasso regression
las=Lasso(normalize=True)
las_model=GridSearchCV(las, param_grid={'alpha': np.logspace(-3, 3, 1000)}, cv=5)
las_model.fit(train_x,train_y)
print(las_model.best_params_)
las=Lasso(alpha=las_model.best_params_['alpha'],normalize=True)
las.fit(train_x,train_y)
print('Lasso Coefficients: \n', las.coef_)
print('Lasso Intercept: \n',las.intercept_)

#Ridge regression
Rid=Ridge(normalize=True)
Rid_model=GridSearchCV(Rid, param_grid={'alpha': np.logspace(-3, 3, 1000)}, cv=5)
Rid_model.fit(train_x,train_y)
print(Rid_model.best_params_)
Rid=Ridge(alpha=Rid_model.best_params_['alpha'],normalize=True)
Rid.fit(train_x,train_y)
print('Ridge Coefficients: \n', Rid.coef_)
print('Ridge Intercept: \n',Rid.intercept_)

# model evaluation
print('Trainset R-Square')
print('Linear Regression R-Square:' ,regr.score(train_x, train_y))
print('Lasso Regression R-Square:',las.score(train_x, train_y))
print('Ridge Regression R-Square:',Rid.score(train_x, train_y))

print('Linear Regression RMSE:', math.sqrt(mean_squared_error(train_y, regr.predict(train_x))))
print('Lasso Regression RMSE:', math.sqrt(mean_squared_error(train_y, las.predict(train_x))))
print('Ridge Regression RMSE:', math.sqrt(mean_squared_error(train_y, Rid.predict(train_x))))

print('validation set R-Square')
print('Linear Regression R-Square:' ,regr.score(validate_x, validate_y))
print('Lasso Regression R-Square:',las.score(validate_x, validate_y))
print('Ridge Regression R-Square:',Rid.score(validate_x, validate_y))

print('Linear Regression RMSE:', math.sqrt(mean_squared_error(validate_y,regr.predict(validate_x))))
print('Lasso Regression RMSE:', math.sqrt(mean_squared_error(validate_y,las.predict(validate_x))))
print('Ridge Regression RMSE:', math.sqrt(mean_squared_error(validate_y,Rid.predict(validate_x))))

#ways to fix overfitting - method_1 - reduce feature
dataset_NM=pd.read_csv('cleanData.csv')
dataset_feature=dataset_NM.drop(['FanSatisfaction','FanComplaints'],axis=1)
y=dataset_feature['TotRevSpend']
x=dataset_feature.iloc[:,2:]

train_x,validate_x,train_y,validate_y= train_test_split(x,y, train_size=0.8,random_state=42)

#linear regression
regr=LinearRegression(normalize=True)
regr.fit(train_x,train_y)
# The coefficients
print('Linear Coefficients: \n', regr.coef_)
# The intercept
print('Linear Intercept: \n',regr.intercept_)

#Lasso regression
las=Lasso(normalize=True)
las_model=GridSearchCV(las, param_grid={'alpha': np.logspace(-3, 3, 1000)}, cv=5)
las_model.fit(train_x,train_y)
print(las_model.best_params_)
las=Lasso(alpha=las_model.best_params_['alpha'],normalize=True)
las.fit(train_x,train_y)
print('Lasso Coefficients: \n', las.coef_)
print('Lasso Intercept: \n',las.intercept_)

#Ridge regression
Rid=Ridge(normalize=True)
Rid_model=GridSearchCV(Rid, param_grid={'alpha': np.logspace(-3, 3, 1000)}, cv=5)
Rid_model.fit(train_x,train_y)
print(Rid_model.best_params_)
Rid=Ridge(alpha=Rid_model.best_params_['alpha'],normalize=True)
Rid.fit(train_x,train_y)
print('Ridge Coefficients: \n', Rid.coef_)
print('Ridge Intercept: \n',Rid.intercept_)

# model evaluation
print('Trainset R-Square')
print('Linear Regression R-Square:' ,regr.score(train_x, train_y))
print('Lasso Regression R-Square:',las.score(train_x, train_y))
print('Ridge Regression R-Square:',Rid.score(train_x, train_y))

print('Linear Regression RMSE:', math.sqrt(mean_squared_error(train_y, regr.predict(train_x))))
print('Lasso Regression RMSE:', math.sqrt(mean_squared_error(train_y, las.predict(train_x))))
print('Ridge Regression RMSE:', math.sqrt(mean_squared_error(train_y, Rid.predict(train_x))))

print('validation set R-Square')
print('Linear Regression R-Square:' ,regr.score(validate_x, validate_y))
print('Lasso Regression R-Square:',las.score(validate_x, validate_y))
print('Ridge Regression R-Square:',Rid.score(validate_x, validate_y))

print('Linear Regression RMSE:', math.sqrt(mean_squared_error(validate_y,regr.predict(validate_x))))
print('Lasso Regression RMSE:', math.sqrt(mean_squared_error(validate_y,las.predict(validate_x))))
print('Ridge Regression RMSE:', math.sqrt(mean_squared_error(validate_y,Rid.predict(validate_x))))

# xgboost modeling
dataset_NM=pd.read_csv('cleanData.csv')

#linear regression
y=dataset_NM['TotRevSpend']
x=dataset_NM.iloc[:,2:]

#split data into training dataset and validation dataset
train_x,validate_x,train_y,validate_y= train_test_split(x,y, train_size=0.8,random_state=42)

dtrain=xgb.DMatrix(train_x,label=train_y)
dvalid=xgb.DMatrix(validate_x,label=validate_y)
watchlist=[(dtrain,'train'),(dvalid,'valid')]

#train model
xgb_pars = {'min_child_weight': 10, 'eta':0.001,'colsample_bytree': 0.8, 'max_depth': 4,
            'subsample': 0.8, 'lambda': 2, 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}

model = xgb.train(xgb_pars, dtrain, 3000, watchlist, early_stopping_rounds=50,
                  maximize=False, verbose_eval=100)

print('Modeling RMSLE %.5f' % model.best_score)

# prediction
Prediction=pd.read_excel('Business-Track-Application-Datasets.xlsx',sheetname='Prediction')
Prediction['Unnamed: 0']=range(len(Prediction))

dataset_NM=pd.read_csv('cleanData.csv')

#linear regression
y=dataset_NM['TotRevSpend']
x=dataset_NM.iloc[:,2:]

#split data into training dataset and validation dataset
train_x,validate_x,train_y,validate_y= train_test_split(x,y, train_size=0.8,random_state=42)
#train Ridge regression

Rid=Ridge(normalize=True)
Rid_model=GridSearchCV(Rid, param_grid={'alpha': np.logspace(-3, 3, 1000)}, cv=5)
Rid_model.fit(train_x,train_y)
print(Rid_model.best_params_)
Rid=Ridge(alpha=Rid_model.best_params_['alpha'],normalize=True)
Rid.fit(train_x,train_y)

test_x=Prediction[Prediction.columns[3:]]
Prediction.TotRevSpend=Rid.predict(test_x)




#Logistic regression
#read in dataset
dataset_NM=pd.read_csv('cleanData.csv')

# create binary variable
for i in range(len(dataset_NM)):
    if dataset_NM.loc[i,'TotRevSpend']>=250:
        dataset_NM.loc[i,'>=250?']=1
    else:
        dataset_NM.loc[i,'>=250?']=0

#data explorary analysis        
print('amount of ">=250" instances:', len(dataset_NM[dataset_NM['>=250?']==1]))
print('amount of ">=250" instances:', len(dataset_NM[dataset_NM['>=250?']==0]))
print('Ratio of 1 to 0: ',len(dataset_NM[dataset_NM['>=250?']==1])/len(dataset_NM[dataset_NM['>=250?']==0]))
print('percentage of >=250: \n',dataset_NM['>=250?'].value_counts()/len(dataset_NM))

#stratified data splitting
split=StratifiedShuffleSplit(n_splits=1,train_size=0.8,random_state=41)
for train_index,test_index in split.split(dataset_NM, dataset_NM['>=250?']):
    train_set=dataset_NM.loc[train_index]
    test_set=dataset_NM.loc[test_index]    
print('percentage of >=250 in train_set: \n',train_set['>=250?'].value_counts()/len(train_set))
print('percentage of >=250 in test_set: \n',test_set['>=250?'].value_counts()/len(test_set))
train_x=train_set.iloc[:,2:8]
train_y=train_set['>=250?']
validate_x=test_set.iloc[:,2:8]
validate_y=test_set['>=250?']

#train logistic regression model
log=LogisticRegression()
log_model=GridSearchCV(log, param_grid={'C': np.logspace(-2, 2, 1000)}, cv=5)
log_model.fit(train_x,train_y)
print(log_model.best_params_)
print('mean accuracy:',log_model.score(validate_x, validate_y))
log=LogisticRegression(C=log_model.best_params_['C'])
log.fit(train_x,train_y)
print('mean accuracy:',log.score(validate_x, validate_y))

#confusion Matrix
# training dataset
checkTable=pd.DataFrame({'pred':log.predict(train_x),'True':train_y})
SubsetZero=checkTable.loc[checkTable['True']==0,:].copy()
SubsetOne=checkTable.loc[checkTable['True']==1,:].copy()
print('TP:',len(SubsetOne.loc[SubsetOne.pred==1,:]))
print('FN:',len(SubsetOne.loc[SubsetOne.pred==0,:]))
print('TN:',len(SubsetZero.loc[SubsetZero.pred==0,:]))
print('FP:',len(SubsetZero.loc[SubsetZero.pred==1,:]))

print('Accuracy of 0:', len(SubsetZero.loc[SubsetZero.pred==0,:])/(len(SubsetZero.loc[SubsetZero.pred==0,:])+len(SubsetOne.loc[SubsetOne.pred==0,:])))
print('Accuracy of 1:', len(SubsetOne.loc[SubsetOne.pred==1,:])/(len(SubsetOne.loc[SubsetOne.pred==1,:])+len(SubsetZero.loc[SubsetZero.pred==1,:])))



# validation dataset
checkTable=pd.DataFrame({'pred':log.predict(validate_x),'True':validate_y})
SubsetZero=checkTable.loc[checkTable['True']==0,:].copy()
SubsetOne=checkTable.loc[checkTable['True']==1,:].copy()
print('TP:',len(SubsetOne.loc[SubsetOne.pred==1,:]))
print('FN:',len(SubsetOne.loc[SubsetOne.pred==0,:]))
print('TN:',len(SubsetZero.loc[SubsetZero.pred==0,:]))
print('FP:',len(SubsetZero.loc[SubsetZero.pred==1,:]))

print('Accuracy of 0:', len(SubsetZero.loc[SubsetZero.pred==0,:])/(len(SubsetZero.loc[SubsetZero.pred==0,:])+len(SubsetOne.loc[SubsetOne.pred==0,:])))
print('Accuracy of 1:', len(SubsetOne.loc[SubsetOne.pred==1,:])/(len(SubsetOne.loc[SubsetOne.pred==1,:])+len(SubsetZero.loc[SubsetZero.pred==1,:])))


# ROC curve
#training set
p=log.predict_proba(train_x)
p1=[i[1] for i in p]

fpr, tpr, thresholds = roc_curve(train_y, p1)

print('AUC score:',auc(fpr,tpr))

plt.figure()
plt.plot(fpr,tpr)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('ROC Curve - train set - AUC: 0.8869')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.show()

#validation set
p=log.predict_proba(validate_x)
p1=[i[1] for i in p]

fpr, tpr, thresholds = roc_curve(validate_y, p1)

print('AUC score:',auc(fpr,tpr))

plt.figure()
plt.plot(fpr,tpr)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('ROC Curve - validation set - AUC: 0.8589')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.show()

#Over Sample
dataset_NM=pd.read_csv('cleanData.csv')

# create binary variable
for i in range(len(dataset_NM)):
    if dataset_NM.loc[i,'TotRevSpend']>=250:
        dataset_NM.loc[i,'>=250?']=1
    else:
        dataset_NM.loc[i,'>=250?']=0

#extend dataset where binary laebl as 0
dataset_0=dataset_NM[dataset_NM['>=250?']==0].sample(526,replace=True, random_state=2017)
ResampleData=dataset_NM.append(dataset_0)
ResampleData.index=range(len(ResampleData))
print('Ratio of 1 to 0: ',len(ResampleData[ResampleData['>=250?']==1])/len(ResampleData[ResampleData['>=250?']==0]))

# stratified data splitting
split=StratifiedShuffleSplit(n_splits=1,train_size=0.8,random_state=40)
for train_index,test_index in split.split(ResampleData, ResampleData['>=250?']):
    train_set=ResampleData.loc[train_index]
    test_set=ResampleData.loc[test_index]
print('percentage of >=250 in train_set: \n',train_set['>=250?'].value_counts()/len(train_set))
print('percentage of >=250 in test_set: \n',test_set['>=250?'].value_counts()/len(test_set))
train_x=train_set.iloc[:,2:8]
train_y=train_set['>=250?']
validate_x=test_set.iloc[:,2:8]
validate_y=test_set['>=250?']

#train logistic regression model
log=LogisticRegression()
log_model=GridSearchCV(log, param_grid={'C': np.logspace(-2, 2, 1000)}, cv=5)
log_model.fit(train_x,train_y)
print(log_model.best_params_) # 'C': 0.058717663907332553
print('mean accuracy:',log_model.score(validate_x, validate_y)) #mean accuracy: 0.760273972603
log=LogisticRegression(C=log_model.best_params_['C'])
log.fit(train_x,train_y)
print('mean accuracy:',log.score(validate_x, validate_y))

# confusion Matrix
checkTable=pd.DataFrame({'pred':log.predict(train_x),'True':train_y})
SubsetZero=checkTable.loc[checkTable['True']==0,:].copy()
SubsetOne=checkTable.loc[checkTable['True']==1,:].copy()
print('Training Dataset')
print('TP:',len(SubsetOne.loc[SubsetOne.pred==1,:]))
print('FN:',len(SubsetOne.loc[SubsetOne.pred==0,:]))
print('TN:',len(SubsetZero.loc[SubsetZero.pred==0,:]))
print('FP:',len(SubsetZero.loc[SubsetZero.pred==1,:]))

print('Accuracy of 0:', len(SubsetZero.loc[SubsetZero.pred==0,:])/(len(SubsetZero.loc[SubsetZero.pred==0,:])+len(SubsetOne.loc[SubsetOne.pred==0,:])))
print('Accuracy of 1:', len(SubsetOne.loc[SubsetOne.pred==1,:])/(len(SubsetOne.loc[SubsetOne.pred==1,:])+len(SubsetZero.loc[SubsetZero.pred==1,:])))



# validation dataset
checkTable=pd.DataFrame({'pred':log.predict(validate_x),'True':validate_y})
SubsetZero=checkTable.loc[checkTable['True']==0,:].copy()
SubsetOne=checkTable.loc[checkTable['True']==1,:].copy()
print('Training Dataset')
print('TP:',len(SubsetOne.loc[SubsetOne.pred==1,:]))
print('FN:',len(SubsetOne.loc[SubsetOne.pred==0,:]))
print('TN:',len(SubsetZero.loc[SubsetZero.pred==0,:]))
print('FP:',len(SubsetZero.loc[SubsetZero.pred==1,:]))

print('Accuracy of 0:', len(SubsetZero.loc[SubsetZero.pred==0,:])/(len(SubsetZero.loc[SubsetZero.pred==0,:])+len(SubsetOne.loc[SubsetOne.pred==0,:])))
print('Accuracy of 1:', len(SubsetOne.loc[SubsetOne.pred==1,:])/(len(SubsetOne.loc[SubsetOne.pred==1,:])+len(SubsetZero.loc[SubsetZero.pred==1,:])))


# ROC curve
#training set
p=log.predict_proba(train_x)
p1=[i[1] for i in p]

fpr, tpr, thresholds = roc_curve(train_y, p1)

print('AUC score:',auc(fpr,tpr))

plt.figure()
plt.plot(fpr,tpr)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('ROC Curve - train set - AUC: 0.8678')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.show()

#validation set
p=log.predict_proba(validate_x)
p1=[i[1] for i in p]

fpr, tpr, thresholds = roc_curve(validate_y, p1)

print('AUC score:',auc(fpr,tpr))

plt.figure()
plt.plot(fpr,tpr)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('ROC Curve - validation set - AUC: 0.8617')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.show()

#Under-Sample
#read in data
train = pd.read_csv('cleanData.csv')
test=Prediction.loc[:,(Prediction.columns!= 'TotRevSpend')&(Prediction.columns!= 'LikelihoodOver250')].copy()

feature_names = list(train.columns)
do_not_use_for_training = ['TotRevSpend','class']
feature_names = [f for f in train.columns if f not in do_not_use_for_training]

#create stratified binary label
def class250(df):
    if df['TotRevSpend'] >= 250:
        val = 1
    elif df['TotRevSpend'] < 250:
        val = 0
    return val

train.loc[:,'class'] = train.apply(class250, axis=1)

print('0: ', train[train['class']==0]['class'].count())
print('1: ',train[train['class']==1]['class'].count())

# unnder-sample the data
train_0=train[train['class']==1].sample(202,replace=True)
train_1=train[train['class']==1].sample(202,replace=True)
train_2=train[train['class']==1].sample(202,replace=True)
train_3=train[train['class']==1].sample(202,replace=True)
train_4=train[train['class']==1].sample(202,replace=True)

train_all_0 = train[train['class']==0]

train_00 = pd.concat((train_all_0, train_0))
train_01 = pd.concat((train_all_0, train_1))
train_02 = pd.concat((train_all_0, train_2))
train_03 = pd.concat((train_all_0, train_3))
train_04 = pd.concat((train_all_0, train_4))

y0 = train_00['class']
y1 = train_01['class']
y2 = train_02['class']
y3 = train_03['class']
y4 = train_04['class']

X_train0, X_test0, y_train0, y_test0 = train_test_split(train_00[feature_names], y0, random_state = 1987, test_size=.0)

X_train1, X_test1, y_train1, y_test1 = train_test_split(train_01[feature_names], y0, random_state = 1987, test_size=.0)

X_train2, X_test2, y_train2, y_test2 = train_test_split(train_02[feature_names], y0, random_state = 1987, test_size=.0)

X_train3, X_test3, y_train3, y_test3 = train_test_split(train_03[feature_names], y0, random_state = 1987, test_size=.0)

X_train4, X_test4, y_train4, y_test4 = train_test_split(train_04[feature_names], y0, random_state = 1987, test_size=.0)

# train five models
# LinearSVC
model = LinearSVC(verbose = 1)
model0 = model.fit(X_train0, y_train0)
train_pred0 = model0.predict(train[feature_names])
test0 = model0.predict(test)

# PassiveAggressiveClassifier
model = PassiveAggressiveClassifier(n_iter=10)

model1= model.fit(X_train1, y_train1)
train_pred1 = model1.predict(train[feature_names])
test1 = model1.predict(test)

#LogisticRegression

model = LogisticRegression(verbose = 1)

model2= model.fit(X_train2, y_train2)
train_pred2 = model2.predict(train[feature_names])
test2 = model2.predict(test)

#SGDClassifier()
model = SGDClassifier()

model3= model.fit(X_train3, y_train3)
train_pred3 = model3.predict(train[feature_names])
test3 = model3.predict(test)

#RandomForestClassifier
model = RandomForestClassifier()
model4 = model.fit(X_train4, y_train4)
train_pred4 = model4.predict(train[feature_names])
test4 = model4.predict(test)

# adding corrected feature
t_pred0 = pd.Series(train_pred0)
t_pred1 = pd.Series(train_pred1)
t_pred2 = pd.Series(train_pred2)
t_pred3 = pd.Series(train_pred3)
t_pred4 = pd.Series(train_pred4)

train.loc[:,'t_pred0'] = t_pred0.values
train.loc[:,'t_pred1'] = t_pred1.values
train.loc[:,'t_pred2'] = t_pred2.values
train.loc[:,'t_pred3'] = t_pred3.values
train.loc[:,'t_pred4'] = t_pred4.values

#stratified data splitting
split=StratifiedShuffleSplit(n_splits=1,train_size=0.8,random_state=1988)
for train_index,test_index in split.split(train, train['class']):
    train_set=train.loc[train_index]
    test_set=train.loc[test_index]
    
print('percentage of >=250 in train_set: \n',train_set['class'].value_counts()/len(train_set))
print('percentage of >=250 in test_set: \n',test_set['class'].value_counts()/len(test_set))

train_x=train_set.iloc[:,[2,3,4,5,6,7,9,10,11,12,13]]
train_y=train_set['class']
validate_x=test_set.iloc[:,[2,3,4,5,6,7,9,10,11,12,13]]
validate_y=test_set['class']

#train second level of Logistic regression
log=LogisticRegression()
log_model=GridSearchCV(log, param_grid={'C': np.logspace(-2, 2, 1000)}, cv=5)
log_model.fit(train_x,train_y)
print(log_model.best_params_)

log=LogisticRegression(C=log_model.best_params_['C'])
log.fit(train_x,train_y)

# confusion Matrix
#training dataset
checkTable=pd.DataFrame({'pred':log.predict(train_x),'True':train_y})
SubsetZero=checkTable.loc[checkTable['True']==0,:].copy()
SubsetOne=checkTable.loc[checkTable['True']==1,:].copy()
print('Training Dataset')
print('TP:',len(SubsetOne.loc[SubsetOne.pred==1,:]))
print('FN:',len(SubsetOne.loc[SubsetOne.pred==0,:]))
print('TN:',len(SubsetZero.loc[SubsetZero.pred==0,:]))
print('FP:',len(SubsetZero.loc[SubsetZero.pred==1,:]))

print('Accuracy of 0:', len(SubsetZero.loc[SubsetZero.pred==0,:])/(len(SubsetZero.loc[SubsetZero.pred==0,:])+len(SubsetOne.loc[SubsetOne.pred==0,:])))
print('Accuracy of 1:', len(SubsetOne.loc[SubsetOne.pred==1,:])/(len(SubsetOne.loc[SubsetOne.pred==1,:])+len(SubsetZero.loc[SubsetZero.pred==1,:])))



# validation dataset
checkTable=pd.DataFrame({'pred':log.predict(validate_x),'True':validate_y})
SubsetZero=checkTable.loc[checkTable['True']==0,:].copy()
SubsetOne=checkTable.loc[checkTable['True']==1,:].copy()
print('Training Dataset')
print('TP:',len(SubsetOne.loc[SubsetOne.pred==1,:]))
print('FN:',len(SubsetOne.loc[SubsetOne.pred==0,:]))
print('TN:',len(SubsetZero.loc[SubsetZero.pred==0,:]))
print('FP:',len(SubsetZero.loc[SubsetZero.pred==1,:]))

print('Accuracy of 0:', len(SubsetZero.loc[SubsetZero.pred==0,:])/(len(SubsetZero.loc[SubsetZero.pred==0,:])+len(SubsetOne.loc[SubsetOne.pred==0,:])))
print('Accuracy of 1:', len(SubsetOne.loc[SubsetOne.pred==1,:])/(len(SubsetOne.loc[SubsetOne.pred==1,:])+len(SubsetZero.loc[SubsetZero.pred==1,:])))


# ROC curve
#training set
p=log.predict_proba(train_x)
p1=[i[1] for i in p]

fpr, tpr, thresholds = roc_curve(train_y, p1)

print('AUC score:',auc(fpr,tpr))

plt.figure()
plt.plot(fpr,tpr)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('ROC Curve - train set - AUC: 0.8716')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.show()

#validation set
p=log.predict_proba(validate_x)
p1=[i[1] for i in p]

fpr, tpr, thresholds = roc_curve(validate_y, p1)

print('AUC score:',auc(fpr,tpr))

plt.figure()
plt.plot(fpr,tpr)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('ROC Curve - validation set - AUC: 0.8952')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.show()


#prediction
t_pred0 = pd.Series(test0)
t_pred1 = pd.Series(test1)
t_pred2 = pd.Series(test2)
t_pred3 = pd.Series(test3)
t_pred4 = pd.Series(test4)

test.loc[:,'t_pred0'] = t_pred0.values
test.loc[:,'t_pred1'] = t_pred1.values
test.loc[:,'t_pred2'] = t_pred2.values
test.loc[:,'t_pred3'] = t_pred3.values
test.loc[:,'t_pred4'] = t_pred4.values

test=test[test.columns[1:]]
log.predict(test)
Prediction.LikelihoodOver250=[i[1] for i in log.predict_proba(test)]
Prediction[Prediction.columns[1:]].to_csv('PredictionFor100Instance.csv',index=False)














