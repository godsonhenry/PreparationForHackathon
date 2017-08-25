# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 21:59:38 2017

@author: wuyuhang
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,LogisticRegression, Lasso, Ridge,ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math

# read in data
dataset=pd.read_excel('Business-Track-Application-Datasets.xlsx',sheetname='Training Data')
dataset=dataset.iloc[:,1:]

# data exploration
# explore the missing value
print('the missing matrix is \n',dataset.isnull().sum())

dataset.hist(figsize=(40,30),bins=50)
plt.show()

# Through check the dataset, some element which is 0 actually is missing value.
print('the missing values in column "FanSatisfaction" have',dataset.loc[dataset.FanSatisfaction==0,'FanSatisfaction'].count())
print('the missing values in column "TotRevSpend" have',dataset.loc[dataset.TotRevSpend==0,'TotRevSpend'].count())
# the TotRevSpend is the label so we must eliminate the missing value in label variable
dataset_NoMissing=dataset[dataset.TotRevSpend!=0].copy()
dataset_NoMissing=dataset_NoMissing[dataset_NoMissing.FanSatisfaction!=0].copy()
#dataset_NoMissing.loc[dataset_NoMissing.TotRevSpend==0,'TotRevSpend'].count()
#dataset_NoMissing.loc[dataset_NoMissing.FanSatisfaction==0,'FanSatisfaction'].count()
dataset_NoMissing.hist(figsize=(40,30),bins=50)
plt.show()

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
# there is no strong relationship in 

data=pd.DataFrame(columns=[dataset_NoMissing.columns[1:]])
for i in data.columns:
    data[i]=normalize(dataset_NoMissing[i].reshape(1,-1)).tolist()[0]
data.boxplot(figsize=(30,15))
for i,d in enumerate(data):
    y = data[d]
    x = np.random.normal(i+1, 0.04, len(y))
    plt.plot(x, y,mec='k', ms=7, marker="o", linestyle="None")

dataset_NoMissing.to_csv('cleanData.csv')






dataset_NM=pd.read_csv('cleanData.csv')

#Linear Regression
y=dataset_NM['TotRevSpend']
x=dataset_NM.iloc[:,2:]


#Seperate dataset
train_x,validate_x,train_y,validate_y= train_test_split(x,y, train_size=0.8,random_state=42)

regr=LinearRegression(normalize=True)
regr.fit(train_x,train_y)
train_x.columns
# The coefficients
print('Linear Coefficients: \n', regr.coef_)
# The mean squared error
print('Linear Intercept: \n',regr.intercept_)


# Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % regr.score(validate_x, validate_y))

#regr.predict(validate_x)

#Lasso Regression
las=Lasso(normalize=True)
las_model=GridSearchCV(las, param_grid={'alpha': np.logspace(-3, 3, 1000)}, cv=50)
las_model.fit(train_x,train_y)
print(las_model.best_params_) # alpha=0.011723818032865985 when cross validation = 5
print('R-Square:',las_model.score(validate_x, validate_y))

las=Lasso(alpha=0.1056875971184804,normalize=True)
las.fit(train_x,train_y)
print('Lasso Coefficients: \n', las.coef_)
print('Lasso Intercept: \n',las.intercept_)
#print('R-Square:',las.score(validate_x, validate_y))

#Ridge Regression
Rid=Ridge(normalize=True)
Rid_model=GridSearchCV(Rid, param_grid={'alpha': np.logspace(-3, 3, 1000)}, cv=50)
Rid_model.fit(train_x,train_y)
print(Rid_model.best_params_)#alpha=0.014831025143361045 when cross validation = 5
#print('R-Square:',Rid_model.score(validate_x, validate_y))

Rid=Ridge(alpha=0.068839520696454964,normalize=True)
Rid.fit(train_x,train_y)
print('Ridge Coefficients: \n', Rid.coef_)
print('Ridge Intercept: \n',Rid.intercept_)
#print('R-Square:',Rid.score(validate_x, validate_y))

#ElasticNet
#EN=ElasticNet(normalize=True)
#EN_model=GridSearchCV(EN, param_grid={'alpha': np.logspace(-3, 3, 1000)}, cv=5)
#EN_model.fit(train_x,train_y)
#print(EN_model.best_params_)
#print('R-Square:',EN_model.score(validate_x, validate_y))

#EN=ElasticNet(alpha=0.001,normalize=True)
#EN.fit(train_x,train_y)
#print('Coefficients: \n', EN.coef_)
#print('Intercept: \n',EN.intercept_)
#print('R-Square:',EN.score(validate_x, validate_y))

print('Trainset R-Square')
print('Linear Regression R-Square:' ,regr.score(train_x, train_y))
print('Lasso Regression R-Square:',las.score(train_x, train_y))
print('Ridge Regression R-Square:',Rid.score(train_x, train_y))
#print('ElasticNet R-Square:',EN.score(train_x, train_y))

print('Linear Regression RMSE:', math.sqrt(mean_squared_error(train_y, regr.predict(train_x))))
print('Lasso Regression RMSE:', math.sqrt(mean_squared_error(train_y, las.predict(train_x))))
print('Ridge Regression RMSE:', math.sqrt(mean_squared_error(train_y, Rid.predict(train_x))))

print('validation set R-Square')
print('Linear Regression R-Square:' ,regr.score(validate_x, validate_y))
print('Lasso Regression R-Square:',las.score(validate_x, validate_y))
print('Ridge Regression R-Square:',Rid.score(validate_x, validate_y))
#print('ElasticNet R-Square:',EN.score(validate_x, validate_y))

print('Linear Regression RMSE:', math.sqrt(mean_squared_error(validate_y,regr.predict(validate_x))))
print('Lasso Regression RMSE:', math.sqrt(mean_squared_error(validate_y,las.predict(validate_x))))
print('Ridge Regression RMSE:', math.sqrt(mean_squared_error(validate_y,Rid.predict(validate_x))))
#print('ElasticNet MSE:', mean_squared_error(validate_y,EN.predict(validate_x)))


# check the coefficient significance
#mod=sm.OLS(train_y,train_x)
#res=mod.fit()
#print(res.summary())#P value看不出来
#print('OLS RMSE:', math.sqrt(mean_squared_error(validate_y,res.predict(validate_x))))# x效果不好

# reduce the feature
dataset_NM=pd.read_csv('cleanData.csv')
dataset_feature=dataset_NM.drop(['FanSatisfaction','FanComplaints'],axis=1)
y=dataset_feature['TotRevSpend']
x=dataset_feature.iloc[:,2:]


#Seperate dataset
train_x,validate_x,train_y,validate_y= train_test_split(x,y, train_size=0.8,random_state=42)

#linear Refression
regr=LinearRegression(normalize=True)
regr.fit(train_x,train_y)
# The coefficients
print('Linear Coefficients: \n', regr.coef_)
# The mean squared error
print('Linear Intercept: \n',regr.intercept_)

#Lasso Regression
las=Lasso(normalize=True)
las_model=GridSearchCV(las, param_grid={'alpha': np.logspace(-3, 3, 1000)}, cv=5)
las_model.fit(train_x,train_y)
print(las_model.best_params_)
print('R-Square:',las_model.score(validate_x, validate_y))

las=Lasso(alpha=0.011723818032865985,normalize=True)
las.fit(train_x,train_y)
print('Lasso Coefficients: \n', las.coef_)
print('Lasso Intercept: \n',las.intercept_)
#print('R-Square:',las.score(validate_x, validate_y))

#Ridge Regression
Rid=Ridge(normalize=True)
Rid_model=GridSearchCV(Rid, param_grid={'alpha': np.logspace(-3, 3, 1000)}, cv=5)
Rid_model.fit(train_x,train_y)
print(Rid_model.best_params_)
#print('R-Square:',Rid_model.score(validate_x, validate_y))

Rid=Ridge(alpha=0.014831025143361045,normalize=True)
Rid.fit(train_x,train_y)
print('Ridge Coefficients: \n', Rid.coef_)
print('Ridge Intercept: \n',Rid.intercept_)


print('Trainset R-Square')
print('Linear Regression R-Square:' ,regr.score(train_x, train_y))
print('Lasso Regression R-Square:',las.score(train_x, train_y))
print('Ridge Regression R-Square:',Rid.score(train_x, train_y))
#print('ElasticNet R-Square:',EN.score(train_x, train_y))

print('Linear Regression RMSE:', math.sqrt(mean_squared_error(train_y, regr.predict(train_x))))
print('Lasso Regression RMSE:', math.sqrt(mean_squared_error(train_y, las.predict(train_x))))
print('Ridge Regression RMSE:', math.sqrt(mean_squared_error(train_y, Rid.predict(train_x))))

print('validation set R-Square')
print('Linear Regression R-Square:' ,regr.score(validate_x, validate_y))
print('Lasso Regression R-Square:',las.score(validate_x, validate_y))
print('Ridge Regression R-Square:',Rid.score(validate_x, validate_y))
#print('ElasticNet R-Square:',EN.score(validate_x, validate_y))

print('Linear Regression RMSE:', math.sqrt(mean_squared_error(validate_y,regr.predict(validate_x))))
print('Lasso Regression RMSE:', math.sqrt(mean_squared_error(validate_y,las.predict(validate_x))))
print('Ridge Regression RMSE:', math.sqrt(mean_squared_error(validate_y,Rid.predict(validate_x))))



















#Logistic Regression
for i in range(len(dataset_NM)):
    if dataset_NM.loc[i,'TotRevSpend']>=250:
        dataset_NM.loc[i,'>=250?']=1
    else:
        dataset_NM.loc[i,'>=250?']=0

print('amount of ">=250" instances:', len(dataset_NM[dataset_NM['>=250?']==1]))
print('amount of ">=250" instances:', len(dataset_NM[dataset_NM['>=250?']==0]))
print('Ratio of 1 to 0: ',len(dataset_NM[dataset_NM['>=250?']==1])/len(dataset_NM[dataset_NM['>=250?']==0]))
print('percentage of >=250: \n',dataset_NM['>=250?'].value_counts()/len(dataset_NM))

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

log=LogisticRegression()
log_model=GridSearchCV(log, param_grid={'C': np.logspace(-2, 2, 1000)}, cv=5)
log_model.fit(train_x,train_y)
print(log_model.best_params_)
print('mean accuracy:',log_model.score(validate_x, validate_y))
log=LogisticRegression(C=1.1643031329208768)
log.fit(train_x,train_y)
print('mean accuracy:',log.score(validate_x, validate_y))

print('prediction:',log.predict(validate_x))
print('probability [0,1]:',log.predict_proba(validate_x))



# misclasicification rate
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


# extend the '0' values
for i in range(len(dataset_NM)):
    if dataset_NM.loc[i,'TotRevSpend']>=250:
        dataset_NM.loc[i,'>=250?']=1
    else:
        dataset_NM.loc[i,'>=250?']=0

print('amount of ">=250" instances:', len(dataset_NM[dataset_NM['>=250?']==1]))
print('amount of ">=250" instances:', len(dataset_NM[dataset_NM['>=250?']==0]))
print('Ratio of 1 to 0: ',len(dataset_NM[dataset_NM['>=250?']==1])/len(dataset_NM[dataset_NM['>=250?']==0]))
print('percentage of >=250: \n',dataset_NM['>=250?'].value_counts()/len(dataset_NM))

dataset_0=dataset_NM[dataset_NM['>=250?']==0].sample(526,replace=True)

ResampleData=dataset_NM.append(dataset_0)
ResampleData.index=range(len(ResampleData))
print('Ratio of 1 to 0: ',len(ResampleData[ResampleData['>=250?']==1])/len(ResampleData[ResampleData['>=250?']==0]))
        
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

log=LogisticRegression()
log_model=GridSearchCV(log, param_grid={'C': np.logspace(-2, 2, 1000)}, cv=5)
log_model.fit(train_x,train_y)
print(log_model.best_params_) # 'C': 0.058717663907332553
print('mean accuracy:',log_model.score(validate_x, validate_y)) #mean accuracy: 0.760273972603
log=LogisticRegression(C=0.058717663907332553)
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
plt.title('ROC Curve - train set - AUC: 0.8771')
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
plt.title('ROC Curve - validation set - AUC: 0.8495')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.show()


