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
import matplotlib.pyplot as plt

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

dataset_NoMissing.to_csv('cleanData.csv')






dataset_NM=pd.read_csv('cleanData.csv')

#Linear Regression
y=dataset_NM['TotRevSpend']
x=dataset_NM.iloc[:,2:]


#Seperate dataset
train_x,validate_x,train_y,validate_y= train_test_split(x,y, train_size=0.8,random_state=42)

regr=LinearRegression(normalize=True)
regr.fit(train_x,train_y)
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Intercept: \n',regr.intercept_)

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(validate_x, validate_y))

#regr.predict(validate_x)

#Lasso Regression
las=Lasso(normalize=True)
las_model=GridSearchCV(las, param_grid={'alpha': np.logspace(-3, 3, 1000)}, cv=5)
las_model.fit(train_x,train_y)
print(las_model.best_params_)
print('R-Square:',las_model.score(validate_x, validate_y))

las=Lasso(alpha=0.011723818032865985,normalize=True)
las.fit(train_x,train_y)
print('Coefficients: \n', las.coef_)
print('Intercept: \n',las.intercept_)
print('R-Square:',las.score(validate_x, validate_y))

#Ridge Regression
Rid=Ridge(normalize=True)
Rid_model=GridSearchCV(Rid, param_grid={'alpha': np.logspace(-3, 3, 1000)}, cv=5)
Rid_model.fit(train_x,train_y)
print(Rid_model.best_params_)
print('R-Square:',Rid_model.score(validate_x, validate_y))

Rid=Ridge(alpha=0.014831025143361045,normalize=True)
Rid.fit(train_x,train_y)
print('Coefficients: \n', Rid.coef_)
print('Intercept: \n',Rid.intercept_)
print('R-Square:',Rid.score(validate_x, validate_y))

#ElasticNet
EN=ElasticNet(normalize=True)
EN_model=GridSearchCV(EN, param_grid={'alpha': np.logspace(-3, 3, 1000)}, cv=5)
EN_model.fit(train_x,train_y)
print(EN_model.best_params_)
print('R-Square:',EN_model.score(validate_x, validate_y))

EN=ElasticNet(alpha=0.001,normalize=True)
EN.fit(train_x,train_y)
print('Coefficients: \n', EN.coef_)
print('Intercept: \n',EN.intercept_)
print('R-Square:',EN.score(validate_x, validate_y))

print('Linear Regression R-Square:' ,regr.score(validate_x, validate_y))
print('Lasso Regression R-Square:',las.score(validate_x, validate_y))
print('Ridge Regression R-Square:',Rid.score(validate_x, validate_y))
print('ElasticNet R-Square:',EN.score(validate_x, validate_y))









#Logistic Regression
for i in range(len(dataset_NM)):
    if dataset_NM.loc[i,'TotRevSpend']>=250:
        dataset_NM.loc[i,'>=250?']=1
    else:
        dataset_NM.loc[i,'>=250?']=0

print('amount of ">=250" instances:', len(dataset_NM[dataset_NM['>=250?']==1]))
print('amount of ">=250" instances:', len(dataset_NM[dataset_NM['>=250?']==0]))

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
print(EN_model.best_params_)
print('mean accuracy:',log_model.score(validate_x, validate_y))
log=LogisticRegression(C=0.001)
log.fit(train_x,train_y)
print('mean accuracy:',log.score(validate_x, validate_y))

print('prediction:',log.predict(validate_x))
print('probability [0,1]:',log.predict_proba(validate_x))



        



