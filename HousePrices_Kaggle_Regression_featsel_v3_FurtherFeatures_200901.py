# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 11:49:37 2020

@author: obaris
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
        
train_data = pd.read_csv("C:/01_Projects/09_CriticalFormulasandTools/PythonScripts/Kaggle_HousePrices/train.csv", index_col='Id')
test_data = pd.read_csv("C:/01_Projects/09_CriticalFormulasandTools/PythonScripts/Kaggle_HousePrices/test.csv", index_col='Id')

condition = ['KitchenQual', 'PoolQC', 'GarageCond', 'BsmtCond', 'FireplaceQu', 'HeatingQC', 'BsmtQual', 'GarageQual', 'ExterQual', 'ExterCond']

for i in condition:
    train_nan = train_data.loc[:,i].isna()
    test_nan = test_data.loc[:,i].isna()
    for j in train_data.index:
        if train_data.loc[j,i]=='Gd':
            train_data.loc[j,i]=0.8
        elif train_data.loc[j, i]=='Po':
            train_data.loc[j,i]=0.2
        elif train_data.loc[j, i]=='Ex':
            train_data.loc[j, i]=float(1.0)
        elif train_data.loc[j, i]=='Fa':
            train_data.loc[j, i]=0.4
        elif train_data.loc[j, i]=='TA':
            train_data.loc[j, i]=0.6
        elif train_data.loc[j, i]=='NA' or train_data.loc[j, i]=='na' or train_nan[j]:
            train_data.loc[j, i]=float(0.0)
    for z in test_data.index:
        if test_data.loc[z,i]=='Gd':
            test_data.loc[z,i]=0.8
        elif test_data.loc[z, i]=='Po':
            test_data.loc[z,i]=0.2
        elif test_data.loc[z, i]=='Ex':
            test_data.loc[z, i]=float(1.0)
        elif test_data.loc[z, i]=='Fa':
            test_data.loc[z, i]=0.4
        elif test_data.loc[z, i]=='TA':
            test_data.loc[z, i]=0.6
        elif test_data.loc[z, i]=='NA' or test_data.loc[z, i]=='na' or test_nan[z]:
            test_data.loc[z, i]=float(0.0)

features = ['LotArea', 'LotFrontage', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF','MasVnrArea', 'OverallQual', 'FullBath', 'HalfBath',
            'BedroomAbvGr', 'TotRmsAbvGrd', 'GrLivArea', 'GarageArea', 'Fireplaces', 'OpenPorchSF', 'WoodDeckSF', 'TotalBsmtSF', 'BsmtFinSF1',
            'BsmtFullBath', 'BsmtHalfBath', 'YrSold']
features.extend(condition)

corelation = features[:]
corelation.append('SalePrice')
# Looking at the correlation matrix Id, MSSubClass, BsmtFinSF2, LowQualFinSF, BsmtHalfBath, 35snPorch, ScreenPorch, PoolArea
# MiscVal have weak correlation to price sold, already removed.
train_reduced = train_data[corelation]
corrmat = train_reduced.corr(method='spearman')
f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corrmat, ax=ax, annot=True, fmt=".1f", annot_kws={'size':8}, center=0, linewidths=0.1)
#After checking correlation matrix, remove PoolQC and ExterCond features since no corelation to SalePrice is observed
features.remove('PoolQC')
features.remove('ExterCond')
y = train_data['SalePrice']
X = train_data[features]
X_realtest = test_data[features]

values = {'LotFrontage': 0, 'MasVnrArea': 0}
test_values = {'GarageArea':0, 'TotalBsmtSF': X_realtest['TotalBsmtSF'].mean(), 'BsmtFinSF1': X_realtest['BsmtFinSF1'].mean(), 
               'BsmtFullBath': X_realtest['BsmtFullBath'].mean(), 'BsmtHalfBath': X_realtest['BsmtHalfBath'].mean()}
X.fillna(value=values, inplace=True)
X_realtest.fillna(value=values, inplace=True)
X_realtest.fillna(value=test_values, inplace=True)

X.loc[:,'House_Age']=X.loc[:,'YrSold']-X.loc[:,'YearBuilt']
X.loc[:,'Bath']=X.loc[:,'FullBath']+0.5*X.loc[:,'HalfBath']
X.loc[:,'BsmtBath']=X.loc[:,'BsmtFullBath']+0.5*X.loc[:,'BsmtHalfBath']
X.loc[:,'TotalSA']=X.loc[:,'1stFlrSF']+X.loc[:,'2ndFlrSF']+X.loc[:,'TotalBsmtSF']

X_realtest.loc[:,'House_Age']=X_realtest.loc[:,'YrSold']-X_realtest.loc[:,'YearBuilt']
X_realtest.loc[:,'Bath']=X_realtest.loc[:,'FullBath']+0.5*X_realtest.loc[:,'HalfBath']
X_realtest.loc[:,'BsmtBath']=X_realtest.loc[:,'BsmtFullBath']+0.5*X_realtest.loc[:,'BsmtHalfBath']
X_realtest.loc[:,'TotalSA']=X_realtest.loc[:,'1stFlrSF']+X_realtest.loc[:,'2ndFlrSF']+X_realtest.loc[:,'TotalBsmtSF']

X.drop(columns=['YrSold','YearBuilt','Bath','FullBath','HalfBath','BsmtBath','BsmtFullBath',
                'BsmtHalfBath','TotalSA','1stFlrSF','2ndFlrSF','TotalBsmtSF'], inplace=True)
X_realtest.drop(columns=['YrSold','YearBuilt','Bath','FullBath','HalfBath','BsmtBath','BsmtFullBath',
                'BsmtHalfBath','TotalSA','1stFlrSF','2ndFlrSF','TotalBsmtSF'], inplace=True)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_test = scaler.fit_transform(X_realtest)

condition.remove('PoolQC')
condition.remove('ExterCond')

for i in condition:
    X.loc[:,i] = X.loc[:,i].astype(float)
    X_realtest.loc[:,i] = X_realtest.loc[:,i].astype(float)

#Split data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#fit models
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from xgboost import XGBRegressor
model1 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=42)
model2 = GaussianProcessRegressor()
model3 = AdaBoostRegressor(n_estimators=500, learning_rate=0.1)
model4 = GradientBoostingRegressor(n_estimators=500, max_depth=4, min_samples_split=5, learning_rate=0.1, loss='ls')
model5 = MLPRegressor(random_state=42, max_iter=10000, hidden_layer_sizes=(100,), alpha=0.00001)
model6 = XGBRegressor(booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.6, gamma=0, importance_type='gain', learning_rate=0.01, max_delta_step=0,
             max_depth=4, min_child_weight=1.5, n_estimators=2400, n_jobs=1, nthread=None, objective='reg:linear',
             reg_alpha=0.6, reg_lambda=0.6, scale_pos_weight=1, silent=None, subsample=0.8, verbosity=1)

models = [model1, model2, model3, model4, model6]

def score_model(model, X_t=X_train, X_v=X_test, y_t=y_train, y_v=y_test):
    model.fit(X_t, y_t)
    y_pred = model.predict(X_v)
    return mean_squared_log_error(y_v, y_pred), mean_squared_error(y_v, y_pred)

for model in models:
    print(score_model(model))
"""
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)  
model5.fit(X_train, y_train)
y_pred = model5.predict(X_test)
print(mean_squared_log_error(y_test, y_pred))"""




