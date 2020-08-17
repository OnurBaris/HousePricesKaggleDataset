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

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
        
train_data = pd.read_csv("C:/01_Dosyalar/PythonFiles/Codes/KaggleHousePrices/train.csv")
test_data = pd.read_csv("C:/01_Dosyalar/PythonFiles/Codes/KaggleHousePrices/test.csv")
train_data.set_index('Id')
test_data.set_index('Id')

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
            train_data.loc[j, i]=1.0
        elif train_data.loc[j, i]=='Fa':
            train_data.loc[j, i]=0.4
        elif train_data.loc[j, i]=='TA':
            train_data.loc[j, i]=0.6
        elif train_data.loc[j, i]=='NA' or train_data.loc[j, i]=='na' or train_nan[j]==True:
            train_data.loc[j, i]=0.0
    for z in test_data.index:
        if test_data.loc[z,i]=='Gd':
            test_data.loc[z,i]=0.8
        elif test_data.loc[z, i]=='Po':
            test_data.loc[z,i]=0.2
        elif test_data.loc[z, i]=='Ex':
            test_data.loc[z, i]=1.0
        elif test_data.loc[z, i]=='Fa':
            test_data.loc[z, i]=0.4
        elif test_data.loc[z, i]=='TA':
            test_data.loc[z, i]=0.6
        elif test_data.loc[z, i]=='NA' or test_data.loc[z, i]=='na' or test_nan[z]==True:
            test_data.loc[z, i]=0.0
            
print(train_data.loc[:,condition].head())
print(test_data.loc[:,condition].head())