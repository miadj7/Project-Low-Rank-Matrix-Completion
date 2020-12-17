# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from datetime import datetime
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

import os

import warnings
warnings.filterwarnings('ignore')

def read_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    return train, test

def clear_data(train, test):
    
    quantitative = [f for f in train.columns if train.dtypes[f] != 'object']
    quantitative.remove('SalePrice')
    quantitative.remove('Id')
    qualitative = [f for f in train.columns if train.dtypes[f] == 'object']
    
    qual_encoded = []
    for q in qualitative:  
        _encode(train, q)
        qual_encoded.append(q+'_E')
        
    features = quantitative + qual_encoded
    
    train.drop(['Id'], axis=1, inplace=True)
    test.drop(['Id'], axis=1, inplace=True)
    
    train = train[train.GrLivArea < 4500]
    train.reset_index(drop=True, inplace=True)
    train["SalePrice"] = np.log1p(train["SalePrice"])
    y = train['SalePrice'].reset_index(drop=True)
    
    train_features = train.drop(['SalePrice'], axis=1)
    test_features = test
    features = pd.concat([train_features, test_features]).reset_index(drop=True)
    
    features['MSSubClass'] = features['MSSubClass'].apply(str)
    features['YrSold'] = features['YrSold'].astype(str)
    features['MoSold'] = features['MoSold'].astype(str)
    features['Functional'] = features['Functional'].fillna('Typ') 
    features['Electrical'] = features['Electrical'].fillna("SBrkr") 
    features['KitchenQual'] = features['KitchenQual'].fillna("TA") 
    features["PoolQC"] = features["PoolQC"].fillna("None")
    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0]) 
    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
    
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        features[col] = features[col].fillna(0)

    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        features[col] = features[col].fillna('None')

    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        features[col] = features[col].fillna('None')

    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
    
    objects = []
    for i in features.columns:
        if features[i].dtype == object:
            objects.append(i)
    features.update(features[objects].fillna('None'))

    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerics = []
    for i in features.columns:
        if features[i].dtype in numeric_dtypes:
            numerics.append(i)
    features.update(features[numerics].fillna(0))
    
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerics2 = []
    for i in features.columns:
        if features[i].dtype in numeric_dtypes:
            numerics2.append(i)
    skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

    high_skew = skew_features[skew_features > 0.5]
    skew_index = high_skew.index

    for i in skew_index:
        features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))
        
    
    features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']
    features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']

    features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                                 features['1stFlrSF'] + features['2ndFlrSF'])

    features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +
                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))

    features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                              features['EnclosedPorch'] + features['ScreenPorch'] +
                              features['WoodDeckSF'])
    
    features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    
    final_features = pd.get_dummies(features).reset_index(drop=True)
    
    X = final_features.iloc[:len(y), :]
    X_sub = final_features.iloc[len(y):, :]
    X.shape, y.shape, X_sub.shape
    
    outliers = [30, 88, 462, 631, 1322]
    X = X.drop(X.index[outliers])
    y = y.drop(y.index[outliers])

    overfit = []
    for i in X.columns:
        counts = X[i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(X) * 100 > 99.94:
            overfit.append(i)

    overfit = list(overfit)
    X = X.drop(overfit, axis=1)
    X_sub = X_sub.drop(overfit, axis=1)
    
    return X, X_sub, y
    
def _encode(frame, feature):
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['spmean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
    ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1, ordering.shape[0]+1)
    ordering = ordering['ordering'].to_dict()
    
    for cat, o in ordering.items():
        frame.loc[frame[feature] == cat, feature+'_E'] = o
        
def ـspearman(frame, features):
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['spearman'] = [frame[f].corr(frame['SalePrice'], 'spearman') for f in features]
    spr = spr.sort_values('spearman')
    plt.figure(figsize=(6, 0.25*len(features)))
    sns.barplot(data=spr, y='feature', x='spearman', orient='h')
