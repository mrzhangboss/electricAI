
# coding: utf-8

# In[1]:

import sys
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import scipy as sp

from sklearn.ensemble import IsolationForest


# In[2]:

if sys.argv[1] == 'test':
    is_train = False
    train_path = '../dataset/fetures/test.csv'
else:
    is_train = True
    if sys.argv[2].endswith('.json'):
        month = '9'
    else:
        month = sys.argv[2]
    train_path = '../dataset/fetures/{}/train.csv'.format(month)


# In[3]:

train = pd.read_csv(train_path, parse_dates=['record_date'])


# In[4]:

train.info()


# In[5]:

threshold = 0.95


# In[6]:

def clean_quantile(df):
    up = df.power_consumption.quantile(threshold)
    down = df.power_consumption.quantile(1-threshold)
    mean = df.power_consumption.mean()
    print(df['user_id'].iloc[0], 'up:', up, 'down:', down, 'mean:', mean, end='\n', file=open('../dataset/clean.txt', 'a+'))
    df.loc[(df.power_consumption>=up)|(df.power_consumption<=down), 'power_consumption'] = df.power_consumption.mean()
    return df


# In[7]:

train = train.groupby('user_id', as_index=True).apply(clean_quantile)


# # 去除这些公司后结果变差,可能是因为未来也会造成影响

# In[9]:

def clean_all_zero(df):
    ndf = df.loc[df.power_consumption!=1]
    if not ndf.empty:
        return df
    else:
        print(df['user_id'].iloc[0])
        return ndf


# In[10]:

# train = train.groupby('user_id').apply(clean_all_zero).reset_index(drop=True)
# train.info()


# In[ ]:

# test = test.groupby('user_id').apply(clean_all_zero).reset_index(drop=True)
# test.info()


# # 添加公司用电量平滑处理

# In[11]:

def rolling_power_consumption(df):
    ndf = df.set_index('record_date')
    ndf = ndf.rolling(2).mean()
    return ndf.reset_index()


# In[12]:

train = train.groupby('user_id').apply(rolling_power_consumption).dropna(subset=['user_id']).reset_index(drop=True)


# In[13]:

train.to_csv(train_path, index=False)
train.info()

