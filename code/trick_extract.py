
# coding: utf-8

# In[59]:

import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import scipy as sp


# In[60]:

def loss_score(predict, real):
    f = (real - predict)/real
    n = len(f)
    f = f.replace([np.nan, -np.nan], 0)
    score = 1 - np.abs(f).sum()/n
    return score 
# from sklearn.metrics import r2_score
# loss_score = r2_score


# In[61]:

if sys.argv[1] == 'test':
    is_train = False
else:
    is_train = True


# In[62]:

if is_train:
    train_path = '../dataset/fetures/train.csv'
    predict_path = '../dataset/fetures/train_predict.csv'
    feture_path = '../dataset/fetures/train_feture.csv'
else:
    train_path = '../dataset/fetures/test.csv'
    predict_path = '../dataset/fetures/test_predict.csv'
    feture_path = '../dataset/fetures/test_feture.csv'


# In[63]:

train = pd.read_csv(train_path, parse_dates=['record_date'])


# In[64]:

predict = pd.read_csv(predict_path, parse_dates=['predict_date'], index_col=['predict_date'])


# # 移动划窗规则

# In[65]:

days = len(predict)


# In[66]:

avg_df = train.groupby('record_date')[['power_consumption']].sum()


# In[67]:

avg_df.index = avg_df.index + timedelta(days)


# In[68]:

avg_df.index.name = 'predict_date'


# In[69]:

avg_df['predict_power_consumption'] = avg_df.power_consumption.astype(int)


# In[70]:

rule_predict = pd.DataFrame(avg_df.predict_power_consumption, index=predict.index)


# In[71]:

if is_train:
    print('move windows score', loss_score( rule_predict, predict))


# # 平均值

# In[73]:

train.record_date.unique().shape


# In[74]:

def avg_score(month, year='2016'):
    df = train.set_index('record_date').loc['{}/{}/1'.format(year, month):]
    days = df.index.unique().shape[0]
    mean_consumption = np.int64(df.power_consumption.sum()/days)

    predict['avg'] = mean_consumption

    if is_train:
        loss = loss_score(predict.avg, predict.predict_power_consumption)
        print('from {}-{} '.format(year, month), 'avg {} score', mean_consumption, loss_score(predict.predict_power_consumption, predict.avg))

    


# In[75]:

for year in [2015, 2016]:
    for month in range(1, 13):
        if not(year == 2016 and month > 6):
            avg_score(month, year)


# # 选择2015-1月到至今的平均值

# In[76]:

avg_score(6, 2016) if is_train else avg_score(9, 2016)


# In[78]:

if not is_train:
    predict['predict_power_consumption'] = predict.avg
    predict.index = predict.index.map(lambda x:x.strftime('%Y%m%d'))
    predict.index.name = 'predict_date'
    predict.to_csv('Tianchi_power_predict_table.csv', columns=['predict_power_consumption'])

