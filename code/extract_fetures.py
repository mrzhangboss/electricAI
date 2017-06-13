
# coding: utf-8

# In[1]:

import sys
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import scipy as sp


# In[2]:

def loss_score(predict, real):
    f = (real - predict)/real
    n = len(f)
    f = f.replace([np.nan, -np.nan], 0)
    score = 1 - np.abs(f).sum()/n
    return score 


# In[3]:

if sys.argv[1] == 'test':
    is_train = False
    train_path = '../dataset/fetures/test.csv'
    predict_path = '../dataset/fetures/test_predict.csv'
    feture_path = '../dataset/fetures/test_feture.csv'
    
else:
    is_train = True
    if sys.argv[2].endswith('.json'):
        month = '9'
    else:
        month = sys.argv[2]
    train_path = '../dataset/fetures/{}/train.csv'.format(month)
    predict_path = '../dataset/fetures/{}/train_predict.csv'.format(month)
    feture_path = '../dataset/fetures/{}/train_feture.csv'.format(month)
    


# In[4]:

train = pd.read_csv(train_path, parse_dates=['record_date'])
predict = pd.read_csv(predict_path, parse_dates=['predict_date'])


# In[5]:

end_date = train.record_date.max().date()


# In[6]:

def create_timespan(end_date, m_span):
    predict_start = end_date + timedelta(1)
    n_month = predict_start.month - m_span
    if n_month < 1:
        n_month = 12 + n_month
        n_year = predict_start.year - 1
    else:
        n_year = predict_start.year
    n_date = date(n_year, n_month, predict_start.day)
    return (predict_start - n_date).days - 1


# In[7]:

# create_timespan(end_date, 13)


# In[8]:

time_spans = [30, 60, 90, 180, 366]
# time_spans = [30, 60, 90, 150, 180, 210, 270, 300, 366] # over fit


# In[9]:

# for span in time_month_spans:
#     time_spans.append( create_timespan(end_date, span) )


# In[10]:

print(time_spans)


# In[11]:

used_data_fetures = [
    'dayofweek', 'dayofyear', 'days_in_month', 'quarter', 'week', 'weekofyear',
    'month', 'year'
]


# In[12]:

def get_train_df(date_span, train):
    begin_date = end_date - timedelta(date_span)
    train_df = train.set_index(['record_date']).loc[str(begin_date):str(end_date)].reset_index()
    return train_df


# In[13]:

def add_feture_in_date(used_feture, predict, train_df):
    for f in used_data_fetures:
        predict[f] = getattr(predict['predict_date'].dt, f)
        train_df[f] = getattr(train_df['record_date'].dt, f)


# In[14]:

def extract_describe_feture(train_df, predict, f, date_span):
    df = train_df.groupby(f).describe()['power_consumption'].unstack()
    column_fmt = 'post{}_{}_{}'
    df.columns = [column_fmt.format(date_span, f, x) for x in   df.columns]
    predict = predict.join(df, on=f)
    return predict


# In[15]:

def extract_post_day_describe(date_span, train, predict, used_data_fetures, add_feture=True):
    train_df = get_train_df(date_span, train)
    if add_feture:
        add_feture_in_date(used_data_fetures, predict, train_df)
    for f in used_data_fetures:
        predict = extract_describe_feture(train_df, predict, f, date_span)
    return predict
    


# In[16]:

for date_span in time_spans:
    predict = extract_post_day_describe(date_span, train, predict, used_data_fetures)


# # 节假日特征 

# In[17]:

use_holiday_fetures = ['is_week', 'is_weekend', 'is_festival', 'is_holiday']


# In[18]:

holiday_df = pd.read_csv('../dataset/holiday.csv', parse_dates=['date'])
holiday_df.head(2)


# In[19]:

holiday_df['is_holiday'] = 0

holiday_df.loc[holiday_df.holiday!=0, 'is_holiday'] = 1


# In[20]:

df = pd.get_dummies(holiday_df.holiday)


# In[21]:

df.columns = ['is_week', 'is_weekend', 'is_festival']


# In[22]:

holiday_df = holiday_df.join(df)


# In[23]:

holiday_df.drop('holiday', axis=1, inplace=True)


# In[24]:

holiday_df.set_index('date', inplace=True)


# #### 添加日期特征至训练集 

# In[25]:

train = train.join(holiday_df, on='record_date')
predict = predict.join(holiday_df, on='predict_date')


# In[26]:

for date_span in time_spans:
    predict = extract_post_day_describe(
        date_span,
        train,
        predict,
        use_holiday_fetures,
        add_feture=False)


# ### 假期平均统计特征 

# In[27]:

def apply_describe_day_sum(group):
    df = group.groupby('record_date')[['power_consumption']].sum().describe().unstack()['power_consumption']
    return df


# In[28]:

def extract_day_consumption_describe(span, f, train, predict, add_dt_feture=False):
    tdf = get_train_df(span, train)
    if add_dt_feture:
        add_feture_in_date([f], predict, tdf)
    name_fmt = 'post{}_{}_{}_day_consumption'
    df = tdf.groupby(f).apply(apply_describe_day_sum)
    df.columns = [name_fmt.format(span, f, x) for x in df.columns]
    predict = predict.join(df, on=f)
    return predict


# In[29]:

# extract_day_consumption_describe(30, 'is_week', train, predict)


# In[30]:

for f in use_holiday_fetures:
    for span in time_spans:
        if f != 'is_weekend':  # compelete same with is_week
            predict = extract_day_consumption_describe(span, f, train, predict)


# # 总数统计特征 

# In[31]:

used_sum_fetures = [
    'dayofweek', 'dayofyear', 'days_in_month', 
    'quarter', 'week', 
    'month', 'year'
]


# In[32]:

def extract_mean_consumption(train_df, feture, date_span, predict, div_times):
    column_fmt = 'post{}_{}_mean_consumption'
    df = train_df.groupby(feture)['power_consumption'].sum() / (date_span / div_times)
    df.name = column_fmt.format(date_span, feture)
    predict = predict.join(df)
    return predict


# In[33]:

def extract_post_mean_consumption(feture, div_times, date_span, train, predict):
    train_df = get_train_df(30, train)
    add_feture_in_date(used_sum_fetures, predict, train_df)
    predict = extract_mean_consumption(train_df, feture, date_span, predict, div_times)
    return predict


# In[34]:

def extract_all_post_mean_consumption(used_sum_feture_model, train, predict):   
    for feture, div_times, date_spans in used_sum_feture_model:
        for date_span in date_spans:
            predict = extract_post_mean_consumption(feture, div_times, date_span, train, predict)
    return predict


# In[35]:

used_sum_feture_model = [
    ('dayofweek', 7, [30, 60, 90, 180, 360]),
    ('dayofyear', 1, [30, 60, 90, 180, 360]),
    ('days_in_month', 30, [30, 60, 90, 180, 360]),
    ('quarter', 90, [90, 180, 360]),
    ('week', 52, [180, 360]),
    ('month', 30, [30, 60, 90, 120, 240, 360]),
    ('year', 360, [ 360]), 
]


# In[36]:

# predict = extract_all_post_mean_consumption(used_sum_feture_model, train, predict)  # over fit


# In[37]:

# extract_day_consumption_describe(30, 'dayofweek', train, predict, add_dt_feture=True)


# In[38]:

for f in used_sum_fetures:
    for span in time_spans:
        predict = extract_day_consumption_describe(span, f, train, predict, add_dt_feture=True)


# # 天气特征 

# In[39]:

header = ['weather_date', 'weather_max', 'weather_min', 'weather_type', 'weather_wind', 'wind_type']


# In[40]:

weather_df = pd.read_csv('../dataset/yangzhong.csv', header=None, names=header, parse_dates=['weather_date'])
weather_df.head(3)


# #### 将天气切分成块,再提取块边界 

# In[41]:

weather_df.weather_min = pd.cut(weather_df.weather_min, bins=10)


# In[42]:

weather_df.weather_min = weather_df.weather_min.str.extract('\((-?\d+\.?\d*),').astype(np.float)


# In[43]:

weather_df.weather_max = pd.cut(weather_df.weather_max, bins=10)


# In[44]:

weather_df.weather_max = weather_df.weather_max.str.extract(', (-?\d+\.?\d*)\]').astype(np.float)


# In[45]:

weather_df.shape


# In[46]:

weather_type_count = weather_df.weather_type.value_counts()


# In[47]:

weather_df.weather_type = weather_df.weather_type.replace([
    x for x in weather_type_count.loc[weather_type_count < 2].index
], 'rare_weather')


# In[48]:

weather_df.loc[weather_df.weather_type.str.contains('阵雨'), 'weather_type'] = 'showers_weather'


# In[49]:

weather_df.loc[weather_df.weather_type.str.contains('雨'), 'weather_type'] = 'rain_weather'


# In[50]:

weather_df.loc[~weather_df.weather_type.str.islower(), 'weather_type'] = 'fine_weather'


# In[51]:

weather_df = weather_df.join(pd.get_dummies(weather_df.weather_type))


# In[75]:

weather_num_columns = [
    'weather_max', 'weather_min', 'wind_type', 'fine_weather', 'rain_weather',
    'rare_weather', 'showers_weather'
]


# In[53]:

weather_df.head()


# In[54]:

weather_df.weather_wind.replace('东南风', 'southeast_wind', inplace=True)
weather_df.weather_wind.replace('东北风', 'northeast_wind', inplace=True)
weather_df.weather_wind.replace('西南风', 'southwest_wind', inplace=True)
weather_df.weather_wind.replace('西北风', 'northwest_wind', inplace=True)
weather_df.weather_wind.replace('东风', 'east_wind', inplace=True)
weather_df.weather_wind.replace('北风', 'north_wind', inplace=True)
weather_df.weather_wind.replace('南风', 'south_wind', inplace=True)
weather_df.weather_wind.replace('西风', 'west_wind', inplace=True)


# In[55]:

weather_df.weather_wind.replace(['3-4级', '暂无实况', '无持续风向'], 'unknow_wind', inplace=True)


# In[56]:

weather_df = weather_df.join(pd.get_dummies(weather_df.weather_wind))


# In[57]:

wind_type_count = weather_df.wind_type.value_counts()


# In[58]:

weather_df.wind_type = weather_df.wind_type.replace([
    x for x in wind_type_count.loc[wind_type_count < 5 ].index
], 'rare_wind')


# In[59]:

weather_df.wind_type.replace('rare_wind', 0, inplace=True)
weather_df.wind_type.replace('微风', 1, inplace=True)
weather_df.wind_type.replace('1级', 2, inplace=True)
weather_df.wind_type.replace('2级', 3, inplace=True)
weather_df.wind_type.replace('小于3级', 4, inplace=True)
weather_df.wind_type.replace('3级', 6, inplace=True)
weather_df.wind_type.replace('3-4级转小于3级', 5, inplace=True)
weather_df.wind_type.replace('3-4级', 7, inplace=True)
weather_df.wind_type.replace('4-5级转3-4级', 8, inplace=True)
weather_df.wind_type.replace('4-5级', 9, inplace=True)


# In[60]:

weather_df.wind_type.value_counts()


# In[61]:

weather_df.head()


# In[62]:

train = train.join(weather_df.set_index('weather_date'), on='record_date')


# In[63]:

predict = predict.join(weather_df.set_index('weather_date'), on='predict_date')


# ### 添加天气特征(超前) 

# In[64]:

use_weather_fetures = ['weather_max', 'weather_min', 'weather_type', 'weather_wind', 'wind_type']


# In[65]:

for date_span in time_spans:
    predict = extract_post_day_describe(
        date_span,
        train,
        predict,
        use_weather_fetures,
        add_feture=False)


# In[66]:

# extract_day_consumption_describe(30, 'weather_max', train, predict, add_dt_feture=False)


# In[67]:

for f in use_weather_fetures:
    for span in time_spans:
        predict = extract_day_consumption_describe(span, f, train, predict, add_dt_feture=False)


# # 交叉特征

# ##  时间 X 假期

# In[73]:

def extract_combin_feture_day_consumption_describe(span, f1, f2, predict, train, f1_need_add=True, f2_need_add=False):
    tdf = get_train_df(span, train)
    if f1_need_add:
        add_feture_in_date([f1], predict, tdf)
    if f2_need_add:
        add_feture_in_date([f2], predict, tdf) 
    new_feture_name = 'post{}_combine_{}_and_{}'.format(span, f1, f2)
    predict[new_feture_name] = predict[f1] * predict[f2]
    tdf[new_feture_name] = tdf[f1] * tdf[f2]
    predict = extract_day_consumption_describe(span, new_feture_name, tdf, predict)
    return predict


# In[69]:

# extract_combin_feture_day_consumption_describe(30, 'dayofweek', 'is_week', predict, train)


# In[74]:

for span in time_spans:
    for f1 in used_data_fetures:
        for f2 in use_holiday_fetures:
            predict = extract_combin_feture_day_consumption_describe(span, f1, f2, predict, train)


# ## 时间 X 天气 

# In[78]:

for span in time_spans:
    for f1 in used_data_fetures:
        for f2 in weather_num_columns:
            predict = extract_combin_feture_day_consumption_describe(span, f1, f2, predict, train)


# ## 天气 X 假期

# In[79]:

for span in time_spans:
    for f1 in weather_num_columns:
        for f2 in use_holiday_fetures:
            predict = extract_combin_feture_day_consumption_describe(span, f1, f2, predict, train, False, False)


# In[ ]:

not_use_columns = ['weather_type', 'weather_wind']


# In[ ]:

# data_feture_not_use = [x for x in used_data_fetures if  x.startswith('is_')]


# In[ ]:

feture_columns = [x for x in predict.columns if x not in not_use_columns ]


# In[ ]:

# feture_columns = [x for x in feture_columns if x not in use_weather_fetures]


# In[ ]:

# feture_columns = [x for x in feture_columns if x not in use_holiday_fetures]


# In[ ]:

'weather_max' in feture_columns, 'is_week' in feture_columns


# In[ ]:

predict.to_csv(feture_path, columns=feture_columns, index=False)

