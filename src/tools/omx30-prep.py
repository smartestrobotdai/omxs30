#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import sys
import warnings
import math
sys.path.append("../")
from util import get_stock_name_by_id, get_stock_id_by_name, preprocessing_daily_data, add_step_columns

stock_name_tuple = ('NDA-SE', 'ESSITY-B', 'GETI-B', 'AZN', 'HM-B') 



df=pd.read_csv('../../data/data.csv.gz', compression='gzip', sep=',')
df['timestamp'] = pd.to_datetime(df['time_stamp'], format="%Y-%m-%d %H:%M:%S").dt.tz_convert('CET')


# In[ ]:


stock_delta = df['stock_id'] - df['stock_id'].shift()
time_delta = (df['timestamp'] - df['timestamp'].shift()).fillna(10000).dt.total_seconds()


# In[ ]:


stock_split_index=df.index[stock_delta!=0].tolist()
stock_split_index.append(len(df))


# In[ ]:


day_split_index=df.index[abs(time_delta)>36000].tolist()
day_split_index.insert(0, 0)
day_split_index.append(len(df))


# In[ ]:


stock_id_list = df['stock_id'].unique().tolist()
i = 0
for stock_index in stock_id_list:
    print("index: {}, id:{}".format(i, stock_index))
    i+=1


# In[ ]:
def get_stock_index_by_id(stock_id):
    for i in range(len(stock_id_list)):
        if stock_id_list[i] == stock_id:
            return i
    return None

def get_stock_id_by_index(stock_index):
    return stock_id_list[stock_index]


current_stock_index = 0
time_series_list = []
time_series_all_list = []
for i in range(len(day_split_index) - 1):
    start_index = day_split_index[i]
    end_index = day_split_index[i + 1]
    day_time_series = df.iloc[start_index:end_index]
    time_series_list.append(day_time_series)
    if end_index == stock_split_index[current_stock_index + 1]:
        print("finished stock: {}, got {} time series".format(stock_id_list[current_stock_index], len(time_series_list)))
        time_series_all_list.append(time_series_list)
        time_series_list = []
        current_stock_index += 1


# In[ ]:
print("Checking the minimal change rates:")
for stock_id in range(len(stock_id_list)):
    diff = time_series_all_list[stock_id][0]['open'].diff()

    min_ = diff[diff>=0.005].min()
    open_ = time_series_all_list[stock_id][-1]['open'].mean()

    print("stockid:{} min={}, cost={}".format(stock_id_list[stock_id], min_, min_/open_))


# In[ ]:


# initialize a new list
value_result_list = []
for s_id in range(len(time_series_all_list)):
    new_list = [None] * len(time_series_all_list[s_id])
    value_result_list.append(new_list)
    

#for stock_id in range(len(time_series_all_list)):
for stock_name in stock_name_tuple:
    stock_id = get_stock_id_by_name(stock_name)
    stock_index = get_stock_index_by_id(stock_id)
    print("handling stock_id: {}, stock_index: {}, stock_name:{}".format(stock_id, stock_index, stock_name))
    for day_id in range(len(time_series_all_list[stock_index])):
        # :-1 is that we don't like the last record at 17:30 which is a aggregated number.
        df = time_series_all_list[stock_index][day_id].copy()

        if day_id == 0:
            last_close = None
        else:
            last_close = time_series_all_list[stock_index][day_id-1]['last'].iloc[-1]

        df_new = preprocessing_daily_data(df, last_close, calculate_values=True)
        # drop the first row because diff is nan    
        #df.drop(0, inplace=True)
        value_result_list[stock_index][day_id] = df_new.fillna(0)

# In[ ]:


# save to csv files
column_wanted_in_order = ['timestamp', 'last', 'volume', 
                          'diff_ema_1', 'diff_ema_5', 
                          'diff_ema_10', 'diff_ema_20', 
                          'value_ema_1_beta_98', 'value_ema_1_beta_99',
                          'value_ema_5_beta_98', 'value_ema_5_beta_99', 
                          'value_ema_10_beta_98', 'value_ema_10_beta_99', 
                          'value_ema_20_beta_98', 'value_ema_20_beta_99']


npy_save_path = '../../preprocessed-data/'
#for s_id in range(len(value_result_list)):
for stock_name in stock_name_tuple:
    stock_id = get_stock_id_by_name(stock_name)
    stock_index = get_stock_index_by_id(stock_id)
    df_merged = value_result_list[stock_index][0][column_wanted_in_order]
    df_merged = add_step_columns(df_merged)
    for day_id in range(1, len(value_result_list[stock_index])):
        df = value_result_list[stock_index][day_id][column_wanted_in_order]
        df = add_step_columns(df)
        df_merged = df_merged.append(df)
    
    
    for ema in (20, 10, 1):
        for beta in (99,98):
            print("Saving to files for stock id:{}, name:{} index:{} ema:{} beta:{}".format(stock_id, 
                stock_name,
                stock_index, 
                ema, 
                beta))
            npy_filename = npy_save_path + "{}_{}_ema{}_beta{}.npy".format(stock_name, stock_id, ema, beta, stock_id)
            groups = df_merged.set_index('timestamp').groupby(lambda x: x.date())
            data_list = []
            column_list = ['step_of_day',
                       'step_of_week',
                       'diff_ema_{}'.format(ema), 
                       'volume',
                       'value_ema_{}_beta_{}'.format(ema, beta),
                       'timestamp',
                       'last']
            
            for index, df in groups:
                np_data = df.reset_index().rename({'index':'timestamp'}, axis=1)[column_list].values
                data_list.append(np_data)
            np.save(npy_filename, np.array(data_list))





