#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import sys
import warnings
import math

df=pd.read_csv('../data/data.csv.gz', compression='gzip', sep=',')
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


current_stock_index = 0
time_series_list = []
time_series_all_list = []
for i in range(len(day_split_index) - 1):
    start_index = day_split_index[i]
    end_index = day_split_index[i + 1]
    day_time_series = df.iloc[start_index:end_index]
    time_series_list.append(day_time_series)
    if end_index == stock_split_index[current_stock_index + 1]:
        print("finished stock: {}, got {} time series".format(current_stock_index, len(time_series_list)))
        time_series_all_list.append(time_series_list)
        time_series_list = []
        current_stock_index += 1


# In[ ]:


for stock_id in range(len(stock_id_list)):
    diff = time_series_all_list[stock_id][0]['open'].diff()

    min_ = diff[diff>=0.005].min()
    open_ = time_series_all_list[stock_id][-1]['open'].mean()

    print("stockid:{} min={}, cost={}".format(stock_id, min_, min_/open_))


# In[ ]:


# initialize a new list
value_result_list = []
for s_id in range(len(time_series_all_list)):
    new_list = [None] * len(time_series_all_list[0])
    value_result_list.append(new_list)
    

#for stock_id in range(len(time_series_all_list)):
for stock_id in (5,):
    print("handling stock_id: {}".format(stock_id))
    for day_id in range(len(time_series_all_list[stock_id])):
        # :-1 is that we don't like the last record at 17:30 which is a aggregated number.
        df = time_series_all_list[stock_id][day_id].copy()
        # some data might miss, we must make a right join with full time series
        # and do fillna.
        df2 = df.set_index('timestamp')
        ts = df2.index.min()
        start_time_str = "{}-{:02d}-{:02d} 8:54:00".format(ts.year, ts.month, ts.day)
        start_ts = pd.Timestamp(start_time_str, tz=ts.tz)
        # periods=510 means from 9 to 17.29
        dti = pd.date_range(start_ts , periods=516, freq='min').to_series(keep_tz=True).rename('time')
        # remove from 17.25 - 17.28
        #dti.drop(dti.tail(5).head(4).index, inplace=True)
        df3 = df2.join(dti, how='right')
        if day_id == 0: # the first day, we must set the value from 8.55-8.59 as same as 9.00
            df3['last'].iloc[0] = df3['last'].iloc[6]
        else:
            df3['last'].iloc[0] = time_series_all_list[stock_id][day_id-1]['last'].iloc[-1]
        
        
        df3['last'].interpolate(method='linear', inplace=True)
        df3['volume'].iloc[:6] = df3['volume'].iloc[6] / 6
        df3['volume'].iloc[6] = df3['volume'].iloc[6] / 6
        df3['volume'].iloc[-5:] = df3['volume'].iloc[-1]/5
        
        #TODO: FIXED ME, no nan in volume!
        
        
        df = df3.reset_index().rename({'index':'timestamp'}, axis=1)

        
        #df['timestamp'] = pd.to_datetime(df['time_stamp'], format="%Y-%m-%d %H:%M:%S").dt.tz_convert('Europe/Stockholm')
        df['ema_1'] = df['last']
        df['ema_5'] = df['last'].ewm(span=5, adjust=False).mean()
        df['ema_10'] = df['last'].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df['last'].ewm(span=20, adjust=False).mean()
        df['diff_ema_1']=(df['ema_1'].diff()[1:]/df['ema_1']).fillna(0)
        df['diff_ema_5']=(df['ema_5'].diff()[1:]/df['ema_5']).fillna(0)
        df['diff_ema_10']=(df['ema_10'].diff()[1:]/df['ema_10']).fillna(0)
        df['diff_ema_20']=(df['ema_20'].diff()[1:]/df['ema_20']).fillna(0)
        
        df['volume'] = df['volume'].abs().fillna(0)
        # the first diff at 9:00 is the difference between today's open and yesterday's last.
        df['value_ema_1_beta_99'] = 0
        df['value_ema_5_beta_99'] = 0
        df['value_ema_10_beta_99'] = 0
        df['value_ema_20_beta_99'] = 0
        df['value_ema_1_beta_98'] = 0
        df['value_ema_5_beta_98'] = 0
        df['value_ema_10_beta_98'] = 0
        df['value_ema_20_beta_98'] = 0
        for iter_id in range(20):
            df['value_ema_1_beta_99'] = df['diff_ema_1'].shift(-1).fillna(0) +                 0.99 * df['value_ema_1_beta_99'].shift(-1).fillna(0)
            df['value_ema_1_beta_98'] = df['diff_ema_1'].shift(-1).fillna(0) +                 0.98 * df['value_ema_1_beta_98'].shift(-1).fillna(0)
            df['value_ema_5_beta_99'] = df['diff_ema_5'].shift(-1).fillna(0) +                 0.99 * df['value_ema_5_beta_99'].shift(-1).fillna(0)
            df['value_ema_5_beta_98'] = df['diff_ema_5'].shift(-1).fillna(0) +                 0.98 * df['value_ema_5_beta_98'].shift(-1).fillna(0)
            df['value_ema_10_beta_99'] = df['diff_ema_10'].shift(-1).fillna(0) +                 0.99 * df['value_ema_10_beta_99'].shift(-1).fillna(0)
            df['value_ema_10_beta_98'] = df['diff_ema_10'].shift(-1).fillna(0) +                 0.98 * df['value_ema_10_beta_98'].shift(-1).fillna(0)
            df['value_ema_20_beta_99'] = df['diff_ema_20'].shift(-1).fillna(0) +                 0.99 * df['value_ema_20_beta_99'].shift(-1).fillna(0)
            df['value_ema_20_beta_98'] = df['diff_ema_20'].shift(-1).fillna(0) +                 0.98 * df['value_ema_20_beta_98'].shift(-1).fillna(0)
        # drop the first row because diff is nan    
        #df.drop(0, inplace=True)
        value_result_list[stock_id][day_id] = df.fillna(0)


# In[ ]:


value_result_list[5][0]


# In[ ]:


# save to csv files
column_wanted_in_order = ['timestamp', 'last', 'volume', 
                          'diff_ema_1', 'diff_ema_5', 
                          'diff_ema_10', 'diff_ema_20', 
                          'value_ema_1_beta_98', 'value_ema_1_beta_99',
                          'value_ema_5_beta_98', 'value_ema_5_beta_99', 
                          'value_ema_10_beta_98', 'value_ema_10_beta_99', 
                          'value_ema_20_beta_98', 'value_ema_20_beta_99']

def add_step_columns(df):
    df['step_of_day'] = np.arange(0, len(df))
    day_of_week = df['timestamp'].iloc[0].weekday()
    df['step_of_week'] = len(df) * day_of_week + df['step_of_day']
    return df

csv_save_path = 'csv_files/'
npy_save_path = 'npy_files/'
#for s_id in range(len(value_result_list)):
for s_id in (5,):
    df_merged = value_result_list[s_id][0][column_wanted_in_order]
    df_merged = add_step_columns(df_merged)
    for day_id in range(1, len(value_result_list[s_id])):
        df = value_result_list[s_id][day_id][column_wanted_in_order]
        df = add_step_columns(df)
        df_merged = df_merged.append(df)
    
    
    for ema in (10, 20):
        for beta in (99, 98):
            print("Saving to files for stock id:{} ema:{} beta:{}".format(s_id, ema, beta))
            npy_filename = npy_save_path + "ema{}_beta{}_{}.npy".format(ema, beta, s_id)
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


# In[ ]:




