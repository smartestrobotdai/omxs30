import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from functools import partial
from util import timestamp2date
from ipywidgets import interact
import ipywidgets as widgets
import pandas as pd

class HistoricData:
  def __init__(self, start_day_index, end_day_index):
    pass

  def set_training_data(self, data):
    self.data = data
    self.training_data_len = len(data)

  def append(self, data):
    first_date = timestamp2date(data[0,0,0])
    find_date = False
    for i in range(len(self.data)):
      date = timestamp2date(self.data[i,0,0])
      if date == first_date:
          find_date = True
          break

    if find_date == True:
        origin_last_date = timestamp2date(self.data[-1,0,0])
        last_date = timestamp2date(data[-1,0,0])
        print("overwritting data ended at date: {} from date: {} to date: {} ".format(origin_last_date, first_date, last_date))
        self.data = np.concatenate((self.data[:i], data), axis=0)
    else:
        i+=1
        origin_last_date = timestamp2date(self.data[-1,0,0])
        last_date = timestamp2date(data[-1,0,0])
        print("appending data ended at date: {} from date: {} to date: {} ".format(origin_last_date, first_date, last_date))
        self.data = np.concatenate((self.data, data), axis=0)
    return len(data)

  def get_training_data(self):
    training_data_length = self.training_data_len
    return self.data[:training_data_length]

  def get_test_data(self):
    training_data_length = self.training_data_len
    return self.data[training_data_length:]

  def get_daily_data(self):
    stock_change_rate = self.data[:,:,3]
    asset_change_rate = self.data[:,:,4]
    daily_stock_change_rate = np.prod(stock_change_rate+1, axis=1)-1
    daily_asset_change_rate = np.prod(asset_change_rate+1, axis=1)-1
    date = self.data[:,0,0]
    return np.stack((date, daily_stock_change_rate, daily_asset_change_rate), axis=1)

  def get_training_data_len(self):
    return self.training_data_len

  def plot(self):
    assert(self.data is not None)
    training_data_length = self.get_training_data_len()

    daily_data = self.get_daily_data()
    print("preparing training plot")
    self.plot_helper(daily_data[:training_data_length])

    length = len(daily_data)
    if length > training_data_length:
      print("preparing test plot")
      self.plot_helper(daily_data[training_data_length:])

  def plot_helper(self, daily_data):
    x1 = daily_data[:,0]

    stock_change = np.cumprod(daily_data[:,1]+1)
    asset_change = np.cumprod(daily_data[:,2]+1)

    plt.subplot(1, 1, 1)
    
    plt.plot(x1,stock_change, label='stock')
    
    plt.plot(x1,asset_change, label='asset')
    plt.legend()

    plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%m-%d'))
    #plt.gca().xaxis.set_major_locator(dates.DateLocator())

    plt.gcf().autofmt_xdate()
    plt.grid()
    plt.show()

  def daily_plot(self, strategy_model):
    data_len = self.data.shape[0]
    self.strategy_model = strategy_model
    #daily_plot_func = partial(self.daily_plot_func, strategy_model)
    interact(self.daily_plot_func, day_index=widgets.IntSlider(min=0, max=data_len-1, value=data_len-1))


  def daily_plot_func(self, day_index):
    strategy_model = self.strategy_model
    # the stock price change rate
    stock = np.cumprod(self.data[day_index,:,3]+1)
    # the asset change rate
    asset = np.cumprod(self.data[day_index,:,4]+1)
    # the date
    date = timestamp2date(self.data[day_index, 0, 0])

    #actions 
    action = self.data[day_index,:,5]
    timestamp = self.data[day_index, :, 0]
    df_action = pd.DataFrame(np.stack((timestamp, action), axis=1), columns=['timestamp','action'])
    print(df_action[df_action['action'] != 0])


    print("Date: {}, Day_index:{}, ".format(date, day_index))
    # timestamp
    x = self.data[day_index, :, 0]

    plt.subplot(3, 1, 1)
    plt.plot(x,stock,label='stock')
    plt.plot(x,asset,label='asset')
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.grid()


    # get the thresholds, 
    strategy_features = self.strategy_model.get_features()
    buy_threshold = np.array([strategy_features[0]] * len(x))
    sell_threshold = np.array([strategy_features[1]] * len(x))

    # the real values
    real_values = self.data[day_index,:,6]
    # the predicted values


    plt.subplot(3, 1, 2)
    plt.plot(x, real_values,label='real')
    plt.plot(x, buy_threshold, label='buy')
    plt.plot(x, sell_threshold, label='sell')
    #plt.legend()
    plt.gcf().autofmt_xdate()
    plt.grid()

    predicted_values = self.data[day_index,:,1]
    value_ma = strategy_features[5]

    # do ma on values
    if value_ma != 1:
        values = pd.Series(predicted_values)
        values_ma = values.ewm(span=value_ma, adjust=False).mean()
        values_ma = values_ma.values
    else:
        values_ma = predicted_values

    plt.subplot(3, 1, 3)
    plt.plot(x, values_ma, label='predicted')
    plt.plot(x, buy_threshold, label='buy')
    plt.plot(x, sell_threshold, label='sell')
    #plt.legend()
    plt.gcf().autofmt_xdate()
    plt.grid()
    plt.show()

