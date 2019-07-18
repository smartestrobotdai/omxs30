import sys
import os.path
import numpy as np
sys.path.append("../")
from util import *
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from hmmlearn import hmm
from numpy import linalg
import pandas as pd
from tradestrategy import TradeStrategyFactory, TradeStrategy
import pickle

class HmmModel:
  def __init__(self, stock_name, start_day_index, end_day_index, slippage=0.00015):
    self.stock_name = stock_name
    self.stock_id = get_stock_id_by_name(stock_name)
    self.start_day_index = start_day_index
    self.end_day_index = end_day_index
    self.slippage = slippage
    self.X_list = None
    self.remodel = None
    self.strategy_model = None
    self.scaler = None
    self.value_table = None

  def prepare_data(self):
    n_components = self.n_components
    ema = self.ema
    beta = self.beta
    use_volume = self.use_volume
    ref_stock_id = self.ref_stock_id
    time_format = self.time_format
    stock_name = self.stock_name
    stock_id = self.stock_id
    start_day_index = self.start_day_index
    end_day_index = self.end_day_index

    arr = np.load(get_npy_filename(stock_name, stock_id, ema, beta), allow_pickle=True)
    arr = arr[start_day_index:end_day_index]

    input_data = arr[:,:,[2]]

    if use_volume == 1:
      input_data = np.concatenate((input_data, arr[:,:,[3]]), axis=2)

    output_data = arr[:,:,4]
    timestamps = arr[:,:,5]
    prices = arr[:,:,6]
    
    time_steps = None
    if time_format == 0:
        time_steps = arr[:,:,[1]]
    elif time_format == 1:
        time_steps = arr[:,:,[2]]

    if ref_stock_id != -1 and ref_stock_id != stock_id:
        ref_stock_name = get_stock_name_by_id(ref_stock_id)
        arr_ref = np.load(get_npy_filename(ref_stock_name, ref_stock_id, ema, beta), allow_pickle=True)
        arr_ref = arr_ref[start_day_index:end_day_index]
        input_data = np.concatenate((input_data, arr_ref[:,:,[2]]), axis=2)
    
    if time_steps is not None:
        input_data = np.concatenate((input_data, time_steps), axis=2)

    return input_data, output_data, timestamps, prices
		

  def train(self, X_list, strategy_X_list=None):
    if len(X_list.shape)==2:
      X_list = X_list[0]

    self.n_components = int(X_list[0])
    self.ema = int(X_list[1])
    self.beta = int(X_list[2])
    self.use_volume = int(X_list[3])
    self.ref_stock_id = int(X_list[4])
    self.time_format = int(X_list[5])

    n_components = int(X_list[0])
    input_data, values, timestamps, prices = self.prepare_data()
    
    scaler = MinMaxScaler()
    shape = input_data.shape
    scaler.fit(input_data.reshape((shape[0]*shape[1],shape[2])))
    scaled = scaler.transform(input_data.reshape((shape[0]*shape[1],shape[2])))
    np.random.seed(42)
    remodel = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=10, random_state=42)
    states = None
    try:
      remodel.fit(scaled, lengths=[shape[1]]*shape[0])
      states = remodel.predict(scaled, lengths=[shape[1]]*shape[0])
    except (ValueError, linalg.LinAlgError) as ve:
      print("prediction failed: {}".format(ve))
    
    if states is None:
      return np.array(-1).reshape((1,1))
    
    print(np.unique(states, return_counts=True))
    values = values.flatten()
    states_values = np.stack((states, values), axis=1)

    df_s_v = pd.DataFrame(states_values, columns=['state','value']).fillna(0)
    value_table = df_s_v.groupby(['state']).mean().to_numpy()

    df_s_v['avg_values'] = df_s_v['state'].apply(lambda x: value_table[x])
    shape = timestamps.shape
    strategy_data_input = np.stack((timestamps, df_s_v['avg_values'].to_numpy().reshape(shape), prices), axis=2)
    strategy_data_input_no_central = remove_centralized(strategy_data_input)
    if strategy_X_list is None:
      factory = TradeStrategyFactory(slippage=self.slippage)
      strategy_model = trade_strategy.create_trade_strategies(strategy_data_input_no_central, iter=1)
    else:
    	strategy_model = TradeStrategy(strategy_X_list, slippage=self.slippage)

    total_profit, profit_daily, results = strategy_model.get_profit(strategy_data_input_no_central)
   
    self.remodel = remodel
    self.strategy_model = strategy_model
    self.scaler = scaler
    self.value_table = value_table
    return total_profit

  def get_strategy_features(self):
    return self.strategy_model.to_list()

  def save(self, save_path):
    create_if_not_exist(save_path)
    hmm_model_filename = os.path.join(save_path, "hmm_model.pkl")
    assert(self.remodel is not None)
    with open(hmm_model_filename, "wb") as file: 
      pickle.dump(self, file)

  def load(self, load_path):
    pass