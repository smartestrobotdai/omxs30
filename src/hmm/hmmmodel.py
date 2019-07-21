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
from historicdata import HistoricData
import pickle

class HmmModel:
  def __init__(self, stock_name, slippage=0.00015):
    self.stock_name = stock_name
    self.stock_id = get_stock_id_by_name(stock_name)

    self.slippage = slippage
    self.X_list = None
    self.remodel = None
    self.strategy_model = None
    self.scaler = None
    self.np_value_table = None
    self.historic_data = None

  def prepare_data(self, start_day_index, end_day_index=None):
    n_components = self.n_components
    ema = self.ema
    beta = self.beta
    use_volume = self.use_volume
    ref_stock_id = self.ref_stock_id
    time_format = self.time_format
    stock_name = self.stock_name
    stock_id = self.stock_id

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
		

  def train(self, X_list, start_day_index, end_day_index, strategy_X_list=None):
    if len(X_list.shape)==2:
      X_list = X_list[0]

    self.n_components = int(X_list[0])
    self.ema = int(X_list[1])
    self.beta = int(X_list[2])
    self.use_volume = int(X_list[3])
    self.ref_stock_id = int(X_list[4])
    self.time_format = int(X_list[5])
    self.start_day_index = start_day_index
    self.end_day_index = end_day_index

    n_components = int(X_list[0])
    training_end_day_index = start_day_index + int((end_day_index - start_day_index) * 0.7)
    input_data, values, timestamps, prices = self.prepare_data(start_day_index, training_end_day_index)
    
    scaler = MinMaxScaler()
    shape = input_data.shape
    scaler.fit(input_data.reshape((shape[0]*shape[1],shape[2])))
    scaled = scaler.transform(input_data.reshape((shape[0]*shape[1],shape[2])))
    np.random.seed(42)
    remodel = hmm.GaussianHMM(n_components=n_components, 
      covariance_type="full", 
      n_iter=10, 
      random_state=42)

    states = None
    try:
      remodel.fit(scaled, lengths=[shape[1]]*shape[0])
      states = remodel.predict(scaled, lengths=[shape[1]]*shape[0])
    except (ValueError, linalg.LinAlgError) as ve:
      print("prediction failed: {}".format(ve))
    
    if states is None:
      return np.array(-1).reshape((1,1)), np.array(-1).reshape((1,1))
    
    values_flatten = values.flatten()
    states_values = np.stack((states, values_flatten), axis=1)

    df_s_v = pd.DataFrame(states_values, columns=['state','value']).fillna(0)
    value_table = df_s_v.groupby(['state']).mean()

    for i in range(n_components):
      if i not in value_table.index:
        value_table.loc[i] = 0

    np_value_table = value_table.to_numpy().flatten()


    strategy_input_data, real_values = self.prepare_strategy_input_data(remodel, scaler, 
      np_value_table, training_end_day_index, end_day_index)

    if strategy_X_list is None:
      strategy_factory = TradeStrategyFactory(slippage=self.slippage)
      strategy_model = strategy_factory.create_trade_strategies(strategy_input_data, iter=2)
    else:
      print("build strategy model: {}".format(strategy_X_list))
      strategy_model = TradeStrategy(strategy_X_list, slippage=self.slippage)

    total_profit, profit_daily, training_results = strategy_model.get_profit(strategy_input_data)

    profit_daily_avg = profit_daily.mean()
    print("training finished: total_profit: {}, profit_daily: {}".format(total_profit, profit_daily_avg))
    self.remodel = remodel
    self.strategy_model = strategy_model
    self.scaler = scaler
    self.np_value_table = np_value_table
    self.historic_data = HistoricData(start_day_index, end_day_index)
    print(strategy_input_data.shape)
    print(training_results.shape)
    print(values.shape)
    historic_data = np.concatenate((strategy_input_data, 
      training_results,
      remove_centralized(real_values[:,:,np.newaxis])), axis=2)

    print("historic data shape: {}".format(historic_data.shape))
    self.historic_data.set_training_data(historic_data)
    return total_profit, profit_daily_avg

  def prepare_strategy_input_data(self, remodel, scaler, 
      np_value_table, start_day_index, end_day_index=None):
    input_data, real_values, timestamps, prices = self.prepare_data(start_day_index, end_day_index)
    shape = input_data.shape
    scaled = scaler.transform(input_data.reshape((shape[0]*shape[1],shape[2]))).reshape(shape)

    n_days = shape[0]
    n_steps = shape[1]

    states = np.zeros((n_days, n_steps))
    for step_idx in range(n_steps):
      states_tmp = remodel.predict(scaled[:, :step_idx+1].reshape(shape[0]*(step_idx+1), shape[2]), 
        lengths=[step_idx+1]*n_days)

      states_tmp_reshaped = states_tmp.reshape((n_days, step_idx+1))
      states[:,step_idx] = states_tmp_reshaped[:,-1]

    print("states.shape")
    print(states.shape)

    values = self.get_values_by_states(states, np_value_table)
    strategy_data_input = np.stack((timestamps, values, prices), axis=2)
    strategy_data_input_no_central = remove_centralized(strategy_data_input)
    return strategy_data_input_no_central, real_values

  def test(self, test_start_day_index, test_end_day_index=None):
    strategy_data_input, real_values = self.prepare_strategy_input_data(self.remodel, self.scaler, 
      self.np_value_table, test_start_day_index, test_end_day_index)

    strategy_model = self.strategy_model

    total_profit, profit_daily, training_results = strategy_model.get_profit(strategy_data_input_no_central)
    print("test finished: total_profit: {}, profit_daily: {}".format(total_profit, profit_daily))

    print(strategy_data_input.shape)
    print(training_results.shape)
    print(real_values.shape)
    historic_data = np.concatenate((strategy_data_input_no_central, 
        training_results, 
        remove_centralized(real_values[:,:,np.newaxis])), axis=2)

    print("historic data shape: {}".format(historic_data.shape))
    self.historic_data.append(historic_data)
    return total_profit

  def get_values_by_states(self, states, np_value_table):
    state_2_value_func = np.vectorize(lambda x: np_value_table[int(x)])
    return state_2_value_func(states)

  def plot(self):
    self.historic_data.plot()

  def daily_plot(self):
    self.historic_data.daily_plot(self.strategy_model)
      
  def get_strategy_features(self):
    return self.strategy_model.to_list()

  def save(self, save_path):
    create_if_not_exist(save_path)
    hmm_model_filename = os.path.join(save_path, "hmm_model.pkl")
    assert(self.remodel is not None)
    with open(hmm_model_filename, "wb") as file: 
      pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

  def load(self, load_path):
    pass

