import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import GPy
import GPyOpt
import sys
import os.path
import os
sys.path.append("../")
from tradestrategy import TradeStrategyFactory
from util import remove_centralized, get_stock_name_by_id, create_if_not_exist, get_stock_id_by_name
from hmm_util import get_cache_filename
from numpy import linalg
from functools import partial
from optimizeresult import OptimizeResult


def get_npy_filename(stock_name, stock_id, ema, beta):
	return "../../preprocessed-data/{}_{}_ema{}_beta{}.npy".format(stock_name, stock_id, ema, beta)

def hmm_prepare_data(stock_name, start_day_index, end_day_index, X_list):
  if len(X_list.shape) == 2:
    X_list = X_list[0]

  n_components = int(X_list[0])
  ema = int(X_list[1])
  beta = int(X_list[2])
  use_volume = int(X_list[3])
  ref_stock_id = int(X_list[4])
  time_format = int(X_list[5])
  
  stock_id = get_stock_id_by_name(stock_name)
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

def opt_func_base(stock_name, start_day_index, end_day_index, X_list):
	if len(X_list.shape) == 2:
		X_list = X_list[0]

	optimize_result = OptimizeResult()
	cache_filename = get_cache_filename(stock_name, start_day_index, end_day_index)

	if os.path.isfile(cache_filename):
	    optimize_result.load(cache_filename)
	    #print("stock worm cache loaded, size={}".format(optimize_result.get_size()))
	else:
	    print("cannot find file cache:{}, will create new cache.".format(cache_filename))
	print("Checking: {}".format(X_list))

	total_profit, remodel, strategy_model, *x = train(stock_name, start_day_index, end_day_index, X_list)
	if strategy_model is None:
		print("Training failed.")
		return total_profit

	print("Finished, total_profit:{}".format(total_profit))
	strategy_features = strategy_model.to_list()
	optimize_result.insert_result(X_list, strategy_features + [total_profit])
	optimize_result.save(cache_filename)
	return total_profit

def train(stock_name, start_day_index, end_day_index, X_list):
    if len(X_list.shape)==2:
        X_list = X_list[0]

    n_components = int(X_list[0])
    input_data, values, timestamps, prices = hmm_prepare_data(stock_name, start_day_index, 
    	end_day_index, X_list)
    
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
        return np.array(0).reshape((1,1)), None, None, None, None
    
    print(np.unique(states, return_counts=True))
    values = values.flatten()
    states_values = np.stack((states, values), axis=1)

    df_s_v = pd.DataFrame(states_values, columns=['state','value']).fillna(0)
    value_table = df_s_v.groupby(['state']).mean()

    df_s_v['avg_values'] = df_s_v['state'].apply(lambda x: value_table.iloc[x])
    shape = timestamps.shape
    strategy_data_input = np.stack((timestamps, df_s_v['avg_values'].to_numpy().reshape(shape), prices), axis=2)

    trade_strategy = TradeStrategyFactory(slippage=0.00015)
    
    strategy_data_input_no_central = remove_centralized(strategy_data_input)
    strategy_model = trade_strategy.create_trade_strategies(strategy_data_input_no_central, iter=1)

    total_profit, profit_daily, results = strategy_model.get_profit(strategy_data_input_no_central)
        
    return total_profit, remodel, strategy_model, scaler, value_table


if len(sys.argv) < 4:
	print("usage: python3 search-models.py stock_name, start_day_index \
		end_day_index")
	sys.exit()

stock_name = sys.argv[1]
stock_index = get_stock_id_by_name(stock_name)
start_day_index = int(sys.argv[2])
end_day_index = int(sys.argv[3])



opt_func = partial(opt_func_base, stock_name, start_day_index, end_day_index)

mixed_domain = [{'name': 'n_components', 'type': 'discrete', 'domain': tuple(range(4, 24, 1))},
      {'name': 'ema', 'type': 'discrete', 'domain': (1,10,20)},
      {'name': 'beta', 'type': 'discrete', 'domain': (99,98)},
      {'name': 'use_volume', 'type': 'discrete', 'domain': (0,1)},
      {'name': 'ref_stock_id', 'type': 'discrete', 'domain': (-1,992,3524,139301,160271)},
      {'name': 'time_format', 'type': 'discrete', 'domain': (-1,0,1)},
     ]

opt_handler = GPyOpt.methods.BayesianOptimization(f=opt_func,  # Objective function       
                             domain=mixed_domain,           # Box-constraints of the problem
                             initial_design_numdata = 30,   # Number data initial design
                             acquisition_type='EI',        # Expected Improvement
                             exact_feval = True, 
                             maximize = True)           # True evaluations, no sample nois
opt_handler.run_optimization(max_iter=300, eps=0)


