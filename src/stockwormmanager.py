import numpy as np
from pathlib import Path
import pandas as pd
import GPy
import GPyOpt
import uuid
import os.path
import os
from datamanipulator import DataManipulator
from statefullstmmodel import StatefulLstmModel
from functools import partial
from tradestrategy import TradeStrategyFactory
from stockworm import StockWorm
from optimizeresult import OptimizeResult
from util import *

NUM_STRATEGIES=10

class StockWormManager:
    mixed_domain = [{'name': 'n_neurons', 'type': 'discrete', 'domain': tuple(range(20,160,20))},
      {'name': 'learning_rate', 'type': 'discrete', 'domain': (0.001,0.002,0.003,0.004)},
      {'name': 'num_layers', 'type': 'discrete', 'domain': (1,2,3,4,5,6,7,8)},
      {'name': 'rnn_type', 'type': 'discrete', 'domain': (0,1,2)},
      {'name': 'learning_period', 'type': 'discrete', 'domain': (20,30,40,50)},
      {'name': 'prediction_period', 'type': 'discrete', 'domain': (5,10,20)},
      {'name': 'n_repeats', 'type': 'discrete', 'domain': (1,3,5,10,20,30,40)},
      {'name': 'beta', 'type': 'discrete', 'domain': (99,98)},
      {'name': 'ema', 'type': 'discrete', 'domain': (1,10,20)},
      {'name': 'time_format', 'type': 'discrete', 'domain': (0,1,2)}, #1 for stepofday, 2 for stepofweek
      {'name': 'volume_input', 'type': 'discrete', 'domain': (0,1)},
      {'name': 'use_centralized_bid', 'type': 'discrete', 'domain': (0,1)},
      {'name': 'split_daily_data', 'type': 'discrete', 'domain': (0,1)},
      {'name': 'is_stateful', 'type': 'discrete', 'domain': (0,1)},
      {'name': 'ref_stock_id', 'type': 'discrete', 'domain': (-1,992,3524,139301,160271)},
     ]

    mixed_domain_test = [{'name': 'n_neurons', 'type': 'discrete', 'domain': tuple(range(20,160,20))},
      {'name': 'learning_rate', 'type': 'discrete', 'domain': (0.001,0.002,0.003,0.004)},
      {'name': 'num_layers', 'type': 'discrete', 'domain': (1,2,3,4)},
      {'name': 'rnn_type', 'type': 'discrete', 'domain': (0,1,2)},
      {'name': 'learning_period', 'type': 'discrete', 'domain': (20,)},
      {'name': 'prediction_period', 'type': 'discrete', 'domain': (10,)},
      {'name': 'n_repeats', 'type': 'discrete', 'domain': (1,)},
      {'name': 'beta', 'type': 'discrete', 'domain': (99,)},
      {'name': 'ema', 'type': 'discrete', 'domain': (20,)},
      {'name': 'time_format', 'type': 'discrete', 'domain': (0,1,2)}, #1 for stepofday, 2 for stepofweek
      {'name': 'volume_input', 'type': 'discrete', 'domain': (0,1)},
      {'name': 'use_centralized_bid', 'type': 'discrete', 'domain': (0,1)},
      {'name': 'split_daily_data', 'type': 'discrete', 'domain': (0,1)},
      {'name': 'is_stateful', 'type': 'discrete', 'domain': (0,1)},
      {'name': 'ref_stock_id', 'type': 'discrete', 'domain': (-1,992,3524,139301,160271)},
     ]

    mixed_domain_future = [{'name': 'n_neurons', 'type': 'discrete', 'domain': tuple(range(20,160,20))},
      {'name': 'learning_rate', 'type': 'discrete', 'domain': (0.001,0.002,0.003,0.004)},
      {'name': 'num_layers', 'type': 'discrete', 'domain': (1,2,3,4,5,6,7,8)},
      {'name': 'rnn_type', 'type': 'discrete', 'domain': (0,1,2)},
      {'name': 'learning_period', 'type': 'discrete', 'domain': (20,30,40,50)},
      {'name': 'prediction_period', 'type': 'discrete', 'domain': (5,10,20)},
      {'name': 'n_repeats', 'type': 'discrete', 'domain': (1,3,5,10,20,30)},
      {'name': 'beta', 'type': 'discrete', 'domain': (99,98)},
      {'name': 'ema', 'type': 'discrete', 'domain': (1,10,20)},
      {'name': 'time_format', 'type': 'discrete', 'domain': (0,1,2)}, #1 for stepofday, 2 for stepofweek
      {'name': 'volume_input', 'type': 'discrete', 'domain': (0,1)},
      {'name': 'use_centralized_bid', 'type': 'discrete', 'domain': (1,)},
      {'name': 'split_daily_data', 'type': 'discrete', 'domain': (0,)},
      {'name': 'is_stateful', 'type': 'discrete', 'domain': (0,1)},
      {'name': 'ref_stock_id', 'type': 'discrete', 'domain': (-1,800005)},
     ]

    def __init__(self, stock_name, stock_data_path, npy_files_path, is_future=False, slippage=0):
        self.stock_name = stock_name

        self.stock_id = get_stock_id_by_name(stock_name)
        if self.stock_id == None:
          print("cannot find stock id with name: {}".format(stock_name))
          os.exit()
        

        self.stock_data_path = stock_data_path
        self.npy_files_path = npy_files_path
        self.worm_list = []
        self.is_future = is_future
        self.slippage = slippage

    def search_worms(self, start_day_index, end_day_index, 
      max_iter=300, is_test=False):
        if is_test == True:
            mixed_domain = self.mixed_domain_test
        else:
            mixed_domain = self.mixed_domain

        if self.is_future == True:
          mixed_domain = self.mixed_domain_future

        self.optimize_result = OptimizeResult()
        stock_worm_cache_file = self.get_stockworm_cache_file(start_day_index, end_day_index)
        create_if_not_exist(os.path.dirname(stock_worm_cache_file))


        if os.path.isfile(stock_worm_cache_file):
            self.optimize_result.load(stock_worm_cache_file)
            print("stock worm cache loaded, size={}".format(self.optimize_result.get_size()))
        else:
            print("cannot find file cache:{}, will create new cache.".format(stock_worm_cache_file))

        self.stock_worm_cache_file = stock_worm_cache_file
        opt_func = partial(self.opt_func, start_day_index, end_day_index)
        opt_handler = GPyOpt.methods.BayesianOptimization(f=opt_func,  # Objective function       
                                     domain=mixed_domain,           # Box-constraints of the problem
                                     initial_design_numdata = 30,   # Number data initial design
                                     acquisition_type='EI',        # Expected Improvement
                                     exact_feval = True, 
                                     maximize = True)           # True evaluations, no sample noise
        opt_handler.run_optimization(max_iter, eps=0)

    def opt_func(self, start_day, end_day, X_list):
        assert(len(X_list) == 1)
        features = X_list[0]

        ref_stock_id = features[14]
        if ref_stock_id == self.stock_id:
          ref_stock_id = -1

        features[14] = ref_stock_id
        
        print("starting test: {}".format(self.get_parameter_str(features)))  
        cached_result, index = self.optimize_result.find_result(features)
        if cached_result is not None:
            print("find from cache. skip...")
        else:
            stock_worm = StockWorm(self.stock_name, self.stock_id, self.npy_files_path, is_future=self.is_future, slippage=self.slippage)
            if stock_worm.validate(features, start_day, end_day) == False:
              print("validate failed for worm: {}".format(self.get_parameter_str(features)))
              return np.array(-1).reshape((1,1))

            total_profit, profit_daily, errors_daily = stock_worm.init(features, 
                start_day, end_day)

            n_days = len(profit_daily)
            profit_mean = np.mean(profit_daily)
            error_mean = np.mean(errors_daily)

            # to do insert strategy features.
            strategy_features = list(stock_worm.get_strategy_features())
            self.optimize_result.insert_result(features, strategy_features + 
              [total_profit, n_days, error_mean, profit_mean])


            print("result saved to: {}".format(self.stock_worm_cache_file))
            self.optimize_result.save(self.stock_worm_cache_file)

        print("total_profit:{} in {} days, profit_mean:{} error:{} parameters:{}".format(total_profit, 
                                    n_days,
                                    profit_mean,
                                    error_mean,
                                    self.get_parameter_str(features)))

        return np.array(profit_mean).reshape((1,1))

    def get_swarm_path(self, start_day_index, end_day_index):
        return os.path.join(self.stock_data_path, 
            "{}_{}".format(self.stock_name, self.stock_id),
            "{}-{}".format(start_day_index, end_day_index))

    def get_stockworm_cache_file(self, start_day_index, end_day_index):
        swarm_path = self.get_swarm_path(start_day_index, end_day_index)
        return os.path.join(swarm_path, "stockworm_cache.txt")


    def update_worms_from_cache(self, n_number, start_day_index, end_day_index):
      optimize_result = OptimizeResult()
      stockworm_cache_file = self.get_stockworm_cache_file(start_day_index, end_day_index)
      optimize_result.load(stockworm_cache_file)
      top_worms = optimize_result.get_best_results(n_number)

      assert(len(top_worms) == n_number)
      for i in range(n_number):
        features = top_worms[i, :15]
        strategy_features = top_worms[i, 15:21]
        md5 = top_worms[i, -1]
        model_save_path = self.get_model_save_path(start_day_index, end_day_index, md5)
        new_worm = StockWorm(self.stock_name, self.stock_id, self.npy_files_path, 
          model_save_path, is_future=self.is_future, slippage=self.slippage)

        if os.path.isdir(model_save_path) and new_worm.load() == True:
            pass
        else:
          total_profit, profit_daily, errors_daily = new_worm.init(features, 
            start_day_index, 
            end_day_index,
            strategy_features=strategy_features)
          new_worm.save()
          print("training finished for model {}, total_profit:{}".format(i, total_profit))
        
        new_worm.report()

        testing_total_profit, testing_profit_daily, n_data_appended = new_worm.test()
        if n_data_appended > 0:
          print("testing finished for model {}, total_profit:{} in {} days, new data for {} days appended".format(i, 
              testing_total_profit, len(testing_profit_daily),
              n_data_appended))
          new_worm.save()

        new_worm.report()
        self.worm_list.append(new_worm)

    def report(self):
      assert(self.worm_list is not None)
      for i in range(len(self.worm_list)):
        print("Report for Worm No.{}".format(i+1))
        self.worm_list[i].report()

    def get_model_save_path(self, start_day_index, end_day_index, md5=None):
      swarm_path = self.get_swarm_path(start_day_index, end_day_index)
      if md5 is None:
        model_save_path = os.path.join(swarm_path, "models")
      else:
        model_save_path = os.path.join(swarm_path, "models", md5)
      return model_save_path

    def load(self):
      model_save_path = self.get_model_save_path(0, 60)
      directories = [os.path.join(model_save_path, f) for f in os.listdir(model_save_path) if os.path.isdir(os.path.join(model_save_path, f))]
      self.worm_list = []
      for d in directories:
        new_worm = StockWorm(self.stock_id, self.npy_files_path, d)
        new_worm.load()
        self.worm_list.append(new_worm)

    def get_worm(self, n):
      assert(len(self.worm_list) > n)
      return self.worm_list[n]

    def test(self):
      profit_avg_list = []
      profit_list = []
      for worm in self.worm_list:
        training_total_profit, \
                training_daily_profit, testing_total_profit, \
                testing_daily_profit = worm.get_historic_metrics()

        testing_len = len(testing_daily_profit)
        traning_len = len(training_daily_profit)
        all_profit = np.concatenate((training_daily_profit, testing_daily_profit), axis=0)
        all_profit_cumsum = np.cumsum(all_profit)
        profit_avg = all_profit_cumsum/(np.arange(len(all_profit_cumsum))+1)
        profit_avg_list.append(profit_avg[traning_len-1:])
        profit_list.append(testing_daily_profit)

        assert(len(profit_avg[traning_len-1:]) == len(testing_daily_profit) + 1)

      profit_avg_arr = np.array(profit_avg_list)
      profit_arr = np.array(profit_list)

      # find the argmax from profit_avg_arr
      argmax_arr = np.argmax(profit_avg_arr, axis=0)
      overall_profit = profit_arr[argmax_arr[:-1], np.arange(len(profit_arr[0]))]

      
      print(argmax_arr[:-1])
      print(overall_profit)
      print("OVERALL RESULTS:")
      print(np.prod(overall_profit+1)-1)

    def plot(self):
      
      pass

        
    def get_parameter_str(self, X):
        parameter_str = ""
        for i in range(len(self.mixed_domain)):
            parameter_str += self.mixed_domain[i]["name"]
            parameter_str += ':'
            parameter_str += str(X[i])
            parameter_str += ','
        return parameter_str    


if __name__ == '__main__':
    stock_worm_manager = StockWormManager('Nordea', 5, '../stock-data/', '../preprocessed-data/')
    stock_worm_manager.load()
    stock_worm_manager.report()
    #total_profit, overall_profit_daily = stock_worm_manager.test()
    #print("Test finished, total profit: %f" % total_profit)

    #stock_worm_manager.plot()

    #stock_worm_manager.search_worms("../models/190128-190423/strategy_cache.txt", "../models/190128-190423/worm_cache.txt", 0, 60, is_test=False)
    #stock_worm_manager.create_from_file()
