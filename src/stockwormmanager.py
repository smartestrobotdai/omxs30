import numpy as np
from pathlib import Path
import pandas as pd
import GPy
import GPyOpt
import uuid
import os.path
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
      {'name': 'learning_period', 'type': 'discrete', 'domain': (20,30,40)},
      {'name': 'prediction_period', 'type': 'discrete', 'domain': (2,5,10,20)},
      {'name': 'n_repeats', 'type': 'discrete', 'domain': (1,3,5,10,20,30,40)},
      {'name': 'beta', 'type': 'discrete', 'domain': (99,)},
      {'name': 'ema', 'type': 'discrete', 'domain': (20,)},
      {'name': 'time_format', 'type': 'discrete', 'domain': (0,1,2)}, #1 for stepofday, 2 for stepofweek
      {'name': 'volume_input', 'type': 'discrete', 'domain': (0,1)},
      {'name': 'use_centralized_bid', 'type': 'discrete', 'domain': (0,1)},
      {'name': 'split_daily_data', 'type': 'discrete', 'domain': (1,)}
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
      {'name': 'split_daily_data', 'type': 'discrete', 'domain': (0,1)}
     ]

    def __init__(self, stock_name, stock_index, stock_data_path, npy_files_path):
        self.stock_name = stock_name
        self.stock_index = stock_index
        self.stock_data_path = stock_data_path
        self.npy_files_path = npy_files_path
        self.worm_list = []

    def search_worms(self, start_day_index, end_day_index, 
        max_iter=300, is_test=False, search_strategy=False):
        if is_test == True:
            mixed_domain = self.mixed_domain_test
        else:
            mixed_domain = self.mixed_domain

        self.optimize_result = OptimizeResult(result_column_index=15)
        stock_worm_cache_file = self.get_stockworm_cache_file(start_day_index, end_day_index)

        if os.path.isfile(stock_worm_cache_file):
            self.optimize_result.load(stock_worm_cache_file)
            print("stock worm cache loaded, size={}".format(self.optimize_result.get_size()))
        else:
            print("cannot find file cache:{}, will create new cache.".format(stock_worm_cache_file))

        self.stock_worm_cache_file = stock_worm_cache_file

        strategy_cache_file = self.get_strategy_cache_file(start_day_index, end_day_index)
        
        trade_strategy_factory = TradeStrategyFactory(cache_file=strategy_cache_file)
        if os.path.isfile(strategy_cache_file) and search_strategy == False:
            print("find strategy_cache:{}, loading...".format(strategy_cache_file))
            strategy_list = trade_strategy_factory.create_from_file(strategy_cache_file, NUM_STRATEGIES)
        else:
          if search_strategy == True:
            print("search_strategy is True, searching strategies again...")
          else:
            print("cannot find strategy cache:{}, generating...".format(strategy_cache_file))

          data = load_strategy_input_data(self.stock_index, start_day_index, end_day_index)
          # the input data: timestamp, value, and price.
          strategy_list = trade_strategy_factory.create_trade_strategies(data, 5)

        opt_func = partial(self.opt_func, strategy_list, start_day_index, end_day_index)

        opt_handler = GPyOpt.methods.BayesianOptimization(f=opt_func,  # Objective function       
                                     domain=mixed_domain,           # Box-constraints of the problem
                                     initial_design_numdata = 30,   # Number data initial design
                                     acquisition_type='EI',        # Expected Improvement
                                     exact_feval = True, 
                                     maximize = True)           # True evaluations, no sample noise
        opt_handler.run_optimization(max_iter, eps=0)

    def opt_func(self, strategy_list, start_day, end_day, X_list):
        assert(len(X_list) == 1)
        features = X_list[0]
        print("starting test: {}".format(self.get_parameter_str(features)))  
        cached_result, index = self.optimize_result.find_result(features)
        if cached_result is not None:
            total_profit = cached_result[0]
            n_days = cached_result[1]
            profit_mean = cached_result[2]
            error_mean = cached_result[3]
            print("find from cache. skip...")
        else:
            model_save_path = self.get_model_save_path(start_day, end_day, features)
            stock_worm = StockWorm(self.stock_index, self.npy_files_path, model_save_path)
            total_profit, profit_daily, errors_daily = stock_worm.init(features, 
                strategy_list, start_day, end_day)


            n_days = len(profit_daily)
            profit_mean = np.mean(profit_daily)
            error_mean = np.mean(errors_daily)

            self.optimize_result.insert_result(features, [total_profit, n_days, profit_mean, error_mean])
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
            "{}_{}".format(self.stock_name, self.stock_index),
            "{}-{}".format(start_day_index, end_day_index))

    def get_stockworm_cache_file(self, start_day_index, end_day_index):
        swarm_path = self.get_swarm_path(start_day_index, end_day_index)
        return os.path.join(swarm_path, "stockworm_cache.txt")

    def get_strategy_cache_file(self, start_day_index, end_day_index):
        swarm_path = self.get_swarm_path(start_day_index, end_day_index)
        return os.path.join(swarm_path, "strategy_cache.txt")

    def update_worms_from_cache(self, n_number, start_day_index, end_day_index):
        optimize_result = OptimizeResult(result_column_index=15)
        stockworm_cache_file = self.get_stockworm_cache_file(start_day_index, end_day_index)
        optimize_result.load(stockworm_cache_file)
        top_worms = optimize_result.get_best_results(n_number)

        trade_strategy_factory = TradeStrategyFactory()
        strategy_cache_file = self.get_strategy_cache_file(start_day_index, end_day_index)
        strategy_list = trade_strategy_factory.create_from_file(strategy_cache_file, NUM_STRATEGIES)

        assert(len(top_worms) == n_number)
        for i in range(n_number):
          features = top_worms[i, :13]
          model_save_path = self.get_model_save_path(start_day_index, end_day_index, features)
          new_worm = StockWorm(self.stock_index, self.npy_files_path, model_save_path)
          if os.path.isdir(model_save_path) and new_worm.load() == True:
              pass
          else:
              total_profit, profit_daily, errors_daily = new_worm.init(features, strategy_list, start_day_index, end_day_index)
              new_worm.save()
              print("training finished for model {}, total_profit:{}".format(i, total_profit))
            

          testing_total_profit, testing_profit_daily, n_data_appended = new_worm.test()
          if n_data_appended > 0:
            print("testing finished for model {}, total_profit:{} in {} days, new data for {} days appended".format(i, 
                testing_total_profit, len(testing_profit_daily),
                n_data_appended))
            new_worm.save()

          self.worm_list.append(new_worm)

    def report(self):
      assert(self.worm_list is not None)
      for i in range(len(self.worm_list)):
        print("Report for Worm No.{}".format(i+1))
        self.worm_list[i].report()

    def get_model_save_path(self, start_day_index, end_day_index, features):
      swarm_path = self.get_swarm_path(start_day_index, end_day_index)
      features_str = self.get_parameter_str(features)
      model_save_path = os.path.join(swarm_path, "models", md5(features_str))
      return model_save_path

    def load(self):
      model_save_path = self.get_model_save_path(0, 60)
      directories = [f for f in listdir(model_save_path) if isdir(join(model_save_path, f))]
      self.worm_list = []
      for d in directories:
        new_worm = StockWorm(self.stock_index, self.npy_files_path, d)
        new_worm.load()
        self.worm_list.append(new_worm)

    def test(self):
      profit_sma_list = []
      testing_profit_daily_list = []
      for i in range(len(self.worm_list)):
        print("Report for Worm No.{}".format(i+1))
        worm = self.worm_list[i]
        training_total_profit, training_daily_profit, \
            testing_total_profit, testing_daily_profit = worm.get_historic_metrics()

        all_daily_profit = np.concatenate((training_daily_profit[-20:], testing_daily_profit))
        all_profit_avg = cumsum_avg(all_daily_profit)


        profit_sma = worm.get_profit_sma()
        assert(len(testing_daily_profit)+1 == len(profit_sma))

        profit_sma_list.append(profit_sma)
        testing_profit_daily_list.append(testing_daily_profit)

      profit_sma_array = np.array(profit_sma_list)
      test_profit_daily_array = np.array(testing_profit_daily_list)
      # go through all the days
      print(profit_sma_array.shape)
      print(test_profit_daily_array.shape)

      overall_profit_list = []
      for i in range(len(test_profit_daily_array[0])):

        best_sma_index = np.argmax(profit_sma_array[:,i])
        print("day:{}, best_sma_index:{}, sma:{}".format(i, best_sma_index, profit_sma_array[:,i]))
        overall_profit = test_profit_daily_array[best_sma_index, i]
        overall_profit_list.append(overall_profit)

      overall_profit_daily = np.array(overall_profit_list)
      total_profit = np.prod(overall_profit_daily+1)-1
      return total_profit, overall_profit_daily

    def plot(self):
      assert(len(self.worm_list)!=0)
      plt.subplot(2, 1, 1)
      for i in range(len(self.worm_list)):
        worm = self.worm_list[i]
        daily_data = self.get_daily_data()
        x1 = daily_data[:training_data_length,0]
        y1 = np.cumprod(daily_data[:training_data_length,1])
        z1 = np.cumprod(daily_data[:training_data_length,2])

        
    def get_parameter_str(self, X):
        parameter_str = ""
        for i in range(len(self.mixed_domain)):
            parameter_str += self.mixed_domain[i]["name"]
            parameter_str += ':'
            parameter_str += str(X[i])
            parameter_str += ','
        return parameter_str    


if __name__ == '__main__':
    stock_worm_manager = StockWormManager('Nordel', 5, '../stock-data/')
    stock_worm_manager.load()
    stock_worm_manager.report()
    total_profit, overall_profit_daily = stock_worm_manager.test()
    print("Test finished, total profit: %f" % total_profit)

    #stock_worm_manager.plot()

    #stock_worm_manager.search_worms("../models/190128-190423/strategy_cache.txt", "../models/190128-190423/worm_cache.txt", 0, 60, is_test=False)
    #stock_worm_manager.create_from_file()
