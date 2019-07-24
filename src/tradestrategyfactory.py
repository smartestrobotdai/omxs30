from functools import partial
import numpy as np
import GPy
import GPyOpt
import os
import pandas as pd
from optimizeresult import OptimizeResult
from tradestrategy import TradeStrategy
from tradestrategyfuture import TradeStrategyFuture

class TradeStrategyFactory:
    mixed_domain = [{'name': 'buy_threshold', 'type': 'continuous', 'domain': (0, 0.005)},
                 {'name': 'sell_threshold', 'type': 'continuous', 'domain': (-0.005, 0)},
                 {'name': 'stop_loss', 'type': 'continuous', 'domain': (-0.02, -0.003)},
                 {'name': 'stop_gain', 'type': 'continuous', 'domain': (0.003, 0.02)},
                 {'name': 'skip_at_beginning', 'type': 'discrete', 'domain': (0,5, 10, 20)},
                 {'name': 'value_ma', 'type': 'discrete', 'domain': (1,3,5,10,20)}
         ]
    def __init__(self, cache_file=None,  n_max_trades_per_day=4, slippage=0, courtage=0, is_future=False):
        self.n_max_trades_per_day = n_max_trades_per_day
        self.slippage = slippage
        self.courtage = courtage
        self.is_future = is_future
        self.trade_strategy = None

        # load the initial data file
        self.optimize_result = OptimizeResult()
        self.cache_file = cache_file
        if cache_file is not None:
            self.optimize_result.load(cache_file)
        return

    def create_from_file(self, filename, n_number):
        optimize_result = OptimizeResult(result_column_index=-1)
        optimize_result.load(filename)
        data = optimize_result.get_best_results(n_number)
        trade_strategy_list = []
        if self.is_future:
            classTradeStrategy = TradeStrategyFuture
        else:
            classTradeStrategy = TradeStrategy

        for i in range(n_number):
            X_list = data[i,:4]
            trade_strategy_list.append(classTradeStrategy(X_list, self.n_max_trades_per_day, 
                self.slippage, self.courtage))

        return trade_strategy_list

    def create_trade_strategies(self, data, iter, max_iter=100):
        #assert(data.shape[1]==504)
        print(data.shape)
        print("strategy data input")
        print(data)
        self.data = data
        init_numdata = int(max_iter / 4)
        trade_strategy_list = []
        self.max_profit = -1
        self.trade_strategy = None
        for i in range(iter):
            print("Searching Strategies, Run: {}".format(i))
            self.n_iter = 0
            myBopt = GPyOpt.methods.BayesianOptimization(self.get_profit,  # Objective function       
                                                 domain=self.mixed_domain,          # Box-constraints of the problem
                                                 initial_design_numdata = init_numdata,   # Number data initial design
                                                 acquisition_type='EI',        # Expected Improvement
                                                 exact_feval = True,
                                                 maximize = True)           # True evaluations, no sample noise

            myBopt.run_optimization(max_iter, eps=0)


        return self.trade_strategy



    def get_profit(self, X_list):
        assert(len(X_list)==1)
        if self.is_future:
            classTradeStrategy = TradeStrategyFuture
        else:
            classTradeStrategy = TradeStrategy
            
        X_list = X_list[0]

        self.n_iter += 1
        cached_result, index = self.optimize_result.find_result(X_list)


        trade_strategy = classTradeStrategy(X_list, n_max_trades_per_day=self.n_max_trades_per_day, 
            slippage=self.slippage, 
            courtage=self.courtage)

        if cached_result is not None:
            print("find cached result: {} for {}".format(cached_result, 
                trade_strategy.get_parameter_str()))
            avg_daily_profit = cached_result[0]
        else:
            total_profit, daily_profit_list,_ =  trade_strategy.get_profit(self.data)
            avg_daily_profit = np.mean(daily_profit_list)
            self.optimize_result.insert_result(X_list, avg_daily_profit)

        if avg_daily_profit > self.max_profit:
            print("find new record: {}, {}".format(avg_daily_profit, 
                    trade_strategy.get_parameter_str()))

            self.max_profit = avg_daily_profit
            self.trade_strategy = trade_strategy
 
        if self.n_iter % 10 == 0:
            print("iteration: {}, cachesize={}, avg_daily_profit:{}".format(self.n_iter, 
                self.optimize_result.get_size(),
                avg_daily_profit))
            if self.cache_file != None:
                self.optimize_result.save(self.cache_file)

        return avg_daily_profit.reshape((1,1))
