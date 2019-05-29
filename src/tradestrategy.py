from functools import partial
import numpy as np
import GPy
import GPyOpt
import os
import pickle
from optimizeresult import OptimizeResult
from util import remove_centralized


class TradeStrategyFactory:
    mixed_domain = [{'name': 'buy_threshold', 'type': 'discrete', 'domain': tuple(np.around(np.arange(0.0, 0.005,0.0001),4))},
                 {'name': 'sell_threshold', 'type': 'discrete', 'domain': tuple(np.around(np.arange(-0.005, 0.0, 0.0001),4))},
                 {'name': 'stop_loss', 'type': 'discrete', 'domain': tuple(np.around(np.arange(-0.01,-0.003, 0.001),3))},
                 {'name': 'stop_gain', 'type': 'discrete', 'domain': tuple(np.around(np.arange(0.002, 0.02,0.001),3))},
         ]
    def __init__(self, cache_file=None,  n_max_trades_per_day=4, slippage=0.00015, courtage=0):
        self.n_max_trades_per_day = n_max_trades_per_day
        self.slippage = slippage
        self.courtage = courtage
        self.trade_strategy = None

        # load the initial data file
        self.optimize_result = OptimizeResult(result_column_index=-1)
        if cache_file is not None:
            self.cache_file = cache_file
            self.optimize_result.load(cache_file)
        return

    def create_from_file(self, filename, n_number):
        optimize_result = OptimizeResult(result_column_index=-1)
        optimize_result.load(filename)
        data = optimize_result.get_best_results(n_number)
        trade_strategy_list = []

        for i in range(n_number):
            X_list = data[i,:4]
            trade_strategy_list.append(TradeStrategy(X_list, self.n_max_trades_per_day, 
                self.slippage, self.courtage))

        return trade_strategy_list

    def create_trade_strategies(self, data, iter, max_iter=200):
        assert(data.shape[1]==504)
        self.data = data
        init_numdata = int(max_iter / 4)
        trade_strategy_list = []
        for i in range(iter):
            print("Searching Strategies, Run: {}".format(i))
            self.n_iter = 0
            self.trade_strategy = None
            self.max_profit = -1
            myBopt = GPyOpt.methods.BayesianOptimization(self.get_profit,  # Objective function       
                                                 domain=self.mixed_domain,          # Box-constraints of the problem
                                                 initial_design_numdata = init_numdata,   # Number data initial design
                                                 acquisition_type='EI',        # Expected Improvement
                                                 exact_feval = True,
                                                 maximize = True)           # True evaluations, no sample noise

            myBopt.run_optimization(max_iter, eps=0)

            trade_strategy_list.append(self.trade_strategy)

        return trade_strategy_list



    def get_profit(self, X_list):
        assert(len(X_list)==1)
        X_list = X_list[0]
        buy_threshold = X_list[0]
        sell_threshold = X_list[1]
        stop_loss = X_list[2]
        stop_gain = X_list[3]

        self.n_iter += 1
        cached_result, index = self.optimize_result.find_result(X_list)
        trade_strategy = TradeStrategy(X_list, self.n_max_trades_per_day, 
            self.slippage, self.courtage)
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
            self.optimize_result.save(self.cache_file)

        return avg_daily_profit.reshape((1,1))

def print_verbose_func(verbose, msg):
    if verbose == True:
        print(msg)


class TradeStrategy:
    def __init__(self, X_list, n_max_trades_per_day, slippage, courtage):
        self.buy_threshold = X_list[0]
        self.sell_threshold = X_list[1]
        self.stop_loss = X_list[2]
        self.stop_gain = X_list[3]
        self.slippage = slippage
        self.courtage = courtage
        self.n_max_trades_per_day = n_max_trades_per_day
        return

    def get_parameter_str(self):
        s = "buy_threshold:{} sell_threshold:{} stop_loss:{} \
            stop_gain:{}".format(self.buy_threshold,
                                  self.sell_threshold,
                                  self.stop_loss,
                                  self.stop_gain)
        return s

    def to_list(self):
        return [self.buy_threshold, self.sell_threshold, self.stop_loss, self.stop_gain]


    def get_profit(self,  input_data, verbose=False):
        X_list = self.to_list()
        tot_profit, n_tot_trades, daily_profit_list, change_rate = self.run_test_core(X_list, 
                                                                                input_data, 
                                                                                verbose)
        return tot_profit, daily_profit_list, change_rate


    def run_test_core(self, X_list, input_data, verbose=False):
        print_verbose = partial(print_verbose_func, verbose)
        buy_threshold = X_list[0]
        sell_threshold = X_list[1]
        stop_loss = X_list[2]
        stop_gain = X_list[3]

        tot_profit = 1
        tot_stock_profit = 1
        buy_step = None
        n_max_trades = self.n_max_trades_per_day
        cost = self.slippage/2 + self.courtage
        n_tot_trades = 0
        # to prepare the result data
        shape = input_data.shape
        price_data = input_data[:,:,2]
        reshaped_price = price_data.reshape((shape[0]*shape[1]))
        
        stock_change_rate = np.diff(reshaped_price) / reshaped_price[:-1]
        stock_change_rate = np.concatenate(([0], stock_change_rate)).reshape((shape[0],shape[1]))
        
        asset_change_rate = np.zeros((stock_change_rate.shape))
        
        
        daily_profit_list = []
        
        for day_idx in range(len(input_data)):
            print_verbose("starting day {}".format(day_idx))
            n_trades = 0
            daily_profit = 1
            trade_profit = 1
            state = 0
            daily_data = input_data[day_idx]
            hold_steps = 0

            for step in range(len(daily_data)):


                time = daily_data[step][0]
                value = daily_data[step][1]
                price = daily_data[step][2]
                change_rate = stock_change_rate[day_idx][step]
                if state == 0 and time.time().hour >= 9 and \
                    n_trades < n_max_trades and \
                    step < len(daily_data)-5 and \
                    value > buy_threshold:
                        state = 1
                        asset_change_rate[day_idx][step] = -cost
                        tot_profit *= (1-cost)
                        daily_profit *= (1-cost)
                        trade_profit *= (1-cost)
                        print_verbose("buy at step: {} price:{}".format(step, price))
                elif state == 1:
                    if value < sell_threshold  or \
                        step == len(daily_data)-1 or \
                        trade_profit-1 < stop_loss or \
                        trade_profit-1 > stop_gain:
                        # don't do more trade today!
                        if trade_profit-1 < stop_loss:
                            print_verbose("stop loss stop trading!")
                            n_trades = n_max_trades

                        change_rate = (1+change_rate)*(1-cost)-1 
                        tot_profit *= (1 + change_rate)
                        daily_profit *= (1 + change_rate)
                        state = 0
                        n_trades += 1
                        print_verbose("sell at step: {} price:{} trade_profit:{} hold_steps:{}".format(step, 
                            price, trade_profit, hold_steps))

                        trade_profit = 1
                        asset_change_rate[day_idx][step] = change_rate
                        hold_steps = 0
                        
                    else:
                        tot_profit *= (1+change_rate)
                        daily_profit *= (1+change_rate)
                        trade_profit *= (1+change_rate)
                        asset_change_rate[day_idx][step] = change_rate
                        hold_steps += 1
            print_verbose("finished day {}, daily profit:{}".format(day_idx,daily_profit))
            daily_profit_list.append(daily_profit - 1)
            n_tot_trades += n_trades
        
        tot_profit -= 1
        change_rate = np.stack((stock_change_rate, asset_change_rate), axis=2)
        return tot_profit, n_tot_trades, daily_profit_list, change_rate
    
    def get_save_filename(self, path):
        return os.path.join(path, 'strategy_desc.pkl')

if __name__ == '__main__':
    # trade_strategy_factory = TradeStrategyFactory()
    # strategy_list = trade_strategy_factory.create_from_file("test.txt", 5)
    # assert(len(strategy_list)==5)

    data = np.load("../data-analytics/npy_files/ema20_beta99_5.npy", allow_pickle=True)
    input_data = data[:60,:,[-2,-3,-1]]
    input_data = remove_centralized(input_data)
    print(input_data.shape)
    trade_strategy_factory = TradeStrategyFactory(input_data, cache_file="strategy_cache.txt")
    strategy_list = trade_strategy_factory.create_trade_strategies(iter=5, max_iter=50)
    assert(len(strategy_list)==5)
    

