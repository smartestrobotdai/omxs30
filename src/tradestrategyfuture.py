from functools import partial
import numpy as np
import GPy
import GPyOpt
import os
import pickle
import pandas as pd
from optimizeresult import OptimizeResult
from util import remove_centralized, print_verbose_func

class TradeStrategyFuture:
    def __init__(self, X_list, n_max_trades_per_day=4, slippage=0, courtage=0, is_future=False):
        self.buy_threshold = X_list[0]
        self.sell_threshold = X_list[1]
        self.stop_loss = X_list[2]
        self.stop_gain = X_list[3]
        self.skip_at_beginning = X_list[4]
        self.value_ma = X_list[5]
        self.slippage = slippage
        self.courtage = courtage
        self.n_max_trades_per_day = n_max_trades_per_day
        self.is_future = is_future
        return

    def get_parameter_str(self):
        s = "buy_threshold:{} sell_threshold:{} stop_loss:{} \
            stop_gain:{}, skip_at_beginning: {}, value_ma: {}".format(self.buy_threshold,
                                  self.sell_threshold,
                                  self.stop_loss,
                                  self.stop_gain,
                                  self.skip_at_beginning,
                                  self.value_ma)
        return s

    def to_list(self):
        return [self.buy_threshold, self.sell_threshold, 
        self.stop_loss, self.stop_gain, self.skip_at_beginning, 
        self.value_ma]

    def get_features(self):
        return (self.buy_threshold, self.sell_threshold, self.stop_loss, 
            self.stop_gain, self.skip_at_beginning, self.value_ma)

    def get_profit(self,  input_data, verbose=False):
        # the centralized part must be removed
        #assert(input_data.shape[1] == 504)
        X_list = self.to_list()
        tot_profit, n_tot_trades, daily_profit_list, results = self.run_test_core(X_list, 
                                                                                input_data, 
                                                                                verbose)
        return tot_profit, daily_profit_list, results

    def get_length(self, daily_data):
        for i in range(len(daily_data)):
            if np.isnan(daily_data[i,2]):
                return i
        return len(daily_data)

    def is_hold(self, state):
        return state == -1 or state == 1

    def need_to_sell(self, value, state):
        if state == -1 and value > 0:
            return True
        if state == 1 and value < 0:
            return False

    def get_action(self, value, buy_threshold, sell_threshold):
        if value > buy_threshold:
            return 1
        elif value < sell_threshold:
            return 2
        else:
            return 0

    def run_test_core(self, X_list, input_data, verbose=False):
        print_verbose = partial(print_verbose_func, verbose)
        buy_threshold = X_list[0]
        sell_threshold = X_list[1]
        stop_loss = X_list[2]
        stop_gain = X_list[3]
        skip_at_beginning = int(X_list[4])
        value_ma = int(X_list[5])

        tot_profit = 1
        tot_stock_profit = 1
        buy_step = None
        n_max_trades = self.n_max_trades_per_day

        n_tot_trades = 0
        # to prepare the result data
        shape = input_data.shape
        price_data = input_data[:,:,2]
        reshaped_price = price_data.reshape((shape[0]*shape[1]))
        
        stock_change_rate = np.diff(reshaped_price) / reshaped_price[:-1]
        stock_change_rate = np.concatenate(([0], stock_change_rate)).reshape((shape[0],shape[1]))
        
        asset_change_rate = np.zeros((stock_change_rate.shape))
        actions = np.zeros((stock_change_rate.shape))
        
        daily_profit_list = []
        
        for day_idx in range(len(input_data)):
            n_trades = 0
            daily_profit = 1
            trade_profit = 1
            state = 0
            daily_data = input_data[day_idx]
            hold_steps = 0
            hit_stop = False

            length = self.get_length(daily_data)
            print_verbose("starting sequence {}, length: {}".format(day_idx, length))
            daily_data = daily_data[:length]

            # do ma on values
            if value_ma != 1:
                values = pd.Series(daily_data[:,1])
                values_ma = values.ewm(span=value_ma, adjust=False).mean()
                values_ma = values_ma.values
            else:
                values_ma = daily_data[:,1]

            for step in range(length):
                time = daily_data[step][0]
                value = values_ma[step]
                price = daily_data[step][2]

                cost = self.slippage / price + self.courtage
                if self.is_hold(state) == False and n_trades < n_max_trades and \
                    step < len(daily_data)-5 and \
                    step > skip_at_beginning and \
                    hit_stop == False:
                        action = self.get_action(value, buy_threshold, sell_threshold)
                        if action != 0:
                            actions[day_idx][step] = action  # do long
                            if action == 1:  # long
                                state = 1
                                print_verbose("do long at step: {} value:{} price:{} cost:{}".format(step, value, price, cost))
                            elif action == 2:  # short
                                state = -1
                                print_verbose("do short at step: {} value:{} price:{} cost:{}".format(step, value, price, cost))

                            asset_change_rate[day_idx][step] = -cost
                            tot_profit *= (1-cost)
                            daily_profit *= (1-cost)
                            trade_profit *= (1-cost)
                            print_verbose("trade_profit:{}".format(trade_profit))

                elif self.is_hold(state):
                    change_rate = stock_change_rate[day_idx][step]
                    if state == -1:
                        change_rate = -change_rate
                    trade_profit_no_stop = trade_profit*(1+change_rate)
                    if self.need_to_sell(value, state)  or \
                        step == length-1 or \
                        trade_profit_no_stop-1 < stop_loss or \
                        trade_profit_no_stop-1 > stop_gain:
                        # don't do more trade today!
                        if trade_profit_no_stop-1 < stop_loss:
                            print_verbose("stop loss stop trading! trade_profit: {}, \
                                change_rate: {} stop_loss: {} price:{}".format(trade_profit, 
                                change_rate, stop_loss, price))

                            hit_stop = True
                            assert(stop_loss < trade_profit-1)
                            change_rate = (1+max(stop_loss-(trade_profit-1),change_rate))*(1-cost)-1

                        elif trade_profit_no_stop-1 > stop_gain:
                            print_verbose("stop gain stop trading!")
                            hit_stop = True
                            assert(stop_gain > trade_profit-1)
                            change_rate = (1+min(stop_gain-(trade_profit-1),change_rate))*(1-cost)-1                            
                        else:
                            change_rate = (1+change_rate)*(1-cost)-1 

                        actions[day_idx][step] = -1  # sell
                        
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
        results = np.stack((stock_change_rate, asset_change_rate, actions), axis=2)
        return tot_profit, n_tot_trades, np.array(daily_profit_list), results

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
    

