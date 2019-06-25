import numpy as np
from pathlib import Path
import pandas as pd
import uuid
import os.path
import pickle
from datamanipulator import DataManipulator
from statefullstmmodel import StatefulLstmModel
from util import *
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta  
import matplotlib.dates as dates

from ipywidgets import interact
import ipywidgets as widgets


class StockWorm:
    def __init__(self, stock_name, stock_id, input_data_path, save_path):
        self.stock_name = stock_name
        self.stock_id = stock_id 
        self.input_data_path = input_data_path
        self.save_path = save_path
        

        self.last_learning_day_index = None
        self.learning_end_date = None
        self.historic_data = None
        self.data_today = []

    def init(self, features, strategy_model_list, start_day_index, end_day_index):
        n_neurons = int(features[0])
        learning_rate = features[1]
        num_layers = int(features[2])
        rnn_type = int(features[3])
        learning_period = int(features[4])
        prediction_period = int(features[5])
        n_repeats = int(features[6])
        beta = int(features[7])
        ema = int(features[8])
        time_format = int(features[9])
        volume_input = int(features[10])
        use_centralized_bid = int(features[11])
        split_daily_data = int(features[12])

        data_manipulator = DataManipulator(self.stock_name,
                                           self.stock_id,
                                           learning_period,
                                           prediction_period,
                                           beta, ema, 
                                           time_format, 
                                           volume_input, 
                                           use_centralized_bid, 
                                           split_daily_data, 
                                           self.input_data_path)

        data_manipulator.init_scalers(start_day_index, end_day_index)

        model = StatefulLstmModel(n_neurons, learning_rate, num_layers, rnn_type, n_repeats)
        self.data_manipulator = data_manipulator
        self.model = model

        strategy_data_input, real_values, errors_daily = self.test_model_base(start_day_index, end_day_index)

        max_total_profit = -1
        max_profit_daily = None
        best_strategy_model = None
        best_change_rate = None
        assert(len(strategy_model_list)>0)
        for strategy_model in strategy_model_list:
          total_profit, profit_daily, change_rate = strategy_model.get_profit(strategy_data_input)
          if total_profit > max_total_profit:
            max_total_profit = total_profit
            max_profit_daily = profit_daily
            best_strategy_model = strategy_model
            best_change_rate = change_rate

        self.strategy_model = best_strategy_model
        print(strategy_data_input.shape)
        print(real_values.shape)
        self.historic_data = np.concatenate((strategy_data_input, best_change_rate, real_values), axis=2)

        return max_total_profit, max_profit_daily, errors_daily
    
    
    def test(self, end_date=None):
        # find the start_day_index
        assert(self.learning_end_date != None)
        n_learning_days = self.data_manipulator.get_learning_days()
        n_prediction_days = self.data_manipulator.get_prediction_days()
        learning_end_day_index = self.data_manipulator.date_2_day_index(self.learning_end_date)



        start_day_index = learning_end_day_index + 1 - n_learning_days + n_prediction_days
        print("DEBUG: learning_end_day_index: {}, start_day_index:  {}, n_learning_days:{}, n_prediction_days:{}".format(learning_end_day_index,
            start_day_index,
            n_learning_days,
            n_prediction_days))
        if end_date is None:
            end_day_index = self.data_manipulator.get_n_days()
        else:
            end_day_index = self.data_manipulator.date_2_day_index(end_date) + 1

        # the first end_day_index is NOT inclusive. 
        n_data_appended = 0
        if end_day_index != learning_end_day_index + 1:
            strategy_data_input, real_values, errors_daily = self.test_model_base(start_day_index, end_day_index)
            total_profit, profit_daily, change_rate = self.strategy_model.get_profit(strategy_data_input)


            historic_data = np.concatenate((strategy_data_input, change_rate, real_values), axis=2)
            assert(self.historic_data is not None)
            n_data_appended = self.append_historic_data(historic_data)

        
        _,_,testing_total_profit, testing_profit_list = self.get_historic_metrics()
        return  testing_total_profit, testing_profit_list, n_data_appended

    def start_realtime_prediction(self, end_date):
        # do predction until yesterday's close time and change model states.
        self.test(end_date)
        self.last_price = self.data_manipulator.get_last_price(end_date)

    # this method need to be called every minute.
    def test_realtime(self, timestamp, price, volume):
        assert(self.last_price != None)
        self.data_today.append([timestamp, price, volume])
        df = pd.DataFrame(self.data_today,columns=['timestamp', 'last', 'volume'])
        df = preprocessing_daily_data(df, last_close=self.last_price, calculate_values=False)
        df = add_step_columns(df)
        df['value'] = 0
        df_input = df[['step_of_day', 'step_of_week', 'diff_ema_20', 'volume', 'value', 'timestamp', 'last']]
        input_data, timestamp, price = self.data_manipulator.purge_data_realtime(df_input)
        outputs = self.model.predict_realtime(input_data)
        outputs_scaled = self.data_manipulator.inverse_transform_output(outputs)

        output_scaled_daily = self.data_manipulator.seq_data_2_daily_data(outputs_scaled)
        return output_scaled_daily
        
    def func_test_test_realtime(self, date=None, n_steps=5):
        if date == None:
            date = get_current_date()

        df = pd.read_csv('../data/data.csv.gz', compression='gzip', sep=',')
        df = df[df['stock_id'] == self.stock_id]
        df['timestamp'] = pd.to_datetime(df['time_stamp'], format="%Y-%m-%d %H:%M:%S").dt.tz_convert('CET')
        df['date'] = df['timestamp'].apply(timestamp2date)
        df = df[df['date'] == date]

        # find the last date 
        day_index = self.data_manipulator.date_2_day_index(date)
        last_date = self.data_manipulator.day_index_2_date(day_index - 1)

        print("Starting realtime test, today: {}, last_trade_day:{}".format(date, last_date))

        # the data format: daily step, weekly step, diff, volume, value, timestamp, price.
        self.start_realtime_prediction(last_date)


        for i in range(n_steps):
            output_scaled_daily = self.test_realtime(df.iloc[i]['timestamp'], df.iloc[i]['last'], df.iloc[i]['volume'])

        return output_scaled_daily


    def append_historic_data(self, historic_data):
        first_date = timestamp2date(historic_data[0,0,0])
        find_date = False
        for i in range(len(self.historic_data)):
            date = timestamp2date(self.historic_data[i,0,0])
            if date == first_date:
                find_date = True
                break

        if find_date == True:
            origin_last_date = timestamp2date(self.historic_data[-1,0,0])
            last_date = timestamp2date(historic_data[-1,0,0])
            print("overwritting historic_data ended at date: {} from date: {} to date: {} ".format(origin_last_date, first_date, last_date))
            self.historic_data = np.concatenate((self.historic_data[:i], historic_data), axis=0)
            print(self.historic_data.shape)
        else:
            i+=1
            origin_last_date = timestamp2date(self.historic_data[-1,0,0])
            last_date = timestamp2date(historic_data[-1,0,0])
            print("appending historic_data ended at date: {} from date: {} to date: {} ".format(origin_last_date, first_date, last_date))
            self.historic_data = np.concatenate((self.historic_data, historic_data), axis=0)
        return len(historic_data)

    def test_model_base(self, start_day_index, end_day_index):
        data_manipulator = self.data_manipulator

        data_input, data_output, timestamps, price \
            = data_manipulator.prep_data(start_day_index, end_day_index)

        start_date = timestamp2date(timestamps[0][0])
        end_date = timestamp2date(timestamps[-1][0])
        n_days = end_day_index - start_day_index
        n_seqs = len(data_input)
        print("Running model from {} to {}, {} days, {} seqs".format(start_date, end_date, n_days, n_seqs))

        np_values, np_errors, learning_end_seq = self.run_model(data_input, data_output)
        
        
        self.learning_end_date = timestamp2date(timestamps[learning_end_seq][0])
        print("Running model finished, learning end date is:  {}".format(self.learning_end_date))

        errors_daily = np.mean(np_errors, axis=1)
        assert(len(errors_daily) == len(np_errors))

        # find the best trade strategy.
        # prepare data for the strategy optimization, including timestamp, value, price.

        np_values = data_manipulator.inverse_transform_output(np_values)
        np_real_values = data_manipulator.inverse_transform_output(data_output)
        n_learning_seqs = data_manipulator.get_learning_seqs()

        length = len(np_values)
        strategy_data_input = np.stack((timestamps[-length:], 
                                        np_values,
                                        price[-length:]), axis=2)
        np_real_values = np_real_values[-length:]
        strategy_data_input = data_manipulator.seq_data_2_daily_data(strategy_data_input, is_remove_centralized=True)
        np_real_values = data_manipulator.seq_data_2_daily_data(np_real_values, is_remove_centralized=True)
        return strategy_data_input, np_real_values, errors_daily

    # run the model, do learning and prediction at same time, 
    # this will be used for both training and testing.
    # at the test phase, we should do prediction first
    def run_model(self, data_input, data_output):
        # get the date list.
        n_training_seqs = len(data_input)
        errors = None
        all_outputs = None

        n_learning_seqs = self.data_manipulator.get_learning_seqs()
        n_prediction_seqs = self.data_manipulator.get_prediction_seqs()

        # if the model is ready, this is where we should start to predict.
        learning_end = n_learning_seqs - n_prediction_seqs


        for i in range(0, max(n_training_seqs+1,1), n_prediction_seqs):
            # the model is ready, we want to do prediction first.
            if self.model.is_initialized() == True:
                # do prediction first
                prediction_start = learning_end
                prediction_end = min(prediction_start+n_prediction_seqs, len(data_input))

                if prediction_start < prediction_end:
                    print("starting prediction from seq:{} to seq:{}".format(prediction_start, prediction_end-1))
                    outputs = self.model.predict_and_verify(data_input[prediction_start:prediction_end], 
                                data_output[prediction_start:prediction_end])

                    y = data_output[prediction_start:prediction_end]
                    error = np.square(outputs-y)
                    if all_outputs is None:
                        all_outputs = outputs
                        errors = error
                    else:
                        all_outputs = np.concatenate((all_outputs, outputs), axis=0)
                        errors = np.concatenate((errors, error), axis=0)

            
            if i + n_learning_seqs > n_training_seqs:
                print("expected learning end seq: {}, length of data:{}, training finished".format(i + n_learning_seqs, 
                    n_training_seqs))
                break
            learning_end = i + n_learning_seqs
            print("start training from seq:{} - seq:{}".format(i, learning_end-1))
            self.model.fit(data_input[i:learning_end], data_output[:learning_end], n_prediction_seqs)
            self.initialized = True
 
        return np.squeeze(all_outputs), np.squeeze(errors), learning_end-1

    
    def get_data_manipulator_filename(self, path):
        return os.path.join(path, 'data_manipulator.pkl')
    
    def get_strategy_model_filename(self, path):
        return os.path.join(path, 'strategy.pkl')

    def get_historic_data_filename(self, path, learning_end_date):
        return os.path.join(path, learning_end_date, 'historic_data.npy')

    def save(self):
        assert(self.learning_end_date != None)
        create_if_not_exist(self.save_path)
        path = self.save_path
        # what is the last training date?
        self.model.save(path, self.learning_end_date)
        
        # save the data_manipulator
        filename = self.get_data_manipulator_filename(path)
        with open(filename, 'wb') as f:
            pickle.dump(self.data_manipulator, f, pickle.HIGHEST_PROTOCOL)

        # save the strategy model
        filename = self.get_strategy_model_filename(path)
        with open(filename, 'wb') as f:
            pickle.dump(self.strategy_model, f, pickle.HIGHEST_PROTOCOL)
        
        filename = self.get_historic_data_filename(path, self.learning_end_date)
        np.save(filename, self.historic_data)
    


    def load(self, load_date=None):
        path = self.save_path
        # iterate the path, and find out the latest date as last_training_date
        self.model = StatefulLstmModel()
        
        # get the latest directory
        if load_date == None:
            load_date = get_latest_dir(path)

        if load_date == None:
            return False

        
        print("Loading model for date: {} under: {}".format(load_date, path))
        self.model.load(path, load_date)

        with open(self.get_data_manipulator_filename(path), 'rb') as f:
            self.data_manipulator = pickle.load(f)

        with open(self.get_strategy_model_filename(path), 'rb') as f:
            self.strategy_model = pickle.load(f)

        # recover the learning_end_date
        self.learning_end_date = load_date
        filename = self.get_historic_data_filename(path, load_date)
        self.historic_data = np.load(filename, allow_pickle=True)
        return True

    def get_daily_data(self):
        stock_change_rate = self.historic_data[:,:,3]
        asset_change_rate = self.historic_data[:,:,4]
        daily_stock_change_rate = np.prod(stock_change_rate+1, axis=1) - 1
        daily_asset_change_rate = np.prod(asset_change_rate+1, axis=1) - 1
        date = self.historic_data[:,0,0]
        return np.stack((date, daily_stock_change_rate, daily_asset_change_rate), axis=1)

    def get_historic_metrics(self):
        assert(self.historic_data is not None)
        data = self.get_daily_data()
        training_data_length = self.get_training_data_len()
        training_daily_profit = data[:training_data_length, 2]
        testing_daily_profit = data[training_data_length:, 2]

        training_total_profit = np.prod(training_daily_profit+1)-1
        testing_total_profit = np.prod(testing_daily_profit+1)-1
        return training_total_profit, \
                training_daily_profit, testing_total_profit, \
                testing_daily_profit

    def get_last_training_date(self):
        data = self.get_daily_data()
        training_data_length = self.get_training_data_len()
        return timestamp2date(data[training_data_length-1,0])

    def get_last_testing_date(self):
        data = self.get_daily_data()
        return timestamp2date(data[-1,0])

    def get_profit_sma(self, window=20):
        training_total_profit, training_daily_profit, \
            testing_total_profit, testing_daily_profit = self.get_historic_metrics()

        daily_profit = np.concatenate((training_daily_profit[-window:], testing_daily_profit), axis=0)
        sma = pd.Series(daily_profit).rolling(window).mean().dropna()
        return sma.values

    def report(self, window=20):
        print("Save Path: %s" % self.save_path)
        training_total_profit, training_daily_profit, \
            testing_total_profit, testing_daily_profit = self.get_historic_metrics()
        

        print("Last Training_Date: %s" % self.get_last_training_date())
        print("Training Total Profit: %f" % training_total_profit)
        print("Training Avg Profit: %f" % mean(training_daily_profit))
        print("Training Profit Std %f" % stdev(training_daily_profit))

        training_daily_profit_last_n = self.get_profit_sma(window)
        total_profit_last_training_days = np.prod(training_daily_profit_last_n+1)-1
        print("Training Last %d Days Profit: %f" % (window, total_profit_last_training_days))
        print("Training Last %d Days Avg Profit: %f" % (window, mean(training_daily_profit_last_n)))
        print("Training Last %d Days Profit Std: %f" % (window, stdev(training_daily_profit_last_n)))

        print("Last Testing Date: %s" % self.get_last_testing_date())
        print("Testing Total Profit: %f" % testing_total_profit)
        print("Testing Avg Profit: %f" % mean(testing_daily_profit))
        print("Testing Profit Std %f" % stdev(testing_daily_profit))

        overall_profit = np.concatenate((training_daily_profit, testing_daily_profit), axis=0)
        print("Overall Avg Profit: %f" % mean(overall_profit))
        print("Overall Profit Std: %f" % stdev(overall_profit))

    def get_training_data_len(self):
        return self.data_manipulator.get_training_data_len()

    def plot(self):
        assert(self.historic_data is not None)
        training_data_length = self.get_training_data_len()
        daily_data = self.get_daily_data()
        print("preparing plotting")
        print(daily_data.shape)
        x1 = daily_data[:training_data_length,0]

        stock_training = np.cumprod(daily_data[:training_data_length,1]+1)
        asset_training = np.cumprod(daily_data[:training_data_length,2]+1)

        plt.subplot(2, 1, 1)
        plt.plot(x1,stock_training)
        plt.legend("stock")
        plt.plot(x1,asset_training)
        plt.legend("asset")

        plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%m-%d'))
        #plt.gca().xaxis.set_major_locator(dates.DateLocator())
        
        plt.gcf().autofmt_xdate()
        plt.grid()
        x2 = daily_data[training_data_length:, 0]
        stock_testing = np.cumprod(daily_data[training_data_length:, 1]+1)
        asset_testing = np.cumprod(daily_data[training_data_length:, 2]+1)
        plt.subplot(2, 1, 2)
        plt.plot(x2,stock_testing, label='stock_testing')
        plt.legend("stock")
        plt.plot(x2,asset_testing, label='asset_testing')
        plt.legend("asset")
        plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%m-%d'))
        #plt.gca().xaxis.set_major_locator(dates.DateLocator())
        plt.gcf().autofmt_xdate()
        plt.grid()
        plt.show()

    def get_historic_data(self):
        return self.historic_data

    def plot_day_index(self, day_index):

        # the stock price change rate
        stock = np.cumprod(self.historic_data[day_index,:,3]+1)
        # the asset change rate
        asset = np.cumprod(self.historic_data[day_index,:,4]+1)
        # the date
        date = timestamp2date(self.historic_data[day_index, 0, 0])
        print("Date: {}".format(date))
        # timestamp
        x = self.historic_data[day_index, :, 0]

        plt.subplot(3, 1, 1)
        plt.plot(x,stock,label='stock')
        #plt.legend()
        plt.plot(x,asset,label='asset')
        #plt.legend()
        plt.gcf().autofmt_xdate()
        plt.grid()


        # get the thresholds, 
        strategy_features = self.strategy_model.get_features()
        buy_threshold = np.array([strategy_features[0]] * len(x))
        sell_threshold = np.array([strategy_features[1]] * len(x))

        # the real values
        real_values = self.historic_data[day_index,:,5]
        # the predicted values



        plt.subplot(3, 1, 2)
        plt.plot(x, real_values,label='real')
        plt.plot(x, buy_threshold, label='buy')
        plt.plot(x, sell_threshold, label='sell')
        #plt.legend()
        plt.gcf().autofmt_xdate()
        plt.grid()

        predicted_values = self.historic_data[day_index,:,1]
        plt.subplot(3, 1, 3)
        plt.plot(x, predicted_values, label='predicted')
        plt.plot(x, buy_threshold, label='buy')
        plt.plot(x, sell_threshold, label='sell')
        #plt.legend()
        plt.gcf().autofmt_xdate()
        plt.grid()
        plt.show()

    def plot_days(self):
        data_len = self.historic_data.shape[0]
        interact(self.plot_day_index, day_index=widgets.IntSlider(min=0, max=data_len-1, value=data_len-1))

    def plot_date(self, date):
        day_index = self.data_manipulator.get_historic_day_index(date)
        print("Plotting date:{} day index: {}".format(date, day_index))
        assert(day_index is not None)
        self.plot_day_index(day_index)

    def get_values_by_date(self, date):
        day_index = self.data_manipulator.get_historic_day_index(date)
        assert(day_index is not None)
        return self.get_values_by_index(day_index)

    def get_values_by_index(self, day_index):
        real_values = self.historic_data[day_index,:,5]
        # the predicted values
        predicted_values = self.historic_data[day_index,:,1]
        return real_values, predicted_values


if __name__ == '__main__':
    import shutil

    model_save_path = 'my_model'
    shutil.rmtree(model_save_path)


    npy_path = get_preprocessed_data_dir()
    stock_data_path = get_stock_data_dir()
    strategy_cache_file = os.path.join(stock_data_path, "HM-B_992", "0-80", "strategy_cache.txt")
    from tradestrategy import TradeStrategyFactory
    trade_strategy_factory = TradeStrategyFactory()

    strategy_list = trade_strategy_factory.create_from_file(strategy_cache_file, 10)
    stock_worm = StockWorm('HM-B', 992, npy_path, 'my_model')

    features=[60.0 , 0.004 , 1.0 , 0.0 , 40.0 , 20.0 ,  1.0 , 99.0,  20.0 , 1.0,  1.0 , 1.0,  1.0]
    total_profit, profit_daily, errors_daily = stock_worm.init(features, strategy_list, 0, 80)
    print("Training finished: total_profit:{}".format(total_profit))
    print("prod of profit_daily:{}".format(np.prod(np.array(profit_daily)+1)-1))
    stock_worm.save()

    total_profit_test, profit_daily_test, n_data_appended = stock_worm.test('190619')
    print("Testing finished: total_profit:{}, data for {} days appended".format(total_profit_test, n_data_appended))
    stock_worm.save()

    print("Testing Realtime function:")
    stock_worm2 = StockWorm('HM-B', 992, npy_path, 'my_model')
    stock_worm2.load()
    #stock_worm2.plot()
    stock_worm2.report()
    n_steps = 20
    outputs_realtime = stock_worm2.func_test_test_realtime('190620', n_steps=n_steps)
    print("predicted values - realtime")
    print(outputs_realtime[:, 7:n_steps])


    print("Testing non-realtime function")
    stock_worm3 = StockWorm('HM-B', 992, npy_path, 'my_model')
    stock_worm3.load()
    stock_worm3.test('190620')
    historic_data = stock_worm3.get_historic_data()
    outputs_non_realtime = historic_data[-1, :n_steps, 1]
    # print the predicted values.
    print("predicted values - non realtime")
    print(outputs_non_realtime[:n_steps])






