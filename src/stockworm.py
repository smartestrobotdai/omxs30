import numpy as np
from pathlib import Path
import pandas as pd
import uuid
import os.path
import pickle
from tradestrategy import TradeStrategy
from tradestrategyfactory import TradeStrategyFactory


from datamanipulator import DataManipulator
from statefullstmmodel import StatefulLstmModel
from util import *
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta  
import matplotlib.dates as dates

from ipywidgets import interact
import ipywidgets as widgets

from historicdata import HistoricData


class StockWorm:
    def __init__(self, stock_name, stock_id, input_data_path, save_path=None, is_future=False, slippage=0):
        self.stock_name = stock_name
        self.stock_id = stock_id 
        self.input_data_path = input_data_path
        self.save_path = save_path
        self.is_future = is_future
        self.last_learning_day_index = None
        self.learning_end_date = None
        self.historic_data = None
        self.data_today = []
        self.slippage = slippage

    def validate(self, features, start_day_index, end_day_index):
        learning_period = int(features[4])
        prediction_period = int(features[5])
        data_len = end_day_index - start_day_index
        if (data_len - learning_period) % prediction_period == 0:
            return True
        else:
            return False    

    def init(self, features, start_day_index, end_day_index, strategy_features=None, is_test=False):
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
        is_stateful = int(features[13])
        ref_stock_id = int(features[14])



        data_manipulator = DataManipulator(self.stock_name,
                                           self.stock_id,
                                           learning_period,
                                           prediction_period,
                                           beta, ema, 
                                           time_format, 
                                           volume_input, 
                                           use_centralized_bid, 
                                           split_daily_data,
                                           ref_stock_id,
                                           self.input_data_path)

        data_manipulator.init_scalers(start_day_index, end_day_index)


        model = StatefulLstmModel(n_neurons, learning_rate, num_layers, rnn_type, n_repeats, is_stateful)
        self.data_manipulator = data_manipulator
        self.model = model

        strategy_data_input, real_values, errors_daily = self.test_model_base(start_day_index, end_day_index)
        strategy_factory = TradeStrategyFactory(is_future=self.is_future, slippage=self.slippage)

        if strategy_features is None:    
            if is_test:
                max_iter = 50
            else:
                max_iter = 100
            strategy_model = strategy_factory.create_trade_strategies(strategy_data_input, 
                iter=1, max_iter=max_iter)

        else:
            strategy_model = strategy_factory.create_strategy(strategy_features)
            
        total_profit, profit_daily, results = strategy_model.get_profit(strategy_data_input)

        self.strategy_model = strategy_model

        data = np.concatenate((strategy_data_input, results, real_values), axis=2)

        self.historic_data = HistoricData(start_day_index, end_day_index)
        self.historic_data.set_training_data(data)

        return total_profit, profit_daily, errors_daily
    
    
    def test(self, end_date=None, verbose=False):
        # find the start_day_index
        assert(self.learning_end_date != None)
        n_learning_days = self.data_manipulator.get_learning_days()
        n_prediction_days = self.data_manipulator.get_prediction_days()
        learning_end_day_index = self.data_manipulator.date_2_day_index(self.learning_end_date)



        start_day_index = learning_end_day_index + 1 - n_learning_days + n_prediction_days
        start_date = self.data_manipulator.day_index_2_date(start_day_index)
        print("DEBUG: learning_end_day_index: {}, start_day_index:  {}, \
            start_date:{}, n_learning_days:{}, n_prediction_days:{}".format(learning_end_day_index,
            start_day_index,
            start_date,
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
            total_profit, profit_daily, change_rate = self.strategy_model.get_profit(strategy_data_input, verbose)


            data = np.concatenate((strategy_data_input, change_rate, real_values), axis=2)
            assert(self.historic_data is not None)
            n_data_appended = self.historic_data.append(data)

        
        _,_,testing_total_profit, testing_profit_list, _,_,_,_= self.historic_data.get_metrics()
        return  testing_total_profit, testing_profit_list, n_data_appended

    def start_realtime_prediction(self, end_date=None):
        # do predction until yesterday's close time and change model states.
        self.test(end_date)
        self.last_price = self.data_manipulator.get_last_price(end_date)
        self.ref_last_price = self.data_manipulator.get_ref_last_price(end_date)
        self.data_today = []
        self.data_ref_today = []

    def process_rt_data(self, rt_data, last_price):
        df = pd.DataFrame(rt_data,columns=['timestamp', 'last', 'volume'])
        df = preprocessing_daily_data(df, last_price, calculate_values=False)
        df = add_step_columns(df)
        df['value'] = 0
        df_input = df[['step_of_day', 'step_of_week', 'diff_ema_{}'.format(self.data_manipulator.ema), 'volume', 'value', 'timestamp', 'last']]

        return df_input

    # this method need to be called every minute.
    def test_realtime(self, timestamp, price, volume, ref_price=None):
        assert(self.last_price != None)


        self.data_today.append([timestamp, price, volume])
        
        df_input = self.process_rt_data(self.data_today, self.last_price)
        if ref_price is not None:
            self.data_ref_today.append([timestamp, ref_price, 0])
            df_input_ref = self.process_rt_data(self.data_ref_today, self.ref_last_price)
        else:
            df_input_ref = None

        input_data, timestamp, price = self.data_manipulator.purge_data_realtime(df_input, df_input_ref)
        outputs = self.model.predict_realtime(input_data)
        outputs_scaled = self.data_manipulator.inverse_transform_output(outputs)


        outputs_scaled = np.squeeze(outputs_scaled)
        output_scaled_daily = self.data_manipulator.seq_data_2_daily_data(outputs_scaled, is_remove_centralized=True)

        timestamp = self.data_manipulator.seq_data_2_daily_data(timestamp, is_remove_centralized=True)

        price = self.data_manipulator.seq_data_2_daily_data(price, is_remove_centralized=True)


        strategy_input = np.stack((timestamp, output_scaled_daily, price), axis=2)
        tot_profit, daily_profit_list, results = self.strategy_model.get_profit(strategy_input, verbose=True)

        actions = results[:,:,2]
        n_steps = len(self.data_today)
        assert(actions.shape[0] == 1)
        assert(output_scaled_daily.shape[0] == 1)
        return output_scaled_daily[0], actions[0]

    def get_dataframe_from_raw(self, stock_id, date):
        df = pd.read_csv('../data/data.csv.gz', compression='gzip', sep=',')
        df = df[df['stock_id'] == stock_id]
        df['timestamp'] = pd.to_datetime(df['time_stamp'], format="%Y-%m-%d %H:%M:%S").dt.tz_convert('CET')
        df['date'] = df['timestamp'].apply(timestamp2date)
        df = df[df['date'] == date]
        return df
        
    def func_test_test_realtime(self, date=None, n_steps=5):
        if date == None:
            date = get_current_date()

        df = self.get_dataframe_from_raw(self.stock_id, date)
        
        ref_stock_id = self.data_manipulator.ref_stock_id

        if ref_stock_id != -1:
            df_ref = self.get_dataframe_from_raw(ref_stock_id, date)        
        else:
            df_ref = None
        # find the last date 
        day_index = self.data_manipulator.date_2_day_index(date)
        last_date = self.data_manipulator.day_index_2_date(day_index - 1)

        print("Starting realtime test, today: {}, last_trade_day:{}".format(date, last_date))

        # the data format: daily step, weekly step, diff, volume, value, timestamp, price.
        self.start_realtime_prediction(last_date)


        for i in range(n_steps):
            if df_ref is None:
                ref_price = None
            else:
                ref_price = df_ref.iloc[i]['last']

            output_scaled_daily, actions = self.test_realtime(df.iloc[i]['timestamp'], df.iloc[i]['last'], df.iloc[i]['volume'],ref_price)

        return output_scaled_daily, actions


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
        print("Running model finished, learning_end_seq: {} learning end date is:  {}".format(learning_end_seq, self.learning_end_date))
        print(np_errors)
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
                    print("starting prediction from seq:{} to seq:{}".format(prediction_start, prediction_end))
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
            print("start training from seq:{} - seq:{}".format(i, learning_end))
            self.model.fit(data_input[i:learning_end], data_output[:learning_end], n_prediction_seqs)
            self.initialized = True
 
        return np.squeeze(all_outputs), np.squeeze(errors), learning_end-1

    def get_test_data(self):
        training_data_length = self.get_training_data_len()
        return self.historic_data[training_data_length:]

    def get_training_data(self):
        training_data_length = self.get_training_data_len()
        return self.historic_data[:training_data_length]
    
    def get_data_manipulator_filename(self, path):
        return os.path.join(path, 'data_manipulator.pkl')
    
    def get_strategy_model_filename(self, path):
        return os.path.join(path, 'strategy.pkl')

    def get_strategy_features(self):
        return self.strategy_model.get_features()

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
        with open(filename, 'wb') as f:
            pickle.dump(self.historic_data, f, pickle.HIGHEST_PROTOCOL)


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
        with open(self.get_historic_data_filename(path, load_date), 'rb') as f:
            self.historic_data = pickle.load(f)
        return True

        date = self.historic_data[:,0,0]
        return np.stack((date, daily_stock_change_rate, daily_asset_change_rate), axis=1)


    def get_last_n_total_profit(self, data_arr, window=20):
        return np.prod(data_arr[-window:]+1)-1

    def report(self, window=20):
        pass
        # print("Save Path: %s" % self.save_path)
        # training_total_profit, training_daily_profit, \
        #     testing_total_profit, testing_daily_profit, \
        #     training_stock_total_profit, taining_stock_daily_profit, \
        #     testing_stock_total_profit, testing_stock_daily_profit = self.historic_data.get_metrics()

        # print(taining_stock_daily_profit)


        # print("Last Training_Date: %s" % self.historic_data.get_last_training_date())
        # print("Training Total Profit: %f" % training_total_profit)
        # print("Training Avg Profit: %f" % mean(training_daily_profit))
        # print("Training Profit Std %f" % stdev(training_daily_profit))

        # last_n_training_total_profit = self.get_last_n_total_profit(training_daily_profit, window)

        # print("Training Last %d Days Profit: %f" % (window, last_n_training_total_profit))
        # print("Training Last %d Days Avg Profit: %f" % (window, mean(training_daily_profit[-window:])))
        # print("Training Last %d Days Profit Std: %f" % (window, stdev(training_daily_profit[-window:])))

        # last_n_training_total_profit_stock = self.get_last_n_total_profit(taining_stock_daily_profit, window)
        # print("Training Last %d Days Stock: %f" % (window, last_n_training_total_profit_stock))
        # print("Training Last %d Days Avg Stock: %f" % (window, mean(taining_stock_daily_profit[-window:])))
        # print("Training Last %d Days Stock Std: %f" % (window, stdev(taining_stock_daily_profit[-window:])))

        # if len(testing_daily_profit) == 0:
        #     return

        # print("Last Testing Date: %s" % self.get_last_testing_date())
        # print("Testing Total Profit: %f" % testing_total_profit)
        # print("Testing Avg Profit: %f" % mean(testing_daily_profit))
        # print("Testing Profit Std %f" % stdev(testing_daily_profit))

        # print("Testing Total Stock: %f" % testing_stock_total_profit)
        # print("Testing Avg Stock: %f" % mean(testing_stock_daily_profit))
        # print("Testing Stock Std %f" % stdev(testing_stock_daily_profit))

        # overall_profit = np.concatenate((training_daily_profit, testing_daily_profit), axis=0)
        # print("Overall Avg Profit: %f" % mean(overall_profit))
        # print("Overall Profit Std: %f" % stdev(overall_profit))

    def get_training_data_len(self):
        return self.data_manipulator.get_training_data_len()

    def plot(self):
        assert(self.historic_data is not None)
        self.historic_data.plot()

    def get_historic_data(self):
        return self.historic_data

    def daily_plot(self):
        self.historic_data.daily_plot(self.strategy_model)

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
    npy_path = get_preprocessed_data_dir()
    stock_data_path = get_stock_data_dir()

    # model_save_path = 'my_model'
    # if os.path.isdir(model_save_path):
    #     shutil.rmtree(model_save_path)

    # stock_worm = StockWorm('HM-B', 992, npy_path, 'my_model')

    # features=[40.0 , 0.004,  7.0 , 2.0,  40.0 ,  5.0  ,1.0 , 99.0 , 20.0 , 2.0  ,1.0,  0.0 , 1.0 , 1.0 , 160271]
    # strategy_features = [0.000000 ,-0.000105 ,-0.015337,  0.010453 , 0.0 ,  5.0]

    # total_profit, profit_daily, errors_daily = stock_worm.init(features, 0, 80, 
    #     strategy_features=strategy_features, is_test=True)


    # print("Training finished: total_profit:{}".format(total_profit))
    # print("prod of profit_daily:{}".format(np.prod(np.array(profit_daily)+1)-1))
    # stock_worm.save()

    # total_profit_test, profit_daily_test, n_data_appended = stock_worm.test('190723')
    # print("Testing finished: total_profit:{}, data for {} days appended".format(total_profit_test, n_data_appended))
    # stock_worm.save()

    print("Testing Realtime function:")
    stock_worm2 = StockWorm('AZN', 3524, npy_path, '../stock-data/AZN_3524/0-100/models/dcf2329e6a4e82d1/')
    #stock_worm2 = StockWorm('HM-B', 992, npy_path, './my_model')
    stock_worm2.load()
    #stock_worm2.plot()
    # stock_worm2.report()
    n_steps = 20
    outputs_realtime, actions = stock_worm2.func_test_test_realtime('190724', n_steps=n_steps)
    print("predicted values - realtime")
    print(outputs_realtime[:n_steps])
    print("actions")
    print(actions[:n_steps])



    print("Testing non-realtime function")
    #stock_worm3 = StockWorm('HM-B', 992, npy_path, './my_model')
    stock_worm3 = StockWorm('AZN', 3524, npy_path, '../stock-data/AZN_3524/0-100/models/dcf2329e6a4e82d1/')
    stock_worm3.load()
    stock_worm3.test('190724')
    stock_worm3.save()
    historic_data = stock_worm3.get_historic_data().get_data()

    # NOTE: if we split the daily data, need to check the second last !
    outputs_non_realtime = historic_data[-1, :n_steps, 1]
    outputs_action = historic_data[-1, :n_steps, 5]
    # print the predicted values.
    print("predicted values - non realtime")
    print(outputs_non_realtime[:n_steps])
    print(outputs_action[:n_steps])






