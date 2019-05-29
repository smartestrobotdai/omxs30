#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
import copy
from sklearn import preprocessing
import datetime
import pickle
import os.path


# In[ ]:


def sma(data, window):
    """
    Calculates Simple Moving Average
    http://fxtrade.oanda.com/learn/forex-indicators/simple-moving-average
    """
    if len(data) < window:
        return None
    return sum(data[-window:]) / float(window)

def get_ema(data, window):
    if len(data) < 2 * window:
        raise ValueError("data is too short")
    c = 2.0 / (window + 1)
    current_ema = sma(data[-window*2:-window], window)
    for value in data[-window:]:
        current_ema = (c * value) + ((1 - c) * current_ema)
    return current_ema


# In[ ]:


class NetAttributes:
    def __init__(self, n_neurons = 100, 
                 learning_rate = 0.003, 
                 num_layers = 1,
                 rnn_type = 2,
                 n_repeats = 2):
        self.n_neurons = n_neurons;
        self.learning_rate = learning_rate;
        self.num_layers = num_layers;
        self.rnn_type = rnn_type;
        self.n_repeats = n_repeats
        self.n_steps = None
        self.n_inputs = None
        self.n_outputs = 1
        
    def set_input_dimension(self, n_steps, n_inputs):
        self.n_steps = n_steps
        self.n_inputs = n_inputs


# In[ ]:


class NetStates:
    def __init__(self):
        self.prediction_states = None
        self.training_states = None
    


# In[ ]:


class StatefulLstmModel:
    def __init__(self,
                n_neurons=100,
                learning_rate=0.002,
                num_layers=2,
                rnn_type=1,
                n_repeats=30):

        self.net_attributes = NetAttributes(n_neurons,
                                   learning_rate,
                                   num_layers,
                                   rnn_type,
                                   n_repeats)
        self.net_states = NetStates()
        self.model_initialized = False
        self.sess = None
    
    def __del__(self):
        if self.sess != None:
            self.sess.close()
    
    def get_batch(self, seq_index, data_train_input, data_train_output):
        X_batch = data_train_input[seq_index:seq_index+1]
        y_batch = data_train_output[seq_index:seq_index+1]
        return X_batch, y_batch
    
    
    def initialize_layers(self):
        layers = None
        net_attributes = self.net_attributes
        if net_attributes.rnn_type == 0:
            layers = [tf.nn.rnn_cell.BasicLSTMCell(net_attributes.n_neurons) 
              for _ in range(net_attributes.num_layers)]
        elif net_attributes.rnn_type == 1:
            layers = [tf.nn.rnn_cell.LSTMCell(net_attributes.n_neurons, use_peepholes=False) 
              for _ in range(net_attributes.num_layers)]
        elif net_attributes.rnn_type == 2:
            layers = [tf.nn.rnn_cell.LSTMCell(net_attributes.n_neurons, use_peepholes=True) 
              for _ in range(net_attributes.num_layers)]
        else:
            print("WRONG")
        return layers
    
    def reset_graph(self, seed=42):
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        np.random.seed(seed)
    
    def create_model(self):
        net_attributes = self.net_attributes
        self.X = tf.placeholder(tf.float32, [None, net_attributes.n_steps, net_attributes.n_inputs])
        self.y = tf.placeholder(tf.float32, [None, net_attributes.n_steps, net_attributes.n_outputs])
        layers = self.initialize_layers()
        cell = tf.nn.rnn_cell.MultiRNNCell(layers)
        self.init_state = tf.placeholder(tf.float32, [net_attributes.num_layers, 2, 1, net_attributes.n_neurons])
        
        state_per_layer_list = tf.unstack(self.init_state, axis=0)
        rnn_tuple_state = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
             for idx in range(net_attributes.num_layers)]
        )
        
        rnn_outputs, self.new_states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32, 
                                                    initial_state=rnn_tuple_state)
        
        stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, net_attributes.n_neurons])
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, net_attributes.n_outputs)
        self.outputs = tf.reshape(stacked_outputs, [-1, net_attributes.n_steps, net_attributes.n_outputs])
        
        self.loss = tf.reduce_mean(tf.square(self.outputs - self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=net_attributes.learning_rate)
        self.training_op = optimizer.minimize(self.loss)

        self.init = tf.global_variables_initializer()
        self.model_initialized = True
    
    # train the model, input is the training data for one cycle
    # input is in the shape: [days, steps, features], the features are 
    # 1. diff, 2. volume. 3. timesteps.
    def fit(self, data_train_input, data_train_output, prediction_period):
        net_attributes = self.net_attributes
        net_states = self.net_states
        n_inputs = data_train_input.shape[2]
        n_steps = data_train_input.shape[1]

        net_attributes.set_input_dimension(n_steps, n_inputs)
        batch_size = 1
        days = data_train_input.shape[0]
        
        self.reset_graph()
        self.create_model()
        my_loss_train_list = []
        sess = tf.Session()
        # TODO: load from file.

        self.init.run(session=sess)
        # if this is the first time of fit?
        if self.net_states.training_states == None:
            init_states = np.zeros((net_attributes.num_layers, 2, 1, net_attributes.n_neurons))
        else:
            init_states = self.net_states.training_states
            
        for repeat in range(net_attributes.n_repeats):
            rnn_states = copy.deepcopy(init_states)
            for seq in range(days):
                X_batch, y_batch = self.get_batch(seq, data_train_input, data_train_output)
                feed_dict = {
                        self.X: X_batch,
                        self.y: y_batch,
                        self.init_state: rnn_states}
                my_op, rnn_states, my_loss_train, my_outputs = sess.run([self.training_op, 
                          self.new_states, 
                          self.loss, 
                          self.outputs], feed_dict=feed_dict)

                my_loss_train_list.append(my_loss_train)
                # last repeat , remember the sates
                if seq+1 == prediction_period and repeat == net_attributes.n_repeats-1:
                    # next training loop starts from here
                    training_states = copy.deepcopy(rnn_states)
                my_loss_train_avg = sum(my_loss_train_list) / len(my_loss_train_list)

            print("{} repeat={} training finished, training MSE={}".format(
                datetime.datetime.now().time(),
                repeat, my_loss_train_avg))
        
        self.net_states.training_states = training_states
        self.net_states.prediction_states = rnn_states
        self.sess = sess
        return
    
    def predict_base(self, data_test_input, data_test_output=None):
        net_attributes = self.net_attributes
        net_states = self.net_states
        days = data_test_input.shape[0]
        
        rnn_states = copy.deepcopy(net_states.prediction_states)
        #X, y, init_state, init, training_op, new_states, loss, outputs = self.create_model()
        sess = self.sess
        
        my_loss_test_list = []
        input_shape = data_test_input.shape
        outputs_all_days = np.zeros((input_shape[0], input_shape[1], 1))
        for seq in range(days):
            if data_test_output is None:
                feed_dict = {
                    self.X: data_test_input[seq:seq+1],
                    self.init_state: rnn_states,
                }

                rnn_states, my_outputs = sess.run([self.new_states, self.outputs], feed_dict=feed_dict)
            else:
                feed_dict = {
                    self.X: data_test_input[seq:seq+1],
                    self.y: data_test_output[seq:seq+1],
                    self.init_state: rnn_states,
                }

                rnn_states, my_outputs, my_loss_test = sess.run([self.new_states, 
                                                                 self.outputs, self.loss], feed_dict=feed_dict)
                print("Predicting seq:{} testing MSE: {}".format(seq, my_loss_test))
            outputs_all_days[seq] = my_outputs
            
        
        return outputs_all_days
    
    def predict(self, data_test_input):
        return self.predict_base(data_test_input)
        
    def predict_and_verify(self, data_test_input, data_test_output):
        return self.predict_base(data_test_input, data_test_output)
      
    def get_attributes_filename(self, path):
        if path[-1] != '/':
            path += '/'
        return path + 'net_attributes.pkl'
    
    def get_path(self, path, date):
        return os.path.join(path, date)

    
    def get_states_filename(self, path, date):
        return os.path.join(self.get_path(path, date), 'net_states.pkl')
    
    def get_model_filename(self, path, date):
        return os.path.join(self.get_path(path, date),'tf_session.ckpt')
    
    def save(self, path, date):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, self.get_model_filename(path, date))
        with open(self.get_attributes_filename(path), 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.net_attributes, f, pickle.HIGHEST_PROTOCOL)
        with open(self.get_states_filename(path, date), 'wb') as f:
            pickle.dump(self.net_states, f, pickle.HIGHEST_PROTOCOL)
        print("Model saved in path: %s" % path)
        
            
    def load(self, path, date):
        # TODO: if date is none, load the latest.
        
        # restore hyper-params
        with open(self.get_attributes_filename(path), 'rb') as f:
            self.net_attributes = pickle.load(f)

        # restore states
        with open(self.get_states_filename(path, date), 'rb') as f:
            self.net_states = pickle.load(f)
        
        # 2. restore graph
        if self.model_initialized == False:
            self.reset_graph()
            self.create_model()
        
        # 3. restore session
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, self.get_model_filename(path, date))
        print("Model restored.")


# In[ ]:


class TimeFormat:
    NONE = 0
    DAY = 1
    WEEK = 2

class DataManipulator:
    def __init__(self,  n_learning_days,
                n_prediction_days, beta, ema, time_format, volume_input, use_centralized_bid, 
                split_daily_data, n_training_days):
        self.n_learning_days = n_learning_days
        self.n_prediction_days = n_prediction_days
        self.beta = beta
        self.ema = ema
        self.time_format = time_format
        self.volume_input = volume_input
        self.use_centralized_bid = use_centralized_bid
        self.split_daily_data = split_daily_data
        self.n_training_days = n_training_days
        self.last_learning_date = None
        self.next_prediction_seq = None
        self.next_learning_seq = None
        
        if split_daily_data == True:
            self.n_learning_seqs = self.n_learning_days * 2
            self.n_prediction_seqs = self.n_prediction_days * 2
        else:
            self.n_learning_seqs = self.n_learning_days
            self.n_prediction_seqs = self.n_prediction_days
        
        self.scaler_input = None
        self.scaler_output = None
    
    def update(self, next_prediction_seq, last_learning_date):
        assert(last_learning_date != None)
        print("updating, next_prediction_seq={}, last_learning_date={}".format(next_prediction_seq, last_learning_date))
        self.next_prediction_seq = next_prediction_seq
        self.next_learning_seq = next_prediction_seq - self.n_learning_seqs
        self.last_learning_date = last_learning_date
    
    def volume_transform(self, volume_series):
        # all the volumes must bigger than 0
        assert(np.all(volume_series>=0))
        return  np.log(volume_series.astype('float')+1)

    def inverse_transform_output(self, scaled_outputs):
        ori_shape = scaled_outputs.shape
        outputs_reshaped = scaled_outputs.reshape((ori_shape[0]*ori_shape[1], 
                                                   1))
        #outputs = np.exp(self.scaler_output.inverse_transform(outputs_reshaped)) - 1
        outputs = self.scaler_output.inverse_transform(outputs_reshaped)
        return outputs.reshape(ori_shape)
    
    def transform(self, data, n_inputs, n_outputs):
        input_scaled = self.transform_input(data[:,:,:n_inputs])
        output_scaled = self.transform_output(data[:,:,-n_outputs:])
        return input_scaled, output_scaled
    
    def transform_input(self, data_input):
        return self.transform_helper(self.scaler_input, data_input)
    
    def transform_output(self, data_output):
        return self.transform_helper(self.scaler_output, data_output)
        
    def transform_helper(self, scaler, data):
        shape = data.shape
        data = data.reshape(shape[0]*shape[1],shape[2])
        data_scaled = scaler.transform(data)
        return data_scaled.reshape(shape)
    
    # do fit and transform at same time
    def fit_transform(self, data_all, n_inputs, n_outputs):
        orig_shape = data_all.shape
        data_train_reshape = data_all.astype('float').reshape((orig_shape[0] * orig_shape[1], orig_shape[2]))
        
        self.scaler_input = preprocessing.MinMaxScaler().fit(data_train_reshape[:,:n_inputs])
        data_train_input_scaled = self.scaler_input.transform(data_train_reshape[:,:n_inputs])
        
        # the invalid step, we change it to zero!
        data_train_input_scaled[~np.any(data_train_reshape, axis=1)] = 0
        data_train_input = data_train_input_scaled.reshape(orig_shape[0], orig_shape[1], n_inputs)
        
        self.scaler_output = preprocessing.MinMaxScaler().fit(data_train_reshape[:,-n_outputs:])
        data_train_output_scaled = self.scaler_output.transform(data_train_reshape[:,-n_outputs:])
        # the invalid step, we change it to zero!
        data_train_output_scaled[~np.any(data_train_reshape, axis=1)] = 0
        data_train_output = data_train_output_scaled.reshape(orig_shape[0], orig_shape[1], n_outputs)
        
        return data_train_input, data_train_output

    # to purge data based on parameters like time_input, split_daily_data, etc.
    def purge_data(self, input_path, stock_index):
        # load numpy file
        npy_file_name = input_path + "/ema{}_beta{}_{}.npy".format(self.ema, self.beta, stock_index)
        input_np_data = np.load(npy_file_name, allow_pickle=True)
        
        # date list
        date_list = []
        for i in range(self.n_training_days):    
            date = input_np_data[i][0][5].date().strftime("%y%m%d")
            date_list.append(date_list)
        
        
        # check if we have days more than training period
        assert(input_np_data.shape[0] >= self.n_training_days)
        # the diff is the mandatory
        input_columns = [2]
        
        time_format = self.time_format
        
        if time_format == TimeFormat.DAY:
            input_columns += [0]
        elif time_format == TimeFormat.WEEK:
            input_columns += [1]
        
        if self.volume_input == 1:
            input_columns += [3]
        
        output_columns = [4]
        timestamp_column = [5]
        price_column = [6]
        input_np_data = input_np_data[:,:,input_columns + output_columns + timestamp_column + price_column]
        
        # we must tranform the volume for it is too big.
        if self.volume_input == 1:
            input_np_data[:,:,-4] = self.volume_transform(input_np_data[:,:,-4])
        
        if self.use_centralized_bid == 0:
            # remove all the rows for centralized bid. it should be from 9.01 to 17.24, which is 516-12=504 steps
            input_np_data = input_np_data[:,7:-5,:]
            
        shape = input_np_data.shape
        n_training_sequences = self.n_training_days
        if self.split_daily_data == 1:
            assert(shape[1] % 2 == 0)
            input_np_data = input_np_data.reshape((shape[0]*2, 
                                                  int(shape[1]/2), 
                                                  shape[2]))
            # get the first date and last date
            n_training_sequences *= 2
        
        return input_np_data, n_training_sequences, input_columns
    
    def prep_training_data(self, input_path, stock_index):
        input_np_data, n_training_sequences, input_columns = self.purge_data(input_path, stock_index)
        # to scale the data, but not the timestamp and price
        data_train_input, data_train_output = self.fit_transform(input_np_data[:n_training_sequences,:,:-2], len(input_columns), 1)
        return data_train_input, data_train_output, input_np_data[:n_training_sequences,:,-2], input_np_data[:n_training_sequences,:,-1]
    
    def prep_testing_data(self, input_path, stock_index):
        input_np_data, n_training_sequences, input_columns = self.purge_data(input_path, stock_index)
        test_start_seq = self.next_prediction_seq - self.n_learning_seqs
        data_test_input, data_test_output = self.transform(input_np_data[test_start_seq:,:,:-2], len(input_columns), 1)
        return data_test_input, data_test_output, input_np_data[test_start_seq:,:,-2], input_np_data[test_start_seq:,:,-1]
    


# In[ ]:


import numpy as np
from pathlib import Path
import pandas as pd
import GPy
import GPyOpt

class ValueModel:
    mixed_domain = [{'name': 'n_neurons', 'type': 'discrete', 'domain': tuple(range(20,160,20))},
          {'name': 'learning_rate', 'type': 'discrete', 'domain': (0.001,0.002,0.003,0.004)},
          {'name': 'num_layers', 'type': 'discrete', 'domain': (1,2,3,4)},
          {'name': 'rnn_type', 'type': 'discrete', 'domain': (0,1,2)},
          {'name': 'learning_period', 'type': 'discrete', 'domain': (10,20,30,40)},
          {'name': 'prediction_period', 'type': 'discrete', 'domain': (1,2,5,10)},
          {'name': 'n_repeats', 'type': 'discrete', 'domain': (3,5,10,20,30,40)},
          {'name': 'beta', 'type': 'discrete', 'domain': (99,)},
          {'name': 'ema', 'type': 'discrete', 'domain': (20,)},
          {'name': 'time_format', 'type': 'discrete', 'domain': (0,1,2)}, #1 for stepofday, 2 for stepofweek
          {'name': 'volume_input', 'type': 'discrete', 'domain': (0,1)},
          {'name': 'use_centralized_bid', 'type': 'discrete', 'domain': (0,1)},
          {'name': 'split_daily_data', 'type': 'discrete', 'domain': (0,1)}
         ]
    
    mixed_domain_test = [{'name': 'n_neurons', 'type': 'discrete', 'domain': tuple(range(20,160,20))},
          {'name': 'learning_rate', 'type': 'discrete', 'domain': (0.001,0.002,0.003,0.004)},
          {'name': 'num_layers', 'type': 'discrete', 'domain': (1,2,3,4)},
          {'name': 'rnn_type', 'type': 'discrete', 'domain': (0,1,2)},
          {'name': 'learning_period', 'type': 'discrete', 'domain': (10,20)},
          {'name': 'prediction_period', 'type': 'discrete', 'domain': (5,10)},
          {'name': 'n_repeats', 'type': 'discrete', 'domain': (3,5)},
          {'name': 'beta', 'type': 'discrete', 'domain': (99,)},
          {'name': 'ema', 'type': 'discrete', 'domain': (20,)},
          {'name': 'time_format', 'type': 'discrete', 'domain': (0,1,2)}, #1 for stepofday, 2 for stepofweek
          {'name': 'volume_input', 'type': 'discrete', 'domain': (0,1)},
          {'name': 'use_centralized_bid', 'type': 'discrete', 'domain': (0,1)},
          {'name': 'split_daily_data', 'type': 'discrete', 'domain': (0,1)}
         ]
    
    
    def __init__(self, stock_name, stock_index, n_training_days):
        self.stock_name = stock_name
        self.stock_index = stock_index
        self.n_training_days = n_training_days
        self.save_path = "model_{}_{}".format(stock_name, n_training_days)
        self.last_training_date = None
        self.model = None
        self.max_profit = -999.0
        return
    
    def get_parameter_str(self, X):
        parameter_str = ""
        for i in range(len(self.mixed_domain)):
            parameter_str += self.mixed_domain[i]["name"]
            parameter_str += ':'
            parameter_str += str(X[i])
            parameter_str += ','
        return parameter_str
    
    def get_max_steps(self, groups):
        max_steps = 0
        for index, df in groups:
            df_len = len(df)
            if df_len > max_steps:
                max_steps = df_len
        return max_steps

    
    def get_data_prep_desc_filename(self, path):
        return path + '/data_prep_desc.pkl'
    

    
    def optimize(self, max_iter=300, is_test=False):
        if is_test == True:
            mixed_domain = self.mixed_domain_test
        else:
            mixed_domain = self.mixed_domain
        
        opt_handler = GPyOpt.methods.BayesianOptimization(f=self.opt_func,  # Objective function       
                                     domain=mixed_domain,          # Box-constraints of the problem
                                     initial_design_numdata = 30,   # Number data initial design
                                     acquisition_type='EI',        # Expected Improvement
                                     exact_feval = True, 
                                     maximize = True)           # True evaluations, no sample noise
        opt_handler.run_optimization(max_iter, eps=0)
    
    def get_data_manipulator_filename(self):
        return os.path.join(self.save_path, 'data_manipulator.pkl')
    
    def save(self):
        # what is the last training date?
        self.model.save(self.save_path, self.data_manipulator.last_learning_date)
        
        # save the data_manipulator
        filename = self.get_data_manipulator_filename()
        with open(filename, 'wb') as f:
            pickle.dump(self.data_manipulator, f, pickle.HIGHEST_PROTOCOL)
        
        # save the strategy model
        self.strategy_model.save(self.save_path)
    
    
    def get_latest_dir(self, save_path):
        all_subdirs = [d for d in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, d))]
        max_time = 0
        for dirname in all_subdirs:
            fullname = os.path.join(save_path, dirname)
            time = os.path.getmtime(fullname)
            if time > max_time:
                max_time = time
                result = dirname
        return result

        
    def load(self, load_date=None):
        save_path = self.save_path
        # iterate the path, and find out the latest date as last_training_date
        self.model = StatefulLstmModel()
        
        # get the latest directory
        if load_date == None:
            load_date = self.get_latest_dir(self.save_path)
        
        print("Loading model for date: {}".format(load_date))
        self.model.load(self.save_path, load_date)
        
        # load data manipulator
        with open(self.get_data_manipulator_filename(), 'rb') as f:
            self.data_manipulator = pickle.load(f)
        
        # load strategy
        self.strategy_model = StrategyModel()
        self.strategy_model.load(self.save_path)
        print("Model loaded!")
        
    def get_avg_of_list(self, profit_list):
        return sum(profit_list)/len(profit_list)
        
    def get_ema_profit_per_day(self, profit_list, split_daily_data, window):
        daily_profit_list = self.get_daily_list(profit_list, split_daily_data)
        
        half = int(len(daily_profit_list)/2)
        if window > half:
            window = half
            
        return get_ema(daily_profit_list, window)
        
    def get_daily_list(self, profit_list, split_daily_data):
        result = []
        if split_daily_data == True:
            for i in range(0, len(profit_list), 2):
                daily_profit = (1+profit_list[i])*(1+profit_list[i+1])-1
                result.append(daily_profit)
            return result
        else:
            return profit_list
        
    def get_total_profit_from_list(self, profit_list, seq_num):
        tot_profit = 1
        for i in range(seq_num):
            tot_profit *= (1+profit_list[i])
        return tot_profit - 1
    
    def opt_func(self, X_list):
        assert(len(X_list)==1)
        X_list = X_list[0]
        print(self.get_parameter_str(X_list))
        
        # do 2-layer optimizations.
        error_ema, error_mean, model, data_manipulator, strategy_model =             self.get_profit(X_list)

        max_profit_list = strategy_model.get_max_profit_list()
        print("max_profit_list length:{}".format(len(max_profit_list)))
        
        max_profit_list = self.get_daily_list(max_profit_list, data_manipulator.split_daily_data)
        print("max_profit_list length:{}".format(len(max_profit_list)))
        # get profit for the training period.
        avg_training_profit = self.get_avg_of_list(max_profit_list)
        ema_5_training_profit = get_ema(max_profit_list, 5)
        ema_10_training_profit = get_ema(max_profit_list, 10)
        
        # get the overall profit for the testing period.
        test_profit, test_profit_list = self.test(model, data_manipulator, strategy_model)
        
        test_profit_list = self.get_daily_list(test_profit_list, data_manipulator.split_daily_data)
        profit_1 = test_profit_list[0]
        
        profit_5 = self.get_total_profit_from_list(test_profit_list, 5)
        
        profit_10 = self.get_total_profit_from_list(test_profit_list, 10)
        
        
        print("FINAL RESULT: {},{},{},{},{},{},{},{},{},{},{}".format(data_manipulator.ema,
                                                                      data_manipulator.beta,
                                                                      avg_training_profit, 
                                                                      ema_5_training_profit, 
                                                                      ema_10_training_profit,
                                                                      error_ema, 
                                                                      error_mean, 
                                                                      profit_1, 
                                                                      profit_5,  
                                                                      profit_10,
                                                                      test_profit))
        
        if ema_10_training_profit > self.max_profit and ema_10_training_profit > 0:
            #print("find the new best profit:{}, error:{}".format(profit_ema_per_day, error_ema))
            self.max_profit = ema_10_training_profit
            self.model = model
            self.data_manipulator = data_manipulator
            self.strategy_model = strategy_model
            #self.test()
            self.save()
 
            
        return np.array(ema_10_training_profit).reshape((1,1))

    def test(self, model, data_manipulator, strategy_model):        
        data_testing_input, data_testing_output, timestamps, price             = data_manipulator.prep_testing_data('npy_files', self.stock_index)
        
        # first make a prediction, then do training.
        n_learning_seqs = data_manipulator.n_learning_seqs
        n_prediction_seqs = data_manipulator.n_prediction_seqs
        
        prediction_start = n_learning_seqs - n_prediction_seqs
        prediction_end = prediction_start + n_prediction_seqs
        
        print("starting the first prediction from seq:{} to seq:{}".format(prediction_start, prediction_end-1))
        outputs = model.predict_and_verify(data_testing_input[prediction_start:prediction_end], 
                    data_testing_output[prediction_start:prediction_end])
        
        print("outputs")
        print(outputs.shape)
        shape = outputs.shape
        assert(shape[2]==1)
        outputs = outputs.reshape((shape[0],shape[1]))
        np_values, np_errors, next_prediction_seq = self.run_model(model, 
                                                                  data_testing_input, 
                                                                  data_testing_output, 
                                                                  n_learning_seqs, 
                                                                  n_prediction_seqs)
        
        
        
        last_learning_date = self.get_date(timestamps, next_prediction_seq-1)
        data_manipulator.update(next_prediction_seq, last_learning_date)
        print("timestamps")
        print(timestamps[prediction_start:].shape)
        print(outputs.shape)
        print(np_values.shape)
        print(price[prediction_start:].shape)
        
        outputs = np.concatenate((outputs, np_values), axis=0)
        outputs = data_manipulator.inverse_transform_output(outputs)
        strategy_data_input = np.stack((timestamps[prediction_start:], 
                                outputs,
                                price[prediction_start:]), axis=2)
        tot_profit = 1
        profit_list = []
        for i in range(0, len(strategy_data_input), n_prediction_seqs):
            start = i
            end = min(i+n_prediction_seqs, len(strategy_data_input))
            result, result_list = strategy_model.run_test(strategy_data_input[start:end])
            tot_profit *= result
            profit_list += result_list
            #strategy_model.append_data(strategy_data_input[start:end])
            #strategy_model.optimize()
        
        #print("test finished, total profit: {} in {} seqs".format(tot_profit, len(strategy_data_input)))
        return tot_profit-1, profit_list
    
    def get_date(self, timestamps, seq_no):
        return timestamps[seq_no][0].date().strftime("%y%m%d")

    def get_profit(self, features):
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
        
        data_manipulator = DataManipulator(learning_period,
                                           prediction_period,
                                           beta, ema, 
                                           time_format, 
                                           volume_input, 
                                           use_centralized_bid, 
                                           split_daily_data, 
                                           self.n_training_days)
        
        data_training_input, data_training_output, timestamps, price             = data_manipulator.prep_training_data('npy_files', self.stock_index)
        
        model = StatefulLstmModel(n_neurons, learning_rate, num_layers, rnn_type, n_repeats)
        
        n_learning_seqs = data_manipulator.n_learning_seqs
        n_prediction_seqs = data_manipulator.n_prediction_seqs
        
        np_values, np_errors, next_prediction_seq = self.run_model(model, data_training_input, data_training_output, 
                                            n_learning_seqs, n_prediction_seqs)
        
        last_learning_date = self.get_date(timestamps, next_prediction_seq-1)
        data_manipulator.update(next_prediction_seq, last_learning_date)
       
        daily_errors = np.mean(np_errors, axis=1)
        print("daily_errors")
        print(daily_errors.shape)
        error_ema = get_ema(daily_errors, int(len(daily_errors)/2))
        assert(len(daily_errors) != 0)
        error_mean = np.sum(daily_errors)/len(daily_errors)
        # find the best trade strategy.
        # prepare data for the strategy optimization, including timestamp, value, price.
        np_values = data_manipulator.inverse_transform_output(np_values)
        strategy_data_input = np.stack((timestamps[n_learning_seqs:], 
                                        np_values, 
                                        price[n_learning_seqs:]), axis=2)
        print("strategy_data_input")
        print(strategy_data_input.shape)
        ema_window = int(strategy_data_input.shape[0]/2)
        
        
        if split_daily_data == True:
            n_max_trades_per_seq = 1
        else:
            n_max_trades_per_seq = 2
        strategy_model = StrategyModel(n_max_trades_per_seq, ema_window)
        strategy_model.append_data(strategy_data_input)
        strategy_model.optimize()
        return error_ema, error_mean, model, data_manipulator, strategy_model
    
    
    # run the model, do learning and prediction at same time, 
    # this will be used for both training and testing.
    # at the test phase, we should do prediction first
    def run_model(self, model, data_input, data_output, n_learning_seqs, n_prediction_seqs):
        # get the date list.
        n_training_seqs = len(data_input)
        errors = None
        all_outputs = None
        n_tot_prediction_seqs = 0
        print("start training: training_seq:{}, learning_seq:{}, prediction_seq:{}".format(n_training_seqs, 
                                                                                           n_learning_seqs, 
                                                                                           n_prediction_seqs,
                                                                                          ))
        for i in range(0, n_training_seqs-n_learning_seqs+1, n_prediction_seqs):
            learning_end = i + n_learning_seqs
            print("start training from seq:{} - seq:{}".format(i, learning_end-1))
            model.fit(data_input[i:learning_end], data_output[:learning_end], n_prediction_seqs)
            next_prediction_seq = learning_end
            prediction_end = min(learning_end+n_prediction_seqs, len(data_input))
            
            if prediction_end <= learning_end:
                break
            
            print("start predicting from seq:{} - seq:{}".format(learning_end, 
                                                                       prediction_end-1))
            
            outputs = model.predict_and_verify(data_input[learning_end:prediction_end], 
                                     data_output[learning_end:prediction_end])
            print("output.shape")
            print(outputs.shape)
            y = data_output[learning_end:prediction_end]
            # error is a 1-D array for the every day error
            error = np.square(outputs-y)
            
            n_tot_prediction_seqs += outputs.shape[0]
            if i == 0:
                all_outputs = outputs
                errors = error
            else:
                all_outputs = np.concatenate((all_outputs, outputs), axis=0)
                errors = np.concatenate((errors, error), axis=0)
        return np.squeeze(all_outputs), np.squeeze(errors), next_prediction_seq
    


# In[ ]:


def print_verbose_func(verbose, msg):
    if verbose == True:
        print(msg)


# In[ ]:


class TradeStrategyDesc:
    def __init__(self,
                 X_list,
                 ema_window,
                 optimize_data):
        self.buy_threshold = X_list[0]
        self.sell_threshold = X_list[1]
        self.stop_loss = X_list[2]
        self.stop_gain = X_list[3]
        self.min_hold_steps = X_list[4]
        self.max_hold_steps = X_list[5]
        self.ema_window = ema_window
        self.optimize_data = optimize_data
        
    def get_parameter_str(self):
        s = "buy_threshold:{} sell_threshold:{} stop_loss:{}             stop_gain:{} min_hold_steps:{} max_hold_steps:{} ema_window:{} optimize_data:{}".format(self.buy_threshold,
                                                  self.sell_threshold,
                                                  self.stop_loss,
                                                  self.stop_gain,
                                                  self.min_hold_steps,
                                                  self.max_hold_steps,
                                                  self.ema_window,
                                                  self.optimize_data.shape)
        return s
    
    
    def to_list(self):
        return [[self.buy_threshold, self.sell_threshold, self.stop_loss, self.stop_gain, 
                 self.min_hold_steps,
                 self.max_hold_steps]]


# In[ ]:


from functools import partial

class StrategyModel:
    mixed_domain = [{'name': 'buy_threshold', 'type': 'continuous', 'domain': (0.0, 0.005)},
                 {'name': 'sell_threshold', 'type': 'continuous', 'domain': (-0.005, 0.0)},
                 {'name': 'stop_loss', 'type': 'continuous', 'domain': (-0.01,-0.003)},
                 {'name': 'stop_gain', 'type': 'continuous', 'domain': (0.002, 0.01)},
                 {'name': 'min_hold_steps', 'type': 'discrete', 'domain': range(10,100)},
                 {'name': 'max_hold_steps', 'type': 'discrete', 'domain': range(50,200)},
         ]
    def __init__(self, n_max_trades_per_seq=4, ema_window=None):
        self.max_profit = -999.0
        self.strategy_desc = None
        self.ema_window = ema_window
        self.optimize_data = None
        self.tot_profit = None
        self.n_max_trades_per_seq = n_max_trades_per_seq
        return
    
    # append the data for the optimization
    def append_data(self, data):
        if self.optimize_data is None:
            self.optimize_data = data
        else:
            self.optimize_data = np.concatenate((self.optimize_data, data), axis=0)

    def optimize(self):
        self.trade_strategy_desc = None
        self.max_profit_ema_per_step = -999.0
        self.input_data = self.optimize_data
        myBopt = GPyOpt.methods.BayesianOptimization(self.get_profit_ema,  # Objective function       
                                             domain=self.mixed_domain,          # Box-constraints of the problem
                                             initial_design_numdata = 30,   # Number data initial design
                                             acquisition_type='EI',        # Expected Improvement
                                             exact_feval = True,
                                             maximize = True)           # True evaluations, no sample noise

        myBopt.run_optimization(150,eps=0)
        self.input_data = None
        return 0
    
    def run_test(self, test_data):
        print("starting test: {}".format(self.trade_strategy_desc.get_parameter_str()))
        X_list = self.trade_strategy_desc.to_list()
        return self.get_total_profit(X_list, test_data)
    
    def get_total_profit(self, X_list, test_data):
        assert(len(X_list) == 1)
        tot_profit, n_tot_trades, daily_profit_list, _, _ = self.run_test_core(X_list[0], 
                                                                                     test_data, 
                                                                                     verbose=True)
        
        print("test finished: tot_profit:{} in {} seqs".format(tot_profit,
                                                                    len(daily_profit_list)))
        return tot_profit, daily_profit_list
    
    # the input data is in shape (days, steps, [timestamp, value, price])
    def get_profit_ema(self, X_list):
        assert(len(X_list)==1)
        X_list = X_list[0]
        input_data = self.input_data[-self.ema_window*2:]
        tot_profit, n_tot_trades, seq_profit_list,             stock_change_rate, asset_change_rate = self.run_test_core(X_list, input_data)
            
        profit_ema = get_ema(seq_profit_list, self.ema_window)
        
        profit_ema_per_step = profit_ema / self.input_data.shape[1]
        if profit_ema_per_step > self.max_profit_ema_per_step:
            print("find best profit_per_step: {} profit_ema:{} tot_profit:{} window:{}".format(
                                                                            profit_ema_per_step,
                                                                            profit_ema,
                                                                            tot_profit,
                                                                            self.ema_window))

            self.max_profit_ema_per_step = profit_ema_per_step
            
            self.change_rate = np.concatenate((input_data, 
                                              stock_change_rate,
                                              asset_change_rate), axis=2)
            self.trade_strategy_desc = TradeStrategyDesc(X_list,
                                             self.ema_window,
                                             self.optimize_data)
            self.tot_profit = tot_profit
            self.max_profit_list = seq_profit_list
        
        return np.array(profit_ema_per_step).reshape((1,1))
    
    def run_test_core(self, X_list, input_data, verbose=False):
        print_verbose = partial(print_verbose_func, verbose)
        buy_threshold = X_list[0]
        sell_threshold = X_list[1]
        stop_loss = X_list[2]
        stop_gain = X_list[3]
        min_hold_steps = int(X_list[4])
        max_hold_steps = int(X_list[5])
        tot_profit = 1
        tot_stock_profit = 1
        buy_step = None
        n_max_trades = self.n_max_trades_per_seq
        cost = 0.00015/2
        n_tot_trades = 0
        # to prepare the result data
        shape = input_data.shape

        reshaped_price = input_data[:,:,2].reshape((shape[0]*shape[1]))
        
        stock_change_rate = np.diff(reshaped_price) / reshaped_price[:-1]
        stock_change_rate = np.concatenate(([0], stock_change_rate)).reshape((shape[0],shape[1],1))
        
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
                change_rate = stock_change_rate[day_idx][step][0]
                if state == 0 and time.time().hour >= 9 and                     n_trades < n_max_trades and step < len(daily_data)-min_hold_steps and                     value > buy_threshold:
                        state = 1
                        asset_change_rate[day_idx][step][0] = -cost
                        tot_profit *= (1-cost)
                        daily_profit *= (1-cost)
                        trade_profit *= (1-cost)
                        print_verbose("buy at step: {} price:{}".format(step, price))
                elif state == 1:
                    if (value < sell_threshold and 
                        hold_steps > min_hold_steps) or step == len(daily_data)-1 or \
                        trade_profit-1 < stop_loss or \
                        trade_profit-1 > stop_gain or \
                        hold_steps >= max_hold_steps:
                        # don't do more trade today!
                        if trade_profit-1 < stop_loss:
                            print_verbose("stop loss stop trading!")
                            n_trades = n_max_trades

                        change_rate = (1+change_rate)*(1-cost)-1 
                        tot_profit *= (1 + change_rate)
                        daily_profit *= (1 + change_rate)
                        state = 0
                        n_trades += 1
                        print_verbose("sell at step: {} price:{} trade_profit:{} hold_steps:{}".format(step, price, trade_profit, hold_steps))
                        trade_profit = 1
                        asset_change_rate[day_idx][step] = change_rate
                        hold_steps = 0
                        
                    else:
                        tot_profit *= (1+change_rate)
                        daily_profit *= (1+change_rate)
                        trade_profit *= (1+change_rate)
                        asset_change_rate[day_idx][step][0] = change_rate
                        hold_steps += 1
            print_verbose("finished day {}, daily profit:{}".format(day_idx,daily_profit))
            daily_profit_list.append(daily_profit - 1)
            n_tot_trades += n_trades
        return tot_profit, n_tot_trades, daily_profit_list, stock_change_rate, asset_change_rate
    
    def get_max_profit_list(self):
        return self.max_profit_list
    
    def get_strategy_desc(self):
        return self.trade_strategy_desc
    
    def get_save_filename(self, path):
        return os.path.join(path, 'strategy_desc.pkl')
    
    def save(self, save_path):
        assert(self.trade_strategy_desc != None)
        with open(self.get_save_filename(save_path), 'wb') as f:
            pickle.dump(self.trade_strategy_desc, f, pickle.HIGHEST_PROTOCOL)
            
    def load(self, save_path):
        with open(self.get_save_filename(save_path), 'rb') as f:
            self.trade_strategy_desc = pickle.load(f)
        this.ema_window = self.trade_strategy_desc.ema_window
        this.optimize_data = self.trade_strategy_desc.optimize_data


# In[ ]:


value_model = ValueModel('Nordea', 5, 60)
value_model.optimize(is_test=False)


# In[ ]:




