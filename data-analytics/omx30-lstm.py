#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
import GPy
import GPyOpt
import copy
import time
import datetime
import sys

class Model:
    # Network Parameters
    # n_neurons, learning_rate, num_layers, rnn_type(RNN|BasicLSTM|LSTM|LSTM peelhole)
    # Control Parameters
    # risk_aversion - the margin added to the courtage that leads an buy or sell operation
    # learning_period - how many sequences model should learn before predicting next sequences
    # prediction_period - how many sequences the model should predict
    # max_repeats - how many times in maximum the model should learn
    # min_profit - what is the minimum profit in average during training phase, if the minimum is not reached, the model should not predict
    # gamma - what is the gamma used when preprocessing data
    
    step_profit_list = []
    mixed_domain = [{'name': 'n_neurons', 'type': 'discrete', 'domain': tuple(range(20,160,20))},
          {'name': 'learning_rate', 'type': 'discrete', 'domain': (0.001,0.002,0.003,0.004)},
          {'name': 'num_layers', 'type': 'discrete', 'domain': (1,2,3,4)},
          {'name': 'rnn_type', 'type': 'discrete', 'domain': (0,1,2)},
          {'name': 'learning_period', 'type': 'discrete', 'domain': tuple(range(10,41,10))},
          {'name': 'prediction_period', 'type': 'discrete', 'domain': (1,2,3,5,8,13)},
          {'name': 'max_repeats', 'type': 'discrete', 'domain': tuple(range(1,52,10))},
          {'name': 'beta', 'type': 'discrete', 'domain': (99, 98)},
          {'name': 'ema', 'type': 'discrete', 'domain': (10,20)},
          {'name': 'time_format', 'type': 'discrete', 'domain': (0,1,2)}, #1 for stepofday, 2 for stepofweek
          {'name': 'volume_input', 'type': 'discrete', 'domain': (0,1)},
          {'name': 'use_centralized_bid', 'type': 'discrete', 'domain': (0,1)},
          {'name': 'split_daily_data', 'type': 'discrete', 'domain': (0,1)}
         ]
    def __init__(self, regen):
        if regen == False:
            return
        def column_filter(x):
            if x == 'stepofweek':
                return True
            elif 'diff_ema' in x:
                return True
            elif 'volume' in x:
                return True
            elif 'value_ema' in x:
                return True
            else:
                return False
        for ema in (10, 20):
            for beta in (99, 98):
                filename = "data-prep-ema{}-beta{}.csv".format(ema, beta)
                print("pre-processing {}".format(filename))
                data = pd.read_csv(filename, parse_dates=["timestamp"])
                data['dayofweek'] = data['timestamp'].apply(lambda x: x.weekday())
                groups = data.set_index('timestamp').groupby(lambda x: x.date())
                
                # get maximum steps
                max_steps = 0
                for index, df in groups:
                    df_len = len(df)
                    if df_len > max_steps:
                        max_steps = df_len
                        
                np_data = np.zeros((len(groups), max_steps, 30*3+2))
                filtered_columns = list(filter(column_filter, data.columns))
                i = 0
                for index, df in groups:
                    df['stepofday'] = np.arange(0, max_steps)
                    df['stepofweek'] = df['dayofweek'] * max_steps + df['stepofday']
                    np_data[i] = df[filtered_columns + ['stepofweek','stepofday']].to_numpy()
                    i += 1
                    
                numpy_file_name = "np_ema{}_beta{}.npz".format(ema, beta)
                np.savez_compressed(numpy_file_name, np_data)
                

        return
        
    def get_parameter_str(self, X):
        parameter_str = ""
        for i in range(len(self.mixed_domain)):
            parameter_str += self.mixed_domain[i]["name"]
            parameter_str += ':'
            parameter_str += str(X[i])
            parameter_str += ','
        return parameter_str
    
    def reset_graph(self, seed=42):
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        np.random.seed(seed)
        
    
    def log(self, verbose, msg):
        if verbose:
            print(msg)

    def get_batch(self, seq_index, data_train_input, data_train_output):
        X_batch = data_train_input[seq_index:seq_index+1]
        y_batch = data_train_output[seq_index:seq_index+1]
        return X_batch, y_batch
    
    def transform(self, data_all, n_inputs, n_outputs):
        orig_shape = data_all.shape
        data_train_reshape = data_all.reshape((orig_shape[0] * orig_shape[1], orig_shape[2]))
        
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
    
    def inverse_transform_output(self, scaled_outputs):
        outputs_reshaped = scaled_outputs.reshape((scaled_outputs.shape[1], scaled_outputs.shape[2]))
        #outputs = np.exp(self.scaler_output.inverse_transform(outputs_reshaped)) - 1
        outputs = self.scaler_output.inverse_transform(outputs_reshaped)
        return outputs
    
    def inverse_transform_input(self, scaled_inputs):
        inputs_reshaped = scaled_inputs.reshape((scaled_inputs.shape[1], scaled_inputs.shape[2]))
        #inputs_reshaped[:,4:6] = np.exp(self.scaler_input.inverse_transform(inputs_reshaped)[:,4:6]) - 1
        inputs = self.scaler_input.inverse_transform(inputs_reshaped)
        # TODO: the volume and hold should be transformed back.
        return inputs
        
        
    def get_answer(self, features):
        n_neurons = int(features[0])
        learning_rate = features[1]
        num_layers = int(features[2])
        rnn_type = int(features[3])
        learning_period = int(features[4])
        prediction_period = int(features[5])
        max_repeats = int(features[6])
        beta = int(features[7])
        ema = int(features[8])
        time_input = int(features[9])
        volume_input = int(features[10])
        use_centralized_bid = int(features[11])
        split_daily_data = int(features[12])

        # load data
        file_name = "np_ema{}_beta{}.npz".format(ema, beta)
        data_all = np.load(file_name)['arr_0']
        # pick the data for stock_id
        
        # for the stock 20, the max related stock is 21 (0.94),
        # the medium stock is 18 (0.29), the min related stock is 5 (0.05)
        stock_index = [self.current_stock_index]
        
        # we must convert the array to 2D
        if use_centralized_bid == 0:
            # remove all the rows for centralized bid. it should be from 9.01 to 17.24, which is 516-12=504 steps
            data_all = data_all[:,7:-5,:]
        
        
        orig_shape = data_all.shape
        print("original shape: ")
        print(orig_shape)

        # make it simple, the step must be even number.
        assert(orig_shape[1] % 2 == 0)
        reshaped_data = data_all.reshape((orig_shape[0] * orig_shape[1], 
                                                          orig_shape[2]))
        
        # the mandatory is the diff.
        input_column_list = [30+i for i in stock_index]
        volume_input_list = stock_index
        if time_input == 1:
            input_column_list = [-1] + input_column_list
        elif time_input == 2:
            input_column_list = [-2] + input_column_list
        if volume_input != 0:
            input_column_list = input_column_list + volume_input_list
            
        output_column_list = [60+i for i in stock_index]
        
        reshaped_data_filtered = reshaped_data[:, input_column_list + output_column_list]
        # for volume we use log.
        if volume_input != 0:
            # the last column is the volume
            last_input_index = len(volume_input_list)
            # we must add 1 to the volume value otherwise log(0) is meaningless.
            reshaped_data_filtered[:, -last_input_index:] = np.log(reshaped_data_filtered[:, -last_input_index:]+1)
        
        n_inputs = len(input_column_list)
        n_outputs = len(output_column_list)
        if split_daily_data == 0:
            data_filtered = reshaped_data[:, input_column_list + output_column_list].reshape((orig_shape[0], 
                                                                                          orig_shape[1], 
                                                                                          n_inputs+n_outputs))

        else:
            # we split day data into 2 parts.
            data_filtered = reshaped_data[:, input_column_list + output_column_list].reshape((orig_shape[0]*2, 
                                                                                          int(orig_shape[1]/2), 
                                                                                          n_inputs+n_outputs))
            learning_period *= 2
            prediction_period *= 2
        
        
        np.nan_to_num(data_filtered, copy=False)
        batch_size = 1
        data_train_input, data_train_output = self.transform(data_filtered, n_inputs, n_outputs)

        # data_train_input in the shape [seq, steps, features]
        days = data_train_input.shape[0]

        max_steps = data_train_input.shape[1]
        print("days={}, max_steps={}".format(days, max_steps))
        self.reset_graph()
        
        X = tf.placeholder(tf.float32, [None, max_steps, n_inputs])
        y = tf.placeholder(tf.float32, [None, max_steps, n_outputs])
        
        layers = None
        if rnn_type == 0:
            layers = [tf.nn.rnn_cell.BasicLSTMCell(n_neurons) 
              for _ in range(num_layers)]
        elif rnn_type == 1:
            layers = [tf.nn.rnn_cell.LSTMCell(n_neurons, use_peepholes=False) 
              for _ in range(num_layers)]
        elif rnn_type == 2:
            layers = [tf.nn.rnn_cell.LSTMCell(n_neurons, use_peepholes=True) 
              for _ in range(num_layers)]
        else:
            print("WRONG")
        cell = tf.nn.rnn_cell.MultiRNNCell(layers)
        
        # For each layer, get the initial state. states will be a tuple of LSTMStateTuples.
        init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, n_neurons])
        state_per_layer_list = tf.unstack(init_state, axis=0)
        rnn_tuple_state = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
             for idx in range(num_layers)]
        )

        rnn_outputs, new_states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32, 
                                                    initial_state=rnn_tuple_state)
        
        stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
        outputs = tf.reshape(stacked_outputs, [-1, max_steps, n_outputs])
        
        
        loss = tf.reduce_mean(tf.square(outputs - y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

        # now run the model to get answer:
        rnn_states_before_training = np.zeros((num_layers, 2, batch_size, n_neurons))
        graph_data = []
        my_loss_test_list = []
        my_test_results_list = []
        my_test_answers_list = []
        with tf.Session() as sess:
            init.run()
            for learn_end_seq in range(learning_period, 
                                       days - prediction_period, 
                                       prediction_period):
                learning_start_seq = learn_end_seq - learning_period
                tmp_states = np.zeros((num_layers, 2, batch_size, n_neurons))
                for repeat in range(max_repeats):
                    rnn_states = copy.deepcopy(rnn_states_before_training)
                    my_loss_train_list = []
                    train_asset = 1
                    for seq in range(learning_start_seq, learn_end_seq):
                        X_batch, y_batch = self.get_batch(seq, data_train_input, data_train_output)
                      
                        feed_dict = {
                            X: X_batch,
                            y: y_batch,
                            init_state: rnn_states
                        }
                        
                        my_op, my_new_states, my_loss_train, my_outputs = sess.run([training_op, new_states, loss, outputs], feed_dict=feed_dict)
                        
                        my_loss_train_list.append(my_loss_train)
                        rnn_states = my_new_states
                        if seq-learning_start_seq+1 == prediction_period:
                            # next training loop starts from here
                            tmp_states = copy.deepcopy(rnn_states)
                    my_loss_train_avg = sum(my_loss_train_list) / len(my_loss_train_list)
                    
                    print("{} sequence:{} - {} repeat={} training finished, training MSE={}".format(
                        datetime.datetime.now().time(),
                        learning_start_seq, learn_end_seq, 
                        repeat, my_loss_train_avg))
                # backup the states after training.
                rnn_states_before_training = copy.deepcopy(tmp_states)
                
                
                for seq in range(learn_end_seq, learn_end_seq + prediction_period):
                    X_test, y_test = self.get_batch(seq, data_train_input, data_train_output)
                    feed_dict = {
                        X: X_test,
                        y: y_test,
                        init_state: rnn_states,
                    }
            
                    my_new_states, my_loss_test, my_outputs = sess.run([new_states, loss, outputs], feed_dict=feed_dict)
                    my_loss_test_list.append(my_loss_test)
                    real_outputs = self.inverse_transform_output(my_outputs)
                    real_test = self.inverse_transform_output(y_test)
                    output_and_answer = np.hstack((real_outputs.reshape((max_steps, n_outputs)), 
                                                   real_test.reshape((max_steps, n_outputs))))
                    my_test_results_list.append(output_and_answer)
                    print("sequence:{} test finished, testing MSE={}".format(seq, my_loss_test))
                    rnn_states = my_new_states
            my_loss_test_avg = sum(my_loss_test_list)/len(my_loss_test_list)
            
            return my_loss_test_avg, np.array(my_test_results_list)
                    
    def opt_wrapper(self, X_list):
        answer = np.zeros((X_list.shape[0], 1))
        for i in range(len(X_list)):
            print(self.get_parameter_str(X_list[i]))
            features = X_list[i]
            answer[i][0], results_list = self.get_answer(features)
            #self.draw_step_profit_graph(self.step_profit_list, "step_profit_{}".format(answer[i][0]))
            #self.step_profit_list = []
            if answer[i][0] < self.min_answer:
                print("find new opt for stock {} :{}, {}".format(self.current_stock_index, answer[i][0], self.get_parameter_str(X_list[i])))
                self.min_answer = answer[i][0]
            else:
                print("find result for stock {} :{}, {}".format(self.current_stock_index, answer[i][0], self.get_parameter_str(X_list[i])))
        return answer
                
        
    def optimize(self, stockIndexList, max_iter=300):
        for item in stockIndexList:
            print("Starting optimize stock index {}".format(item))
            self.min_answer = 999
            self.current_stock_index = int(item)
            myBopt = GPyOpt.methods.BayesianOptimization(f=self.opt_wrapper,  # Objective function       
                                                 domain=self.mixed_domain,          # Box-constraints of the problem
                                                 initial_design_numdata = 20,   # Number data initial design
                                                 acquisition_type='EI',        # Expected Improvement
                                                 exact_feval = True)           # True evaluations, no sample noise
            
            myBopt.run_optimization(max_iter,eps=0)
            print("Finishing optimize stock index {}".format(item))
    
    
    # no optimize, we have already knew the answer. run it and save the results into file.
    def run(self, n_neurons, learning_rate, 
            num_layers, rnn_type, 
            learning_period, prediction_period, 
            max_repeats, beta, ema, time_input, volume_input,
            use_centralized_bid, split_daily_data):
        features = [n_neurons, learning_rate, 
            num_layers, rnn_type, 
            learning_period, prediction_period, 
            max_repeats, beta, ema, time_input, volume_input,
            use_centralized_bid, split_daily_data]
        
        answer, my_test_result_list = self.get_answer(features)
        print("Finished, result:{}".format(answer))
        return my_test_result_list


# In[7]:


if len(sys.argv) == 1:
    print("usage: omx30-lstm.py stock_index_list")
    print("ie: omx30-lstm.py 5 14 28")
    sys.exit()

stock_index_list = sys.argv[1:]

# TODO: check the time of npz file and decide should we re-generate it.
model = Model(False)

# In[3]:
model.optimize(stock_index_list)


# In[ ]:




