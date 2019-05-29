import tensorflow as tf
import numpy as np
import copy
import datetime
import os.path
import pickle

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

class NetStates:
    def __init__(self):
        self.prediction_states = None
        self.training_states = None

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
    
    def is_initialized(self):
        return self.model_initialized

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
        
        if self.is_initialized() == False:
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
        if self.is_initialized() == False:
            self.reset_graph()
            self.create_model()
        
        # 3. restore session
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, self.get_model_filename(path, date))
        print("Model restored.")