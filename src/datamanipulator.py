import numpy as np
from sklearn import preprocessing
from util import remove_centralized, timestamp2date, get_stock_name_by_id, date_2_day_index
import os.path

class TimeFormat:
    NONE = 0
    DAY = 1
    WEEK = 2

class DataManipulator:
    def __init__(self,  stock_name, stock_id, n_learning_days,
                n_prediction_days, beta, ema, time_format, volume_input, use_centralized_bid, 
                split_daily_data, ref_stock_id, input_path):
        self.n_learning_days = n_learning_days
        self.n_prediction_days = n_prediction_days
        self.beta = beta
        self.ema = ema
        self.time_format = time_format
        self.volume_input = volume_input
        self.use_centralized_bid = use_centralized_bid
        self.split_daily_data = split_daily_data
        self.last_learning_date = None
        self.next_prediction_seq = None
        self.next_learning_seq = None
        self.stock_name = stock_name
        self.stock_id = stock_id
        if ref_stock_id == stock_id:
            ref_stock_id = -1

        self.ref_stock_id = ref_stock_id
        if split_daily_data == True:
            self.n_learning_seqs = self.n_learning_days * 2
            self.n_prediction_seqs = self.n_prediction_days * 2
        else:
            self.n_learning_seqs = self.n_learning_days
            self.n_prediction_seqs = self.n_prediction_days
        
        self.input_path = input_path
        self.scaler_input = None
        self.scaler_output = None
        self.initialized = False


    # returns the training data's length in the historic data
    def get_training_data_len(self):
        assert(self.n_training_days != None)
        return self.n_training_days - self.n_learning_days

    def get_learning_seqs(self):
        return self.n_learning_seqs

    def get_prediction_seqs(self):
        return self.n_prediction_seqs

    def get_learning_days(self):
        return self.n_learning_days

    def get_prediction_days(self):
        return self.n_prediction_days

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
    
    
    def transform_input(self, data_input):
        return self.transform_helper(self.scaler_input, data_input)
    
    def transform_output(self, data_output):
        return self.transform_helper(self.scaler_output, data_output)
        
    def transform_helper(self, scaler, data):
        shape = data.shape
        data = data.reshape(shape[0]*shape[1],shape[2])
        data_scaled = scaler.transform(data)
        return data_scaled.reshape(shape)
    
    def init_scalers(self, start_day_index, end_day_index):
        input_path = self.input_path
        stock_id = self.stock_id
        input_data, output_data, timestamp, price = self.purge_data()
        start_seq_index = self.day_index_2_seq_index(start_day_index)
        end_seq_index = self.day_index_2_seq_index(end_day_index)
        self.scaler_input = self.build_scaler(input_data[start_seq_index:end_seq_index])
        self.scaler_output = self.build_scaler(output_data[start_seq_index:end_seq_index])
        self.initialized = True
        self.n_training_days = end_day_index - start_day_index
        return

    def build_scaler(self, data):
        shape = data.shape
        assert(len(shape)==3)
        data = data.reshape((shape[0]*shape[1],shape[2]))
        return preprocessing.MinMaxScaler().fit(data)



    def get_data_file_name(self, stock_id=None):
        if stock_id == None:
            stock_id = self.stock_id
            stock_name = self.stock_name
        else:
            stock_name = get_stock_name_by_id(stock_id)

        npy_file_name = os.path.join(self.input_path, 
                "{}_{}_ema{}_beta{}.npy".format(stock_name, stock_id, self.ema, self.beta))

        return npy_file_name


    def get_n_days(self):
        input_path = self.input_path
        stock_id = self.stock_id
        npy_file_name = self.get_data_file_name()
        input_np_data = np.load(npy_file_name, allow_pickle=True)
        n_days = input_np_data.shape[0]
        return n_days

    def purge_data_helper(self, input_np_data, ref_input_np_data):
        n_days = input_np_data.shape[0]
        # the diff is the mandatory
        input_columns = [2]
        
        time_format = self.time_format
        
        if time_format == TimeFormat.DAY:
            input_columns += [0]
        elif time_format == TimeFormat.WEEK:
            input_columns += [1]
        
        if self.volume_input == 1:
            input_columns += [3]
        

        input_data = input_np_data[:,:,input_columns]

        if ref_input_np_data is not None:
            assert(input_np_data.shape == ref_input_np_data.shape)
            print("ref_stock_id:{}".format(self.ref_stock_id))
            input_data = np.concatenate((input_data, ref_input_np_data[:,:,2:3]), axis=2)


        # we must tranform the volume for it is too big.
        if self.volume_input == 1:
            input_data = np.concatenate((input_data, input_np_data[:,:,3:4]), axis=2)
            input_data[:,:,-1] = self.volume_transform(input_data[:,:,-1])

        # output_data must be in shape (day, step, 1)
        output_data = input_np_data[:,:,4:5]
        timestamp = input_np_data[:,:,5]
        price = input_np_data[:,:,6]
        
        if self.use_centralized_bid == 0:
            # remove all the rows for centralized bid. it should be from 9.01 to 17.24, which is 516-12=504 steps
            input_data = remove_centralized(input_data)
            output_data = remove_centralized(output_data)
            timestamp = remove_centralized(timestamp)
            price = remove_centralized(price)
            
        input_data = self.daily_data_2_seq_data(input_data)
        output_data = self.daily_data_2_seq_data(output_data)
        timestamp = self.daily_data_2_seq_data(timestamp)
        price = self.daily_data_2_seq_data(price)
        
        return input_data, output_data, timestamp, price

    # to purge data based on parameters like time_input, split_daily_data, etc.
    def purge_data(self):
        input_path = self.input_path
        stock_id = self.stock_id
        # load numpy file
        npy_file_name = self.get_data_file_name()
        input_np_data = np.load(npy_file_name, allow_pickle=True)

        ref_input_np_data = None
        if self.ref_stock_id != -1:
            ref_npy_file_name = self.get_data_file_name(self.ref_stock_id)
            ref_input_np_data = np.load(ref_npy_file_name, allow_pickle=True)

        return self.purge_data_helper(input_np_data, ref_input_np_data)

    def purge_data_realtime(self, dataframe):
        input_np_data = dataframe.to_numpy()
        shape = input_np_data.shape
        assert(shape[0] == 516)
        assert(shape[1] == 7)
        input_np_data = input_np_data.reshape((1, shape[0], shape[1]))
        input_data, _, timestamp, price = self.purge_data_helper(input_np_data)
        scaled_input_data = self.transform_input(input_data)
        return scaled_input_data, timestamp, price


    
    def prep_data(self, start_day_index, end_day_index):
        assert(self.initialized)
        input_data, output_data, timestamp, price = self.purge_data()

        if end_day_index is None:
            end_day_index = n_days
        # to scale the data, but not the timestamp and price
        start = self.day_index_2_seq_index(start_day_index)
        end = self.day_index_2_seq_index(end_day_index)
        data_input = self.transform_input(input_data[start:end])
        data_output = self.transform_output(output_data[start:end])
        timestamp = timestamp[start:end]
        price = price[start:end]
        return data_input, data_output, timestamp, price

    # end date is the last day of the real time prediction day
    def get_last_price(self, end_date=None):
        npy_file_name = self.get_data_file_name()
        input_np_data = np.load(npy_file_name, allow_pickle=True)
        if end_date != None:
            day_index = self.date_2_day_index(end_date)
            assert(day_index != None and day_index > 0)
            price = input_np_data[day_index, -1, 6]
        else:
            price = input_np_data[-1,-1,6]

        return price

    # input: array : [timestamp, price, volume]
    def get_realtime_input_data(self, realtime_raw_data):
        # make sure that prepare realtime prediction is called
        assert(self.last_day_price != None)
        



    
    def date_2_day_index(self, date):
        npy_file_name = self.get_data_file_name()
        input_np_data = np.load(npy_file_name, allow_pickle=True)

        return date_2_day_index(input_np_data, date)



    def day_index_2_date(self, day_index):
        npy_file_name = self.get_data_file_name()
        input_np_data = np.load(npy_file_name, allow_pickle=True)
        return timestamp2date(input_np_data[day_index, 0, 5])


    def get_historic_day_index(self, date):
        day_index = self.date_2_day_index(date)
        assert(day_index is not None)
        assert(day_index > self.get_learning_days())
        return day_index - self.get_learning_days()

    def day_index_2_seq_index(self, day_index):
        if self.split_daily_data == True:
            return int(day_index * 2)
        else:
            return day_index

    def daily_data_2_seq_data(self, daily_data):
        if self.split_daily_data == False:
            return daily_data
        shape = daily_data.shape
        if len(shape) == 3:
            return daily_data.reshape((shape[0]*2, int(shape[1]/2), shape[2]))
        elif len(shape) == 2:
            return daily_data.reshape((shape[0]*2, int(shape[1]/2)))
        else:
            assert(False)

    def seq_data_2_daily_data(self, seq_data, is_remove_centralized=False):
        if self.split_daily_data == 0:
            daily_arr = seq_data
        else:
            shape = seq_data.shape
            n_days = int(shape[0] / 2)
            n_steps = shape[1] * 2
            if len(shape) == 2:
                daily_arr = seq_data.reshape((n_days, n_steps))    
            else:
                n_columns = shape[2]
                daily_arr = seq_data.reshape((n_days, n_steps, n_columns))

        # remove the centralized bid part of data.
        # from 9:01 to 17:24
        if is_remove_centralized == True:
            if daily_arr.shape[1] == 516:
                daily_arr = remove_centralized(daily_arr)

            assert(daily_arr.shape[1]==504)

        return daily_arr
