import hashlib
import os.path
import numpy as np
import statistics

def md5(in_str):
	m = hashlib.sha256()
	b = bytearray(in_str, 'utf-8')
	m.update(b)
	return m.hexdigest()


def remove_centralized(data):
	assert(data.shape[1]==516)
	return data[:,7:-5]

def add_centralized(data):
	assert(data.shape[1]==504)
	data_before = np.zeros((data.shape[0], 7))
	data_after = np.zeros((data.shape[0], 5))
	data = np.concatenate((data_before, data, data_after), axis=1)
	assert(data.shape[1]==516)
	return data


def timestamp2date(timestamp):
	return timestamp.date().strftime("%y%m%d")

def get_latest_dir(save_path):
    all_subdirs = [d for d in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, d))]
    max_time = 0
    result = None
    for dirname in all_subdirs:
        fullname = os.path.join(save_path, dirname)
        time = os.path.getmtime(fullname)
        if time > max_time:
            max_time = time
            result = dirname
    return result

def get_earliest_dir(save_path):
    all_subdirs = [d for d in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, d))]
    max_time = 0
    for dirname in all_subdirs:
        fullname = os.path.join(save_path, dirname)
        time = os.path.getmtime(fullname)
        if time > max_time:
            max_time = time
            result = dirname
    return result

def get_home_dir():
	home_dir = os.environ['OMXS30_HOME']
	assert(home_dir is not None)
	return home_dir

def get_stock_data_dir():
	return os.path.join(get_home_dir(), 'stock-data')

def get_preprocessed_data_dir():
	return os.path.join(get_home_dir(), 'preprocessed-data')

def get_swarm_dir(stock_name, stock_index, start_day_index, end_day_index):
    return os.path.join(get_stock_data_dir(), 
        "{}_{}".format(stock_name, stock_index),
        "{}-{}".format(start_day_index, end_day_index))

def load_strategy_input_data(stock_index, start_day_index, end_day_index, ema=20, beta=99):
	processed_data_dir = get_preprocessed_data_dir()
	file = os.path.join(processed_data_dir, "ema{}_beta{}_{}.npy".format(ema, beta, stock_index))
	data = np.load(file, allow_pickle=True)
	return remove_centralized(data[start_day_index:end_day_index,:,[-2,-3,-1]])

def mean(list_data):
	return statistics.mean(list_data)

def stdev(list_data):
	return statistics.stdev(list_data)

def create_if_not_exist(path):
    if not os.path.isdir(path):
        os.mkdir(path)