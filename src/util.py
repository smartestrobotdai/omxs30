import hashlib
import os.path
import numpy as np
import statistics

stock_map = [(3966,"ABB"),
	(18634,"ALFA"),
	(402,"ASSA-B"),
	(3524,"AZN"),
	(45,"ATCO-A"),
	(46,"ATCO-B"),
	(47,"ALIV-SDB"),
	(15285,"BOL"),
	(81,"ELUX-B"),
	(101,"ERIC-B"),
	(139301,"ESSITY-B"),
	(812,"GETI-B"),
	(992,"HM-B"),
	(812,"GETI-B"),
	(819,"HEXA-B"),
	(161,"INVE-B"),
	(999,"KINV-B"),
	(160271,"NDA-SE"),
	(4928,"SAND"),
	(323,"SCA-B"),
	(281,"SEB-A"),
	(401,"SECU-B"),
	(340,"SHB-A"),
	(283,"SKA-B"),
	(285,"SKF-B"),
	(300,"SSAB-A"),
	(120,"SWED-A"),
	(361,"SWMA"),
	(1027,"TEL2-B"),
	(5095,"TELIA"),
	(366,"VOLV-B"),
	(0,"OMXS30")
]

def get_stock_name_by_id(stock_id):
	for item in stock_map:
		if item[0] == stock_id:
			return item[1]
	return None

def get_stock_id_by_name(stock_name):
	for item in stock_map:
		if item[1] == stock_name:
			return item[0]
	return None

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
        "()_()".format(stock_name, stock_index),
        "()-()".format(start_day_index, end_day_index))

def load_strategy_input_data(stock_index, start_day_index, end_day_index, ema=20, beta=99):
	processed_data_dir = get_preprocessed_data_dir()
	file = os.path.join(processed_data_dir, "ema()_beta()_().npy".format(ema, beta, stock_index))
	data = np.load(file, allow_pickle=True)
	return remove_centralized(data[start_day_index:end_day_index,:,[-2,-3,-1]])

def mean(list_data):
	return statistics.mean(list_data)

def stdev(list_data):
	return statistics.stdev(list_data)

def create_if_not_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path)