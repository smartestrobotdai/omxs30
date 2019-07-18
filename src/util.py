import hashlib
import os.path
import numpy as np
import pandas as pd
import statistics
import time
import psycopg2

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
	return m.hexdigest()[:16]


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


def get_current_date():
	strDate = time.strftime('%y%m%d')
	return strDate

def date_2_day_index(data, date):
	timestamps = data[:,:,5]
	for i in range(len(timestamps)):
	    if timestamp2date(timestamps[i][0]) == date:
	        return i
	return None

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

def get_npy_filename(stock_name, stock_id, ema, beta):
	preprocess_data_dir = get_preprocessed_data_dir()
	return os.path.join(preprocess_data_dir, "{}_{}_ema{}_beta{}.npy".format(stock_name, stock_id, ema, beta))

def get_preprocessed_data_dir():
	return os.path.join(get_home_dir(), 'preprocessed-data')

def get_swarm_dir(stock_name, stock_id, start_day_index, end_day_index):
    return os.path.join(get_stock_data_dir(), 
        "{}_{}".format(stock_name, stock_id),
        "{}-{}".format(start_day_index, end_day_index))

def load_strategy_input_data(stock_id, start_day_index, end_day_index, ema=20, beta=99):
	processed_data_dir = get_preprocessed_data_dir()
	stock_name = get_stock_name_by_id(stock_id)
	file = os.path.join(processed_data_dir, "{}_{}_ema{}_beta{}.npy".format(stock_name, stock_id, ema, beta))
	data = np.load(file, allow_pickle=True)
	return remove_centralized(data[start_day_index:end_day_index,:,[-2,-3,-1]])

def mean(list_data):
	return statistics.mean(list_data)

def stdev(list_data):
	return statistics.stdev(list_data)

def create_if_not_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path)

# input: df - dataframe, must have timestamp, 
def preprocessing_daily_data(df, last_close=None, calculate_values=True):
	# if the data is not from 9:00:00, we must fix it.
	open_today = df['last'].iloc[0]
	# some data might miss, we must make a right join with full time series
	# and do fillna.

	df2 = df.set_index('timestamp')
	ts = df2.index.min()
	start_time_str = "{}-{:02d}-{:02d} 8:54:00".format(ts.year, ts.month, ts.day)
	start_ts = pd.Timestamp(start_time_str, tz=ts.tz)
	# periods=510 means from 9 to 17.29
	dti = pd.date_range(start_ts , periods=516, freq='min').to_series(keep_tz=True).rename('time')
	# remove from 17.25 - 17.28
	#dti.drop(dti.tail(5).head(4).index, inplace=True)
	df3 = df2.join(dti, how='right')
	if last_close == None: # the first day, we must set the value from 8.55-8.59 as same as 9.00
	    df3['last'].iloc[0] = df3['last'].iloc[6]
	else:
	    df3['last'].iloc[0] = last_close


	# if the daily data is not started from 9:00:00, we must set it as the open price.
	if df3['last'].iloc[6] != open_today:
		df3['last'].iloc[6] = open_today


	df3['last'].interpolate(method='linear', inplace=True)
	df3['volume'].iloc[:6] = df3['volume'].iloc[6] / 6
	df3['volume'].iloc[6] = df3['volume'].iloc[6] / 6
	df3['volume'].iloc[-5:] = df3['volume'].iloc[-1]/5

	#TODO: FIXED ME, no nan in volume!


	df = df3.reset_index().rename({'index':'timestamp'}, axis=1)

	#df['timestamp'] = pd.to_datetime(df['time_stamp'], format="%Y-%m-%d %H:%M:%S").dt.tz_convert('Europe/Stockholm')
	df['ema_1'] = df['last']
	df['ema_5'] = df['last'].ewm(span=5, adjust=False).mean()
	df['ema_10'] = df['last'].ewm(span=10, adjust=False).mean()
	df['ema_20'] = df['last'].ewm(span=20, adjust=False).mean()
	df['diff_ema_1']=(df['ema_1'].diff()[1:]/df['ema_1']).fillna(0)
	df['diff_ema_5']=(df['ema_5'].diff()[1:]/df['ema_5']).fillna(0)
	df['diff_ema_10']=(df['ema_10'].diff()[1:]/df['ema_10']).fillna(0)
	df['diff_ema_20']=(df['ema_20'].diff()[1:]/df['ema_20']).fillna(0)

	df['volume'] = df['volume'].abs().fillna(0)

	if calculate_values == True:
		# the first diff at 9:00 is the difference between today's open and yesterday's last.
		df['value_ema_1_beta_99'] = 0
		df['value_ema_5_beta_99'] = 0
		df['value_ema_10_beta_99'] = 0
		df['value_ema_20_beta_99'] = 0
		df['value_ema_1_beta_98'] = 0
		df['value_ema_5_beta_98'] = 0
		df['value_ema_10_beta_98'] = 0
		df['value_ema_20_beta_98'] = 0
		for iter_id in range(20):
		    df['value_ema_1_beta_99'] = df['diff_ema_1'].shift(-1).fillna(0) +                 0.99 * df['value_ema_1_beta_99'].shift(-1).fillna(0)
		    df['value_ema_1_beta_98'] = df['diff_ema_1'].shift(-1).fillna(0) +                 0.98 * df['value_ema_1_beta_98'].shift(-1).fillna(0)
		    df['value_ema_5_beta_99'] = df['diff_ema_5'].shift(-1).fillna(0) +                 0.99 * df['value_ema_5_beta_99'].shift(-1).fillna(0)
		    df['value_ema_5_beta_98'] = df['diff_ema_5'].shift(-1).fillna(0) +                 0.98 * df['value_ema_5_beta_98'].shift(-1).fillna(0)
		    df['value_ema_10_beta_99'] = df['diff_ema_10'].shift(-1).fillna(0) +                 0.99 * df['value_ema_10_beta_99'].shift(-1).fillna(0)
		    df['value_ema_10_beta_98'] = df['diff_ema_10'].shift(-1).fillna(0) +                 0.98 * df['value_ema_10_beta_98'].shift(-1).fillna(0)
		    df['value_ema_20_beta_99'] = df['diff_ema_20'].shift(-1).fillna(0) +                 0.99 * df['value_ema_20_beta_99'].shift(-1).fillna(0)
		    df['value_ema_20_beta_98'] = df['diff_ema_20'].shift(-1).fillna(0) +                 0.98 * df['value_ema_20_beta_98'].shift(-1).fillna(0)

	return df

def datetime_2_timestamp(dt):
	return pd.Timestamp(dt, tz='Europe/Stockholm')

def add_step_columns(df):
    df['step_of_day'] = np.arange(0, len(df))
    day_of_week = df['timestamp'].iloc[0].weekday()
    df['step_of_week'] = len(df) * day_of_week + df['step_of_day']
    return df

async def write_transaction(conn, stock_id, time, transaction):
	try:
		postgres_insert_query = """ INSERT INTO transactions (stock_id, time_stamp, transaction) VALUES (%s,to_timestamp(%s),%s)"""
		cursor = conn.cursor()
		cursor.execute(postgres_insert_query, (stock_id, time, transaction))
		conn.commit()
	except (Exception, psycopg2.Error) as error :
		print(error)

def connect_postgres():
	print("Connecting to PostgreSQL...")
	conn = psycopg2.connect(host="localhost",database="postgres", user="postgres", password="dai")
	print('PostgreSQL database version:')
	cur = conn.cursor()
	cur.execute('SELECT version()')

	# display the PostgreSQL database server version
	db_version = cur.fetchone()
	print(db_version)

	print("Connected to PostgreSQL...")
	return conn