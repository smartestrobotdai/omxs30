import sys
import os.path
import numpy as np
sys.path.append("../")
from util import get_stock_id_by_name, get_npy_filename
from optimizeresult import OptimizeResult
from hmm_util import get_cache_filename

if len(sys.argv) < 4:
  print("usage: python3 search-worms.py stock_name, traning_start_day_index, training_end_day_index number")
  sys.exit()

stock_name = sys.argv[1]
stock_id = get_stock_id_by_name(stock_name)
start_day_index = int(sys.argv[2])
end_day_index = int(sys.argv[3])
number = 5

if len(sys.argv) == 5:
  number = int(sys.argv[4])

# find out the total raise.
npy_filename = get_npy_filename(stock_name, stock_id, 1, 99)
data = np.load(npy_filename, allow_pickle=True)
start_price = data[0,0,6]
end_price = data[-1,-1,6]
# column 6 is the price!
n_days = data.shape[0]
profit = 1
for d in range(n_days):
	price = data[d,:,6]
	#print("start: {} - end: {}".format(price[0], price[-2]))
	rate = price[-2] / price[0]
	profit = profit * rate

print("Stock profit w/o overnight: {}".format(profit))
print("Stock profit w overnight: {}".format(end_price/start_price))

cache_file = get_cache_filename(stock_name, start_day_index, end_day_index)
worm_results = OptimizeResult(-1)
worm_results.load(cache_file)
print("Top 10 Worms in {} results for {}: swarm: {}-{}".format(worm_results.get_size(), 
	stock_name, start_day_index, end_day_index))

worm_results.get_best_results(number)
columns = ['n_components', 'ema', 'beta', 'use_volume', 'ref_stock_id', 
          'time_format', 'buy_threshold', 'sell_threshold', 'stop_loss', 'stop_gain', 'skip_at_beginning', 'value_ma', 
					'total_profit']
print("Columns:")
for i in range(len(columns)):
  print("{}: {}".format(i, columns[i]))