import sys
import os.path
sys.path.append("../")
from util import get_stock_id_by_name
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