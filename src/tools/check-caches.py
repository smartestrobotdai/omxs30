import sys
import os.path
sys.path.append("../")
from optimizeresult import OptimizeResult
from util import *

if len(sys.argv) < 4:
	print("usage: python3 search-worms.py stock_name, traning_start_day_index, training_end_day_index number")
	sys.exit()

stock_name = sys.argv[1]
stock_id = get_stock_id_by_name(stock_name)
training_start_day_index = int(sys.argv[2])
training_end_day_index = int(sys.argv[3])
number = 10
if len(sys.argv) == 5:
	number = int(sys.argv[4])

swarm_dir = get_swarm_dir(stock_name, stock_id, training_start_day_index, training_end_day_index)

strategy_file = os.path.join(swarm_dir, 'stockworm_cache.txt')
worm_results = OptimizeResult(-1)
worm_results.load(strategy_file)
print("Top 10 Worms in {} results for {}: swarm: {}-{}".format(worm_results.get_size(), stock_name, training_start_day_index, training_end_day_index))

worm_results.get_best_results(number)
columns = ['n_neurons', 'learning_rate', 'num_layers', 'rnn_type', 'learning_period', 
					'prediction_period', 'n_repeats', 'beta', 'ema', 'time_format', 'volume_input', 'use_centralized_bid', 
					'split_daily_data', 'is_stateful', 'ref_stock_id', 'buy_threshold', 'sell_threshold', 'stop_gain', 'stop_gain', 'skip_at_beginning',
					'total_profit', 'days', 'avg_error', 'avg_profit']
print("Columns:")
for i in range(len(columns)):
	print("{}: {}".format(i, columns[i]))

