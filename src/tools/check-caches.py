import sys
import os.path
sys.path.append("../")
from optimizeresult import OptimizeResult
from util import *


if len(sys.argv) < 5:
	print("usage: python3 search-worms.py stock_name, stock_index traning_start_day_index, training_end_day_index")
	sys.exit()

stock_name = sys.argv[1]
stock_index = int(sys.argv[2])
training_start_day_index = int(sys.argv[3])
training_end_day_index = int(sys.argv[4])

swarm_dir = get_swarm_dir(stock_name, stock_index, training_start_day_index, training_end_day_index)

strategy_file = os.path.join(swarm_dir, 'strategy_cache.txt')
result_strategies = OptimizeResult(-1)
result_strategies.load(strategy_file)
print("Top 10 Strategies in {} results for {}: swarm: {}-{}".format(result_strategies.get_size(), stock_name, training_start_day_index, training_end_day_index))

result_strategies.get_best_results(10)

strategy_file = os.path.join(swarm_dir, 'stockworm_cache.txt')
worm_results = OptimizeResult(-2)
worm_results.load(strategy_file)
print("Top 10 Worms in {} results for {}: swarm: {}-{}".format(worm_results.get_size(), stock_name, training_start_day_index, training_end_day_index))

worm_results.get_best_results(10)