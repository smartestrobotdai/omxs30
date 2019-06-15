import sys
import os.path
sys.path.append("../")
from stockwormmanager import StockWormManager
from util import *

if len(sys.argv) < 4:
	print("usage: python3 create-worms.py stock_name, start_day_index end_day_index [number=5]")
	sys.exit()

stock_name = sys.argv[1]
start_day_index = int(sys.argv[2])
end_day_index = int(sys.argv[3])

if len(sys.argv) == 5:
	number = int(sys.argv[4])
else:
	number = 5

stock_data_dir = get_stock_data_dir()
preprocessed_data_dir = get_preprocessed_data_dir()
stock_worm_manager = StockWormManager(stock_name, stock_data_dir, preprocessed_data_dir)
swarm_path = stock_worm_manager.get_swarm_path(start_day_index, end_day_index)
if not os.path.isdir(swarm_path):
	print("{} does not exist, aborting...".format(swarm_path))
	sys.exit()

stock_worm_manager.update_worms_from_cache(n_number=number, 
            						start_day_index=start_day_index, 
            						end_day_index=end_day_index)


stock_worm_manager.report()




