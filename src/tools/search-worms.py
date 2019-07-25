import sys
import os.path
sys.path.append("../")
from stockwormmanager import StockWormManager
from util import *

if len(sys.argv) < 4:
	print("usage: python3 search-worms.py stock_name, start_day_index end_day_index [is_test]")
	sys.exit()

stock_name = sys.argv[1]
stock_index = get_stock_id_by_name(stock_name)
start_day_index = int(sys.argv[2])
end_day_index = int(sys.argv[3])
is_future = False
slippage = 0
if stock_name[:2] == "F-":
	print("target is a future")
	is_future = True
	slippage = 1

is_test = False
if len(sys.argv) == 5:
	is_test = bool(int(sys.argv[4]))

stock_data_dir = get_stock_data_dir()
preprocessed_data_dir = get_preprocessed_data_dir()

stock_worm_manager = StockWormManager(stock_name, stock_data_dir, 
	preprocessed_data_dir, is_future=is_future,
	slippage=slippage)

swarm_path = get_swarm_dir(stock_name, stock_index, start_day_index, end_day_index)

if not os.path.isdir(swarm_path):
	os.makedirs(swarm_path, exist_ok=True)

stock_worm_manager.search_worms(start_day_index, end_day_index, is_test=is_test)
