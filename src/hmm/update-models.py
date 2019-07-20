import sys
import os.path
sys.path.append("../")
from stockwormmanager import StockWormManager
from util import *
from hmm_util import get_cache_filename, get_model_path
import pickle
from optimizeresult import OptimizeResult
from hmmmodel import HmmModel

if len(sys.argv) < 4:
	print("usage: python3 update-models.py stock_name, start_day_index end_day_index [number=1]")
	sys.exit()

stock_name = sys.argv[1]
start_day_index = int(sys.argv[2])
end_day_index = int(sys.argv[3])

if len(sys.argv) == 5:
	number = int(sys.argv[4])
else:
	number = 1

cache_file = get_cache_filename(stock_name, start_day_index, end_day_index)
optimize_result = OptimizeResult()
optimize_result.load(cache_file)
top_worms = optimize_result.get_best_results(number)
for i in range(len(top_worms)):
	features = top_worms[i,:6]
	strategy_features = top_worms[i,6:12]
	md5 = top_worms[i,-1]
	model_path = get_model_path(stock_name, start_day_index, end_day_index)
	save_path = os.path.join(model_path, md5)
	hmm_model_filename = os.path.join(save_path, "hmm_model.pkl")
	hmm_model = None
	if os.path.isfile(hmm_model_filename):
		print("file: {} exists, load it.".format(hmm_model_filename))
		with open(hmm_model_filename, "rb") as file: 
			hmm_model = pickle.load(file)

	else:
		print("file: {} does not exists, generate it.".format(hmm_model_filename))
		hmm_model = HmmModel(stock_name)
		print("Updating model: {}".format(features))
		total_profit = hmm_model.train(features, start_day_index, end_day_index, strategy_features)
		print("Model trained, total profit: {}".format(total_profit))
		hmm_model.save(save_path)

	# TODO: test the model 
	test_start_day_index = end_day_index
	total_profit = hmm_model.test(test_start_day_index)
	hmm_model.save(save_path)

	stock_profit_overnight = get_stock_change_rate(stock_name, test_start_day_index, overnight=True)
	stock_profit = get_stock_change_rate(stock_name, test_start_day_index, overnight=False)
	print("Stock profit w/o overnight: {}".format(stock_profit))
	print("Stock profit overnight: {}".format(stock_profit_overnight))
