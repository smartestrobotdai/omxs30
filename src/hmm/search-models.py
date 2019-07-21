import pandas as pd
import numpy as np
import GPy
import GPyOpt
import sys
import os.path
import os
sys.path.append("../")
from util import get_stock_id_by_name
from hmm_util import get_cache_filename
from functools import partial
from optimizeresult import OptimizeResult
from hmmmodel import HmmModel

def opt_func_base(stock_name, start_day_index, end_day_index, X_list):
	if len(X_list.shape) == 2:
		X_list = X_list[0]

	optimize_result = OptimizeResult()
	cache_filename = get_cache_filename(stock_name, start_day_index, end_day_index)

	if os.path.isfile(cache_filename):
	    optimize_result.load(cache_filename)
	    #print("stock worm cache loaded, size={}".format(optimize_result.get_size()))
	else:
	    print("cannot find file cache:{}, will create new cache.".format(cache_filename))
	print("Checking: {}".format(X_list))

	hmm_model = HmmModel(stock_name)
	total_profit, profit_daily_avg = hmm_model.train(X_list, start_day_index, end_day_index)
	
	if total_profit == -1:
		print("Training failed.")
		return total_profit

	print("Finished, total_profit:{}".format(total_profit))
	strategy_features = hmm_model.get_strategy_features()
	optimize_result.insert_result(X_list, strategy_features + [total_profit, profit_daily_avg])
	optimize_result.save(cache_filename)
	return profit_daily_avg


if len(sys.argv) < 4:
	print("usage: python3 search-models.py stock_name, start_day_index \
		end_day_index")
	sys.exit()

stock_name = sys.argv[1]
stock_index = get_stock_id_by_name(stock_name)
start_day_index = int(sys.argv[2])
end_day_index = int(sys.argv[3])

opt_func = partial(opt_func_base, stock_name, start_day_index, end_day_index)
mixed_domain = [{'name': 'n_components', 'type': 'discrete', 'domain': tuple(range(4, 24, 1))},
      {'name': 'ema', 'type': 'discrete', 'domain': (1,10,20)},
      {'name': 'beta', 'type': 'discrete', 'domain': (99,98)},
      {'name': 'use_volume', 'type': 'discrete', 'domain': (0,1)},
      {'name': 'ref_stock_id', 'type': 'discrete', 'domain': (-1,992,3524,139301,160271)},
      {'name': 'time_format', 'type': 'discrete', 'domain': (-1,0,1)},
     ]

opt_handler = GPyOpt.methods.BayesianOptimization(f=opt_func,  # Objective function       
                             domain=mixed_domain,           # Box-constraints of the problem
                             initial_design_numdata = 30,   # Number data initial design
                             acquisition_type='EI',        # Expected Improvement
                             exact_feval = True, 
                             maximize = True)           # True evaluations, no sample nois
opt_handler.run_optimization(max_iter=300, eps=0)


