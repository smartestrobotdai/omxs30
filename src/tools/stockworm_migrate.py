import sys
import os.path
sys.path.append("../")
from util import *
from optimizeresult import OptimizeResult
from pathlib import Path

stock_data_path = get_stock_data_dir()

# This is to get the directory that the program  
# is currently running in. 

for filename in Path(stock_data_path).glob('**/stockworm_cache.txt'):
	
	optimize_result = OptimizeResult(result_column_index=-2)
	optimize_result.load(filename)
	if (optimize_result.get_n_columns() == 18):
		print("{} has been already migrated.".format(filename))
		continue

	print('migrating {}'.format(filename))
	assert(optimize_result.get_n_columns() == 17)
	optimize_result.add_column(13, 1.0)
	assert(optimize_result.get_n_columns() == 18)
	optimize_result.save(filename)


