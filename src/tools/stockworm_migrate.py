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
	print(optimize_result.get_n_columns())
	if (optimize_result.get_n_columns() == 24):
		print("{} has been already migrated.".format(filename))
		continue

	print('migrating {}'.format(filename))
	assert(optimize_result.get_n_columns() == 23)
	optimize_result.add_column(14, -1)
	assert(optimize_result.get_n_columns() == 24)
	optimize_result.save(filename)


