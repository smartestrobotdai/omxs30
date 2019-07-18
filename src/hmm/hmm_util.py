import sys
import os.path
sys.path.append("../")
from util import create_if_not_exist

def get_cache_path(stock_name, start_day_index, end_day_index):
  path2 = "{}-{}".format(start_day_index, end_day_index)
  create_if_not_exist(os.path.join(stock_name, path2))
  return os.path.join(stock_name, path2)

def get_cache_filename(stock_name, start_day_index, end_day_index):
	path = get_cache_path(stock_name, start_day_index, end_day_index)
	return os.path.join(path, "model_cache.txt")

def get_model_path(stock_name, start_day_index, end_day_index):
	path = get_cache_path(stock_name, start_day_index, end_day_index)
	return os.path.join(path, "models")
