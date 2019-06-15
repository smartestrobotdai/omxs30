import sys
import numpy
import os.path
sys.path.append("../")
from util import *


if len(sys.argv) < 2:
	print("usage: python3 check-data.py stock_name")
	sys.exit()

ema=20
beta=99
stock_name = sys.argv[1]
stock_id = get_stock_id_by_name(stock_name)

filename = "../../preprocessed-data/{}_{}_ema{}_beta{}.npy".format(stock_name, stock_id, ema, beta)
data = np.load(filename, allow_pickle=True)
assert(len(data.shape) == 3)
n_days = len(data)

first_day = timestamp2date(data[0][0][5])
last_day = timestamp2date(data[-1][0][5])

print("Data Summary: ")
print("n_days: %d" % n_days)
print("first day: %s" % first_day)
print("last day: %s" % last_day)
