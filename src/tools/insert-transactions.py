import sys
import os.path
sys.path.append("../")
from util import *
from stockworm import StockWorm
from pathlib import Path
import asyncio



async def run(conn, test_data, stock_name, stock_id):
	buy_count = 0
	sell_count = 0
	for i in range(len(test_data)):
		time = test_data[i, 0]
		action = test_data[i, 5]
		price = test_data[i, 2]

		if action == 1:

			print("time: {} stock_name: {} buy at price: {}".format(time, stock_name, price))
			await write_transaction(conn, stock_id, int(time.timestamp()), 'buy')
			buy_count += 1

		elif action == -1:
			print("time: {} stock_name: {} sell at price: {}".format(time, stock_name, price))
			await write_transaction(conn, stock_id, int(time.timestamp()), 'sell')
			sell_count += 1

	print("insert transactions finished: buy_count: {}, sell_count: {}".format(buy_count, sell_count))

if len(sys.argv) < 3:
	print("usage: python3 insert-transactions.py stock_name, model_dir")
	sys.exit()

try:
	conn = connect_postgres()
except (Exception, psycopg2.DatabaseError) as error:
  print(error)
  sys.exit()

stock_name = sys.argv[1]
model_dir = sys.argv[2]
npy_path = get_preprocessed_data_dir()
stock_id = get_stock_id_by_name(stock_name)
stock_worm = StockWorm(stock_name, stock_id, npy_path, model_dir)
stock_worm.load()
stock_worm.test()
stock_worm.report()
test_data = stock_worm.get_test_data()
shape = test_data.shape
print(shape)
test_data = test_data.reshape(shape[0]*shape[1], shape[2])

loop = asyncio.get_event_loop()
loop.run_until_complete(run(conn, test_data, stock_name, stock_id))




