import sys
import os.path
sys.path.append("../")
from stockworm import StockWorm
from util import *
import psycopg2
import asyncio
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrConnectionClosed, ErrTimeout, ErrNoServers

# note: time must be in seconds like 1562169308

state = 0
async def run(loop, stock_worm, stock_id):
	# connect to NATS
	try:
		print("Connecting to NATS...")
		nc = NATS()
		await nc.connect("localhost:4222", loop=loop)
		print("Connected to NATS...")

		# connect to postgresql
		print("Connecting to PostgreSQL...")
		conn = psycopg2.connect(host="localhost",database="postgres", user="postgres", password="dai")
		print('PostgreSQL database version:')
		cur = conn.cursor()
		cur.execute('SELECT version()')

		# display the PostgreSQL database server version
		db_version = cur.fetchone()
		print(db_version)

		print("Connected to PostgreSQL...")

		#test insertion.
		#print("Write data to PostgreSQL...")
		#await write_transaction(conn, stock_id, 1562169308, "buy")
		#print("Wrote data to PostgreSQL...")
	except (Exception, psycopg2.DatabaseError) as error:
	  print(error)
	  sys.exit()

	async def message_handler(msg):
		global state
		subject = msg.subject
		reply = msg.reply
		data = msg.data.decode()
		print("Received a message on '{subject} {reply}': {data}".format(
				subject=subject, reply=reply, data=data))

		realtime_data_array = json.load(data)
		for item in realtime_data_array:
			value, action = stock_worm.test_realtime(item.time, item.last, item.volume)
			if action == 1 and state == 0:
				print("time:{} buy stock {} at price: {}".format(item.time, stock_id, item.last))
				await write_transaction(conn, stock_id, item.time, "buy")
				state = 1
			elif action == -1 and state == 1:
				print("time:{} sell stock {} at price: {}".format(item.time, stock_id, item.last))
				await write_transaction(conn, stock_id, item.time, "sell")
				state = 0


	sid = await nc.subscribe("instument-{}".format(stock_id), cb=message_handler)

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print("usage: python3 run-model.py stock_name, model_dir")
		sys.exit()

	stock_name = sys.argv[1]
	model_dir = sys.argv[2]
	npy_path = get_preprocessed_data_dir()
	stock_id = get_stock_id_by_name(stock_name)

	stock_worm = StockWorm(stock_name, stock_id, npy_path, model_dir)
	stock_worm.load()
	stock_worm.test()
	stock_worm.start_realtime_prediction()

	loop = asyncio.get_event_loop()
	loop.create_task(run(loop, stock_worm, stock_id))
	loop.run_forever()
	loop.close()
