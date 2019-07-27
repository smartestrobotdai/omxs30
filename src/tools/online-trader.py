import sys
import os.path
sys.path.append("../")
from stockworm import StockWorm
from util import *
import psycopg2
import asyncio
import json
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrConnectionClosed, ErrTimeout, ErrNoServers
import warnings
# note: time must be in seconds like 1562169308


async def run(loop, stock_worm, stock_id, ref_stock_id):
  state = 0
  n_steps = 0
  ref_data_arr = []
  data_arr = []
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

  def copulas_check_length(data_arr, ref_data_arr):
    if len(data_arr) == 0 or len(ref_data_arr) == 0:
      return 0

    n = min(len(data_arr), len(ref_data_arr))
    assert(data_arr[n-1]['time'] == ref_data_arr[n-1]['time'])
    return n

  async def handle_realtime_data(data_arr, ref_data_arr):
    nonlocal state
    nonlocal n_steps
    for i in range(len(data_arr)):
      item = data_arr[i]
      ref_item = ref_data_arr[i]
      print("trying to call test_realtime: step: {}".format(n_steps))
      print(item)
      print(ref_item)
      try:
        value, action = stock_worm.test_realtime(item['time'], item['last'], item['volume'], ref_item['last'])
        if action[-1] == 1 and state == 0:
          print("time:{} buy stock {} at price: {}".format(item['time'], stock_id, item['last']))
          await write_transaction(conn, stock_id, item['time'], "buy")
          state = 1
        elif action[-1] == -1 and state == 1:
          print("time:{} sell stock {} at price: {}".format(item['time'], stock_id, item['last']))
          await write_transaction(conn, stock_id, item.time, "sell")
          state = 0
      except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(e)
        raise e
      n_steps+=1
    print("values:")
    print(value[:n_steps])

    print("actions:")
    print(action[:n_steps])



  def get_stock_id(subject):
    print(subject.split('.')[1])
    return int(subject.split('.')[1])

  def fix_data_arr(data_arr, start_time):
    new_data_arr = []
    t = start_time
    n = 0
    first = data_arr[0]
    while t < data_arr[0]['time']:
      new_data_arr.append({'time': t, 'high':first['high'], 'last':first['last'], 'low':first['low'], 'open': first['open'], 'volume':0})
      n += 1
      t+= 60000
    
    last = data_arr[0]
    new_data_arr.append(last)
    for item in data_arr[1:]:
      time = int(item['time'])
      t = last['time'] + 60000
      while t < time:
        new_data_arr.append({'time': t, 'high':last['high'], 'last':last['last'], 'low':last['low'], 'open': last['open'], 'volume':0})
        n += 1
        t += 60000
      new_data_arr.append(item)
      last = item
    return new_data_arr, n



  async def message_handler(msg):
    nonlocal data_arr
    nonlocal ref_data_arr
    try:
      subject = msg.subject
      stock_id_from_msg = get_stock_id(subject)
      print("received msg, stock_id:{}".format(stock_id))
      reply = msg.reply
      data = msg.data.decode()
      print("data decoded")

      print("Received a message on '{subject} {reply}': {data}".format(
          subject=subject, reply=reply, data=data))

      if stock_id_from_msg == stock_id:
        data_arr = data_arr + json.loads(data)
      elif stock_id_from_msg == ref_stock_id:
        ref_data_arr = ref_data_arr + json.loads(data)
      else:
        print("Wrong message.")

      if len(data_arr) == 0 or len(ref_data_arr) == 0:
        return

      start_time = min(data_arr[0]['time'], ref_data_arr[0]['time'])
      
      print("data_arr size: {}, ref_data_arr size: {}".format(len(data_arr), len(ref_data_arr)))


      data_arr, n = fix_data_arr(data_arr, start_time)
      print("added {} missing entries into data".format(n))
      ref_data_arr, n = fix_data_arr(ref_data_arr, start_time)
      print("added {} missing entries into ref data".format(n))

      n_to_handle = copulas_check_length(data_arr, ref_data_arr)
      print("n_to_handle")
      print(n_to_handle)

      if n_to_handle > 0:
        await handle_realtime_data(data_arr[:n_to_handle], ref_data_arr[:n_to_handle])
      
        data_arr = data_arr[n_to_handle:]
        ref_data_arr = ref_data_arr[n_to_handle:]
        print("data_arr size: {}, ref_data_arr size: {}".format(len(data_arr), len(ref_data_arr)))

    except Exception as e:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      print(exc_type, fname, exc_tb.tb_lineno)
      print(e)
      raise e

  print("starting monitor stock_id: {}".format(stock_id))
  await nc.subscribe("instument.>", cb=message_handler)


def custom_exception_handler(loop, context):
  # first, handle with default handler
  loop.default_exception_handler(context)

  exception = context.get('exception')
  print(exception)
  print(context)
  loop.stop()

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

  ref_stock_id = stock_worm.get_ref_stock_id()
  stock_worm.start_realtime_prediction()
  #value, action = stock_worm.test_realtime("1564124460000", 812.9, 1193, 62.79)
  #value, action = stock_worm.test_realtime("1564124520000", 813.9, 1194, 62.80)

  loop = asyncio.get_event_loop()
  loop.set_debug(True)
  # Report all mistakes managing asynchronous resources.
  warnings.simplefilter('always', ResourceWarning)
  loop.set_exception_handler(custom_exception_handler)
  loop.create_task(run(loop, stock_worm, stock_id, ref_stock_id))
  loop.run_forever()
  loop.close()
