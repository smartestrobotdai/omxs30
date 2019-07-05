from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import cgi
import psycopg2
import sys
import os.path
sys.path.append("../")
from util import *
from datetime import datetime, date
import time

def get_data(conn, days):
  start_timestamp = None
  start_datetime = None
  cursor = conn.cursor()
  if days == 1:
    # To get the start time for 1 day
    
    query = "SELECT time_stamp FROM minute ORDER BY time_stamp DESC LIMIT 1"
    cursor.execute(query)
    latest_time = cursor.fetchall()[0][0]

    start_datetime = datetime(latest_time.year, latest_time.month, latest_time.day, 0, 0)
    start_timestamp = str(int(start_datetime.timestamp()))
  elif days == 7:
    t = datetime.fromtimestamp(time.time() - 7*24*3600)
    start_datetime = datetime(t.year, t.month, t.day, 0, 0)
    start_timestamp = str(int(start_datetime.timestamp()))
  elif days == 31:
    t = datetime.fromtimestamp(time.time() - 31*24*3600)
    start_datetime = datetime(t.year, t.month, t.day, 0, 0)
    start_timestamp = str(int(start_datetime.timestamp()))

  query = "SELECT stock_id, time_stamp, transaction FROM transactions WHERE time_stamp > to_timestamp({})".format(start_timestamp)
  cursor.execute(query)
  transactions = cursor.fetchall()
  stock_ids = list(map(lambda x: int(x[0]), transactions))
  stock_ids = list(dict.fromkeys(stock_ids))
  stock_ids = ','.join('\'' + str(e) + '\'' for e in stock_ids + [0])
  stock_ids =  '(' + stock_ids + ')'

  query = "SELECT time_stamp, stock_id, last FROM minute WHERE stock_id IN {} AND time_stamp > to_timestamp(%s)".format(stock_ids)
  cursor.execute(query, (str(start_timestamp),))
  stock_data = cursor.fetchall() 

  df = pd.DataFrame(stock_data, columns=['timestamp','stock_id','price'])
  df['timestamp'] = df['timestamp'].dt.tz_convert('Europe/Stockholm')

  groups = df.groupby('stock_id')

  all_df = {}
  for stock_id, df in groups:
    all_df[stock_id] = None
    df['day'] = df['timestamp'].apply(lambda x: "{}-{:02d}-{:02d}".format(x.year, x.month, x.day))
    groups_day = df.groupby('day')
    
    for day, df_day in groups_day:
      start_ts = df_day['timestamp'].iloc[0]
      open_price = df_day['price'].iloc[0]
      
      if start_ts.hour != 9 or start_ts.minute != 0:
        print("Stock: {} day: {} started at: {}".format(stock_id, day, start_ts))
        time_str = "{}-{}-{} 09:00:00".format(start_ts.year, start_ts.month, start_ts.day)
        
        ts_9 = pd.Timestamp(time_str, tz='Europe/Stockholm')
        print(df_day.columns.tolist())
        new_record = pd.DataFrame([[ts_9, stock_id, open_price, day]], columns=df_day.columns.tolist())
        df_day = pd.concat([new_record, df_day], axis=0)
        print(df_day)

      all_df[stock_id] = pd.concat([all_df[stock_id], df_day], axis=0)
        
    
    all_df[stock_id]['diff'] = all_df[stock_id]['price'].diff().fillna(0) / all_df[stock_id]['price'].shift(-1)
    all_df[stock_id]['diff'] = all_df[stock_id]['diff'].fillna(0)
    all_df[stock_id] = all_df[stock_id].set_index('timestamp')
      
  omxs = all_df['0']

  def get_ts_at_beginning(t):
    timestr = "{}-{}-{} 09:00:00".format(t.year, t.month, t.day)
    return pd.Timestamp(timestr, tz='Europe/Stockholm')
      

  grouped = all_df['0'].groupby(get_ts_at_beginning)

  result_df=None
  for key, df in grouped:
    dti = pd.date_range(key, periods=511, freq='min').to_series(keep_tz=True)
    if result_df is None:
      result_df = pd.DataFrame(dti, columns=['timestamp']).reset_index(drop=True)
    else:
      new_df = pd.DataFrame(dti, columns=['timestamp']).reset_index(drop=True)
      result_df = result_df.append(new_df)

  result_df['diff'] = 0
  result_df = result_df.set_index('timestamp')

  trans_arr = []
  for i in range(0, len(transactions), 2):
    stock_id = transactions[i][0]
    buy_time = transactions[i][1]
    sell_time = transactions[i+1][1]
    assert(transactions[i][0] == transactions[i+1][0])
    assert(transactions[i][2] == 'buy')
    assert(transactions[i+1][2] == 'sell')
    trans_arr.append([stock_id, buy_time, sell_time])

  for tran in trans_arr:
      stock_id = tran[0]
      start_time = tran[1]
      end_time = tran[2]
      index_in_range = (all_df[stock_id].index > start_time) & (all_df[stock_id].index <= end_time)
      diff = all_df[stock_id].loc[index_in_range, 'diff']
      
      result_df.loc[diff.index, 'diff'] = diff

  result_df['diff'] = result_df['diff'].fillna(0)
  result_df['profit'] = (result_df['diff'] + 1).cumprod()
  omxs['profit'] = (omxs['diff'] + 1).cumprod()

  # remove unecessary records.
  def should_remove(t):
      ts = t.timestamp()
      if days == 1:
          minutes = 1
          return False
      elif days == 7:
          minutes = 5
      elif days == 31:
          minutes = 60
      return bool(ts % (minutes*60) != 0)


  def add_column_should_remove(df):
      if 'should_remove' in df.columns:
          return
      df['timestamp'] = df.index
      df['should_remove'] = df['timestamp'].apply(should_remove)
      return df

  result_df = add_column_should_remove(result_df)
  omxs = add_column_should_remove(omxs)

  omxs_filtered = omxs[omxs.should_remove == False][['profit']]
  result_filtered = result_df[result_df.should_remove == False][['profit']]

  joined = result_filtered.join(omxs_filtered, how='left', rsuffix='_omxs', lsuffix='_result').interpolate(method='linear')
  joined.reset_index(inplace=True)
  joined['time'] = joined['timestamp'].apply(lambda x: x.timestamp())
  joined = joined.drop('timestamp', axis=1)
  return joined.values.tolist()

def MakeHandlerClass(conn):
  class Server(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
      print("init")
      self.conn = conn
      super(Server, self).__init__(*args, **kwargs)
      

    def _set_headers(self):
      self.send_response(200)
      self.send_header('Content-type', 'application/json')
      self.end_headers()
        
    def do_HEAD(self):
      self._set_headers()
        
    # GET sends back a Hello world message
    def do_GET(self):
      self._set_headers()
      data = get_data(self.conn, 7)
      string = json.dumps(data)

      self.wfile.write(string.encode(encoding='utf-8'))
          
    # POST echoes the message adding a JSON field
    def do_POST(self):
      ctype, pdict = cgi.parse_header(self.headers.getheader('content-type'))
      
      # refuse to receive non-json content
      if ctype != 'application/json':
        self.send_response(400)
        self.end_headers()
        return
          
      # read the message and convert it into a python dictionary
      length = int(self.headers.getheader('content-length'))
      message = json.loads(self.rfile.read(length))
      
      # add a property to the object, just to mess with data
      message['received'] = 'ok'
      
      # send the message back
      self._set_headers()
      self.wfile.write(json.dumps(message))
  return Server
        
def run(server_class=HTTPServer, port=8008):
  try:
    conn = connect_postgres()
  except (Exception, psycopg2.DatabaseError) as error:
    print(error)
    sys.exit()

  server_address = ('', port)
  handler_class = MakeHandlerClass(conn)
  httpd = server_class(server_address, handler_class)
  
  print('Starting httpd on port %d...' % port)
  httpd.serve_forever()
    

if __name__ == "__main__":
  from sys import argv
  
  if len(argv) == 2:
    run(port=int(argv[1]))
  else:
    run()