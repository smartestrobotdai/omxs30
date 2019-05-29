const {pgConfig} = require('./config')
const { Client } = require('pg')
const {isWorkingDay, executeQuery, getDateString, fetch, executingWrite} = require('./util')

async function checkData() {
  let res = await executeQuery(client, 'SELECT a.stock_id, a.stock_name, MAX(extract(epoch from time_stamp)) as time_stamp FROM stocks a \
      LEFT JOIN daily b ON a.stock_id = b.stock_id GROUP BY a.stock_id, a.stock_name')
  
  for (let i = 0; i < res.rows.length; i++) {
    let {stock_id, stock_name, time_stamp} = res.rows[i]
    console.log(`starting fetch daily data for stock: ${stock_name}:${stock_id}`)
    try {
      let results = await fetchDailyData(stock_id, time_stamp)
      if (results === null || results.length === 0) {
        console.log(`No data fetched for stock: ${stock_name}:${stock_id}`)
        continue
      }
      await insertRecord(stock_id, results)
      console.log(`daily data for stock: ${stock_name}: ${stock_id} finished`)
    } catch (e) {
      throw e
    }
  }
  return Promise.resolve()
}

// timestamp in second
async function fetchDailyData(stockId, timestamp) {
  let now = Date.now()	
  let d1 = new Date(now);
  if (!timestamp) {
    // no timestamp, get last 100 days
    console.log('no timestamp found for stockId: ' +  stockId)
    timestamp = Date.now() - 3650 * 3600 * 24 * 1000 
  } else {
    // dont want to fetch minute data
    //timestamp *= 1000 // convert to milliseconds
    timestamp = Math.min(now - 15 * 3600 * 24 *1000, timestamp * 1000)
  }
  let d2 = new Date(timestamp)
  let today = getDateString(d1)
  let startDay = getDateString(d2)
  console.log(`https://www.nordnet.se/graph/instrument/11/${stockId}?from=${startDay}&to=${today}&fields=last,open,high,low,volume`)  
  return await fetch(`https://www.nordnet.se/graph/instrument/11/${stockId}?from=${startDay}&to=${today}&fields=last,open,high,low,volume`)
}

async function insertRecord(stockId, records) {
  let count = 0
  records.forEach(async function(record) {
    let {time, open, high, low, last, volume} = record
    let text = 'INSERT INTO daily(stock_id, time_stamp, open, high, low, last, volume) VALUES($1,to_timestamp($2),$3,$4,$5,$6,$7) \
         ON CONFLICT DO NOTHING'
    let value = [stockId, time/1000, open, high, low, last, Math.trunc(volume)]
    let res = await executingWrite(client, text, value)
    if (res.rowCount) {
      count+=1
    } 
  })
  console.log(`${count} records have been inserted`)
  return count
}

const snooze = ms => new Promise(resolve => setTimeout(resolve, ms));

const client = new Client(pgConfig)
client.connect()
  .then(checkData)
  .then(() => {
    return snooze(120000)
  })
  .then(() => {
    process.exit(0)
  })

