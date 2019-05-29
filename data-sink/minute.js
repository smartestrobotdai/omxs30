const {pgConfig} = require('./config')
const { Client } = require('pg')
const {isWorkingDay, executeQuery, getDateString, fetch, executingWrite} = require('./util')

async function checkData() {
  let res = await executeQuery(client, 'SELECT a.stock_id, a.stock_name, MAX(extract(epoch from time_stamp)) as time_stamp FROM stocks a \
      LEFT JOIN minute b ON a.stock_id = b.stock_id GROUP BY a.stock_id, a.stock_name')
  
  for (let i = 0; i < res.rows.length; i++) {
    let {stock_id, stock_name, time_stamp} = res.rows[i]
    console.log(`starting fetch min data for stock: ${stock_name}:${stock_id}, timestamp:${time_stamp}`)
    let results = await fetchMinData(stock_id, time_stamp)
    if (results === null || results.length === 0) {
      console.log(`No data fetched for stock: ${stock_name}:${stock_id}`)
      continue
    }
    let count = await insertRecords(stock_id, results)
    console.log(`minute data for stock: ${stock_name}: ${stock_id} finished, ${count} records were inserted`)
  }
  return Promise.resolve()
}

// timestamp in second
async function fetchMinData(stockId, timestamp) {
  let tsNow = (new Date).getTime();
  if (!timestamp) {
    // no timestamp, get last 100 days
    console.log('no timestamp found for stockId: ' +  stockId)
    timestamp = Date.now() - 7 * 3600 * 24 * 1000 
  } else {
    timestamp *= 1000 // convert to milliseconds
  }

  let results = []
  for (let d=timestamp; d <= tsNow; d += 3600*24*1000) {
    let dateString =  getDateString(new Date(d))
    newResults = await fetch(`https://www.nordnet.se/graph/instrument/11/${stockId}?from=${dateString}&to=${dateString}&fields=last,open,high,low,volume`)
    if (newResults && newResults.length) {
      results.push(...newResults)  
    }
    
  }
  console.log(`fetched data for stock:${stockId}, fetched ${results.length} records`)
  return results
  //console.log(`https://www.nordnet.se/graph/instrument/11/${stockId}?from=${startDay}&to=${today}&fields=last,open,high,low,volume`)  
  //return await fetch(`https://www.nordnet.se/graph/instrument/11/${stockId}?from=${startDay}&to=${today}&fields=last,open,high,low,volume`)
}

async function insertRecords(stockId, records) {
  let count = 0
  records.forEach(async function(record) {
    let {time, open, high, low, last, volume} = record
    let text = 'INSERT INTO minute(stock_id, time_stamp, open, high, low, last, volume) VALUES($1,to_timestamp($2),$3,$4,$5,$6,$7) \
         ON CONFLICT DO NOTHING'
    let value = [stockId, time/1000, open, high, low, last, Math.trunc(volume)]
    let res = await executingWrite(client, text, value)
    if (res.rowCount) {
      count += 1
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
    console.log('minute data inserteda')
    process.exit(0) 
  })

