const {pgConfig} = require('./config')
const { Client } = require('pg')
const {executeQuery, getDateString} = require('./util')
const fs = require('fs')

async function loadStockInfo() {
  return await executeQuery(client, 'SELECT stock_id, stock_name FROM stocks')
}



async function exportStockDailyData(stockId, stockName) {
  let d = new Date();
  let dirName = `data/${getDateString(d)}`
  if (!fs.existsSync(dirName)){
      fs.mkdirSync(dirName);
  }
  var file = fs.createWriteStream(`${dirName}/${stockName}-${stockId}.txt`)

  let res = await executeQuery(client, `SELECT open, high, low, last, volume FROM daily WHERE stock_id='${stockId}' ORDER BY time_stamp`)
  for (let i = 0; i < res.rows.length; i++) {
    let {open, high, low, last, volume} = res.rows[i]
    file.write(`${open},${high},${low},${last},${volume}\n`)
  }
  file.end()
  return (res && res.rows && res.rows.length) || 0
}

async function loadStockData(res) {
  if (!res || !res.rows || !res.rows.length) {
    throw new Error('cannot fetch information from table stocks')
  }
  for (let i = 0; i < res.rows.length; i++) {
    let {stock_id, stock_name} = res.rows[i]
    console.log(`starting export daily data for stock: ${stock_name}:${stock_id}`)
    try {
      let count = await exportStockDailyData(stock_id, stock_name)
      if (count === 0) {
        console.log(`No data fetched for stock: ${stock_name}:${stock_id}`)
        continue
      }
    } catch (e) {
      throw e
    }
  }
}

const snooze = ms => new Promise(resolve => setTimeout(resolve, ms))

const client = new Client(pgConfig)
client.connect()
  .then(loadStockInfo)
  .then(loadStockData)
  .then(() => {
    return snooze(1000)
  })
  .then(() => {
    process.exit(0)
  })
  .catch(e => {
    console.log(e)
  })