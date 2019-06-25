const https = require('https');

function getDateString(date) {
  return `${date.getFullYear()}-${date.getMonth()+1}-${date.getDate()}`
}

async function fetch(url) {
  return new Promise((resolve, reject) => {
    https.get(url, resp => {
      let data = ''
      // A chunk of data has been recieved.
      resp.on('data', (chunk) => {
        data += chunk;
      })
      // The whole response has been received. Print out the result.
      resp.on('end', () => {
        try {
          let daily = JSON.parse(data)    
          resolve(daily)
        } catch(e) {
          console.log(`parse data failed: ${e}, ${url}:${data}`)
          resolve(null)
        }
      })
    }).on("error", (err) => {
      console.log("Error: " + err.message + " " + data);
      reject()
    })
  })
}

function getCompletedRecord(record) {
  let {time, open, high, low, last, volume} = record

  open=open?open:0
  high=high?high:0
  low=low?low:0
  last=last?last:0
  volume=volume?volume:0

  return {time, open, high, low, last, volume}
}

function getDataUrl(stockId, from, to) {
  url = ''
  if (stockId === '0') {
    url = `https://www.nordnet.se/graph/indicator/SSE/OMXSPI?from=${from}&to=${to}&fields=last,open,high,low`
  } else {
    url = `https://www.nordnet.se/graph/instrument/11/${stockId}?from=${from}&to=${to}&fields=last,open,high,low,volume`
  }
  return url
}

async function executeQuery(client, sql) {
  return new Promise(resolve => {
    client.query(sql, (err, res) => {
      if (err) {
        console.log(`executing ${sql} failed: ${err}`)
        throw new Error(err)
      }
      resolve(res)
    })
  })
}

async function executingWrite(client, sql, value) {
  return new Promise(resolve => {
    client.query(sql, value, (err, res) => {
      if (err) {
        console.log(`executing ${sql}, ${value} failed, err: ${err}`)
        throw new Error(err)
      } 
      resolve(res)   
    })
  })
}

function calEMA(oldVal, newVal, days) {
  let multiplier = 2 / (days + 1)
  return (newVal - oldVal) * multiplier + oldVal
}

exports.isWorkingDay = (ts) => {
  if (ts === null) {
    d = new Date()
  } else {
    d = new Date(ts)
  }

  let day = d.getDay()
  return day !== 6 && day !== 0
}

exports.fetch = fetch
exports.executingWrite = executingWrite
exports.executeQuery = executeQuery
exports.getDateString = getDateString
exports.getDataUrl = getDataUrl
exports.getCompletedRecord = getCompletedRecord
