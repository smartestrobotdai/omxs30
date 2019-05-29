const { remote } = require('webdriverio');
const TEST = 1

function delay(t, val) {
   return new Promise(function(resolve) {
       setTimeout(function() {
           resolve(val);
       }, t);
   });
}

function getValidResponse(body, expectTimestamp) {
  let parsed = JSON.parse(body)
  for (let i = 0; i < parsed.length; i++) {
    if (parsed[i].time === expectTimestamp.getTime()) {
      return parsed.slice(0,i)
    }
  }
  return null
}

function getDate(ts) {
  return `${ts.getFullYear()}-${ts.getMonth()+1}-${ts.getDate()}`
}


let responseBodyCache = {}
async function fetch(expectTimestamp, browser, resolve, retries) {
    console.log('fetching ' +  expectTimestamp)
    let date = getDate(expectTimestamp)

    let result = null
    if (date in responseBodyCache) {
      console.log(`${date} fetched from cache`)
      result = responseBodyCache[date]
    } else {
      result = await browser.executeAsync((stockId, date, done) => {
        function get(oReq, url, callback) {
          oReq.open("GET", url)
          oReq.onreadystatechange = function(e) {
              console.log(oReq.readyState)
              if (oReq.readyState === 4) {
                  console.log('price')
                  callback(oReq.responseText)
              }
          }
          oReq.onerror = function() {
              console.log('error!')
              
          }
          oReq.send()
        }


        let url = `graph/instrument/11/${stockId}/?from=${date}&to=${date}`
        let oReq = new XMLHttpRequest()
        get(oReq, url, body => {
          // TODO:  check the timestamp
          done(body)
        })
      }, 101, date)
    }

    console.log('current time after fetching:' +  Date.now())
    // console.log(result)
    // check the timestamp
    // let's check the time
    let day = Math.trunc(expectTimestamp.getTime() / 24*3600*1000)
    let today = Math.trunc(Date.now() / 24*3600*1000)

    // when the date of wanted ts less than today, obviously it is a testing.
    if (day < today && !(date in responseBodyCache)) {
      console.log(`date: ${date} saved to cache`)
      console.log(result)
      responseBodyCache[date] = result
      
    }

    let ret = getValidResponse(result, expectTimestamp)
    if (ret === null) {
      if (retries < 3) {
        console.log(`failed to fetch data at ${expectTimestamp}, retries: ${retries}`)
        await fetch(expectTimestamp, browser, resolve, retries+1)
      } else {
        resolve({'code': -1})
      }
    }
    resolve({'code': 0, 'info': ret}) 
}


async function getPriceDelayed(browser, waitTime, expectTimestamp) {
  return new Promise((resolve,reject) => {
    console.log('current time1:' +  Date.now())
    setTimeout(fetch, waitTime, expectTimestamp, browser, resolve, reject, 0)
  })
}

// TODO:
// https://www.nordnet.se/mux/ajax/marknaden/aktiehemsidan/orderdjup.html
/*
<h2 antal="1">Orderdjup

<div class="iconTank">
    <a onclick="new Ajax.Updater('orderdjup', '/mux/ajax/marknaden/aktiehemsidan/orderdjup.html', {parameters:'identifier=100&marketplace=11&orderdjupsantal=1&country=Sverige'}); return false" href="#"><img src="/now/images/knapp_refresh2.gif" alt="Uppdatera" title="Uppdatera" /></a>
</div>

</h2>






<table class="borders">
    <thead>
        <tr>
            <th><a href="#" style="text-decoration:none;color:#018acf;" onclick="return openHjalpOrd( 'orderdjup_antal', event)">Antal</a></th>
            <th><a href="#" style="text-decoration:none;color:#018acf;" onclick="return openHjalpOrd( 'orderdjup_pris', event)">Pris</a></th>
            <th class="buy"><a href="#" style="text-decoration:none;color:#018acf;" onclick="return openHjalpOrd( 'orderdjup_k%f6p', event)">Köp</a></th>
            <th class="sell"><a href="#" style="text-decoration:none;color:#018acf;" onclick="return openHjalpOrd( 'orderdjup_s%e4lj', event)">Sälj</a></th>
            <th><a href="#" style="text-decoration:none;color:#018acf;" onclick="return openHjalpOrd( 'orderdjup_pris', event)">Pris</a></th>
            <th><a href="#" style="text-decoration:none;color:#018acf;" onclick="return openHjalpOrd( 'orderdjup_antal', event)">Antal</a></th>
        </tr>
    </thead>
    <tr>
        <td class="first">558</td>
            <td>95,80</td>
        <td colspan="2" class="orderdjupStaplar">
            <div class="stapelContainer">
                <div class="stapelUpp" style="width:58px;"><!-- IE friendly --></div>
                <div class="stapelNer" style="width:78px;"><!-- IE friendly --></div>
            </div>
        </td>
            <td>95,80</td>
        <td class="last">749</td>
    </tr>

</table>


const $ = cheerio.load(html)
for (i=0; i<2; i++) {
    console.log(i)
    const row = $(`.borders tr:nth-of-type(${i+1})`)
    //console.log(firstRow)
    const vol = $(row).find('td:nth-of-type(1)').text()
    console.log(vol)
    const prise = $(row).find('td:nth-of-type(2)').text()
    console.log(prise)
}




*/


// at 8:59:01 of the start date to the end 
// 

function addSeconds(timestamp, seconds) {
  return new Date(timestamp.getTime() + seconds*1000)
}


function *getExpectedTimestamp(startDate, endDate) {

  let startDayOpenTime = new Date(startDate + ' 9:00:00')
  let endDayOpenTime = new Date(endDate + ' 9:00:00')
  let startTime = startDayOpenTime

  while (startTime <= endDayOpenTime) {
    // endTime is 17:24:00
    let endTime1724 = addSeconds(startTime, 30240)
    let ts = startTime
    while (ts <= endTime1724) {
      yield ts
      ts = addSeconds(ts, 60)
    }

    //  
    let endTime1729 = addSeconds(endTime1724 ,5 * 60)

    yield endTime1729

    startTime = addSeconds(startTime, 3600 * 24)
  }
}

(async () => {
/*
    var browser = await remote({
        logLevel: 'trace',
        host: '0.0.0.0',
        port: 4444,
        //path: '/', // only for firefox
        capabilities: {
            browserName: 'chrome'
        }
    });
*/
    
    var browser = await remote({
        logLevel: 'warn',
        host: 'localhost',
        port: 4321,
        //path: '/', // only for firefox
        capabilities: {
            browserName: 'chrome'
        }
    });
    

    /*
    var browser = await remote({
        logLevel: 'info',
        host: 'localhost',
        port: 4445,
        path: '/', // only for firefox
        capabilities: {
            browserName: 'firefox'
        }
    });
    */
    await browser.url('https://www.nordnet.se/start.html')
    await delay(2000)
    await browser.setTimeout({ 'script': 60000 });

    let generator = getExpectedTimestamp('2019-04-29','2019-04-30')
    let res = generator.next()
    for(let res = generator.next(); res.done === false; res = generator.next()) {
      let priceInfo = await getPriceDelayed(browser, 0, res.value)
      if (priceInfo.code === -1) {
        // there is no data for this minute, we must report it to model as well.
        console.log(`${res.value}: no data found.`)
        continue
      }

      let info = priceInfo.info
      // print out the last one.
      console.log(info.slice(info.length-1, info.length))
      
    }
    /*
    while (1) {
      // check if today is working day.
      // 

      // get time delta to next minute.

      msNow = Date.now()
      msToNextMinute = 60000 - msNow % 60000
      //msToNextMinute -= 20
      console.log(`current time: ${msNow}, wait: ${msToNextMinute}`)
      let result = await getPriceDelayed(browser, msToNextMinute, msNow+msToNextMinute)
                          .catch(async e => {
                            console.log('error')
                            await delay(10000)
                          })
    }
    */

    /*
    const link = await browser.$('=Logga in')
    //console.log(link)
    text = await link.getText()
    console.log(text)

    await link.click()
    await delay(500)

    // find out the Mobilt bankid
    const mobiltBankId = await browser.$('.button-1-0.primary-1-6.size-m-1-19.block-1-1')
    console.log(await mobiltBankId.getText())

    await mobiltBankId.click()
    await delay(500)

    const id = await browser.$('.text-input.id-number-input')
    await id.setValue('197705103815')
    await delay(500)

    const ok = await browser.$('.ok.large-button.button')
    await ok.click()

    await delay(10000)
   // TODO: login

    */


    const title = await browser.getTitle();
    console.log('Title was: ' + title);
    
    await browser.deleteSession();
})().catch((e) => console.error(e));
