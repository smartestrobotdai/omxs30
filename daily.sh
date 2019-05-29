#!/bin/bash
ORI_DIR=`pwd`
DIR=`dirname $0`
LOG_FILE="$DIR/nightly.log"

if [ $(date +%u) -gt 5 ]; then
   echo "weekend, aborting...">>${LOG_FILE}
   exit 0
fi

if [ ! -z ${DIR} ]; then
  cd ${DIR}
  echo "changing directory to ${DIR}">>${LOG_FILE}
fi

echo "starting docker containers"
docker-compose up -d
sleep 5

echo "`date` started fetching data">>${LOG_FILE}
echo "node version: `node -v`">>${LOG_FILE}
export PATH=$PATH:/usr/local/bin/:/usr/bin
docker-compose up -d
sleep 10
cd data-sink/
node daily.js>>${LOG_FILE} 2>&1
node minute.js>>${LOG_FILE} 2>&1
if [ $? -ne 0 ]; then
  echo "fetching data failed">>${LOG_FILE}
  cd $DIR
  exit -1
fi

cd $DIR
echo "`date` finished fetching data, backup database">>${LOG_FILE}
SUFFIX=`date +%d-%m-%Y"_"%H_%M_%S`
docker exec -t postgres-omxs pg_dumpall -c -U postgres > dbbackup/dump_$SUFFIX.sql

gzip dbbackup/dump_$SUFFIX.sql
cp -f dbbackup/dump_$SUFFIX.sql.gz dbbackup/dump.sql.gz
#git add dbbackup/dump.sql.gz

PGPASSWORD=dai psql -h 0.0.0.0 -U postgres -f export_data.sql
docker cp postgres-omxs:/tmp/data.csv ./data
rm -rf data/data.csv.gz
gzip data/data.csv
#git add data/data.csv.gz
#git commit -m "new backup and new data"
#git push
echo "`date` daily task finished">>${LOG_FILE}

docker-compose down
sleep 5

cd src/tools
/usr/bin/python3 -u omx30-prep.py >> ${LOG_FILE}
if [ $? -ne 0 ]; then
  echo "preprocess data failed">>${LOG_FILE}
  cd $DIR
  exit -1
fi

cd ${DIR}
#git add preprocessed-data/*.npy
#git commit -m "new preprocessed-data"
#git push


cd ${ORI_DIR}
