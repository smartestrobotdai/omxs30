#!/bin/bash

# exit when any command fails
set -e


# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG

# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

ORI_DIR=`pwd`
DIR=`dirname $0`


if [ $(date +%u) -gt 5 ]; then
   echo "weekend, aborting..."
   exit 0
fi

if [ ! -z ${DIR} ]; then
  cd ${DIR}
  echo "changing directory to ${DIR}"
fi

echo "starting docker containers"
docker-compose up -d
sleep 5

echo "`date` started fetching data"
echo "node version: `node -v`"
export PATH=$PATH:/usr/local/bin/:/usr/bin
docker-compose up -d
sleep 10
cd data-sink/
node daily.js
node minute.js
if [ $? -ne 0 ]; then
  echo "fetching data failed"
  cd $DIR
  exit -1
fi

cd $DIR
echo "`date` finished fetching data, backup database"
SUFFIX=`date +%d-%m-%Y"_"%H_%M_%S`
docker exec -t postgres-omxs pg_dumpall -c -U postgres > dbbackup/dump_$SUFFIX.sql

gzip dbbackup/dump_$SUFFIX.sql
cp -f dbbackup/dump_$SUFFIX.sql.gz dbbackup/dump.sql.gz
DATE=`date "+%y%m%d"`

FILE_ID=`gdrive list | grep "dump.sql.gz" | awk '{print $1}'`

gdrive update ${FILE_ID} dbbackup/dump.sql.gz
PGPASSWORD=dai psql -h 0.0.0.0 -U postgres -f export_data.sql
docker cp postgres-omxs:/tmp/data.csv ./data
rm -rf data/data.csv.gz
gzip data/data.csv

FILE_ID=`gdrive list | grep "data.csv.gz" | awk '{print $1}'`

gdrive update ${FILE_ID} data/data.csv.gz
echo "`date` daily task finished"

docker-compose down
sleep 5

cd src/tools
/usr/bin/python3 -u omx30-prep.py
cd ${DIR}

tar -czvf preprocessed-data/preprocessed-data.tar.gz ./preprocessed-data/*.npy
FILE_ID=`gdrive list | grep "preprocessed-data.tar.gz" | awk '{print $1}'`
gdrive update ${FILE_ID} preprocessed-data/preprocessed-data.tar.gz
cd ${ORI_DIR}
