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

if [ ! -z ${OMXS30_HOME} ]; then
  cd ${OMXS30_HOME}
  echo "changing directory to ${OMXS30_HOME}"
fi

echo "starting docker containers"
export PATH=$PATH:/usr/local/bin/:/usr/bin
docker-compose up -d
sleep 5

echo "`date` started fetching data"
echo "node version: `node -v`"
sleep 10
cd data-sink/
node daily.js
node minute.js
if [ $? -ne 0 ]; then
  echo "fetching data failed"
  cd $OMXS30_HOME
  exit -1
fi

cd $OMXS30_HOME
echo "`date` finished fetching data, backup database"
SUFFIX=`date +%d-%m-%Y"_"%H_%M_%S`
docker exec -t postgres-omxs pg_dumpall -c -U postgres > dbbackup/dump_$SUFFIX.sql

gzip dbbackup/dump_$SUFFIX.sql
cp -f dbbackup/dump_$SUFFIX.sql.gz dbbackup/dump.sql.gz
DATE=`date "+%y%m%d"`

if [ $UPLOAD_TO_GDRIVE -eq 1 ]; then
	echo "uploading dump.sql.gz"
	FILE_ID=`gdrive list | grep "dump.sql.gz" | awk '{print $1}'`
	gdrive update ${FILE_ID} dbbackup/dump.sql.gz
fi




PGPASSWORD=dai psql -h 0.0.0.0 -U postgres -f export_data.sql
docker cp postgres-omxs:/tmp/data.csv ./data
rm -rf data/data.csv.gz
gzip data/data.csv

if [ $UPLOAD_TO_GDRIVE -eq 1 ]; then
	echo "uploading data.csv.gz"
	FILE_ID=`gdrive list | grep "data.csv.gz" | awk '{print $1}'`
	gdrive update ${FILE_ID} data/data.csv.gz
fi



docker-compose down
sleep 5

cd src/tools
/usr/bin/python3 -u omx30-prep.py
cd ${OMXS30_HOME}

tar -czvf preprocessed-data/preprocessed-data.tar.gz ./preprocessed-data/*.npy

if [ $UPLOAD_TO_GDRIVE -eq 1 ]; then
	FILE_ID=`gdrive list | grep "preprocessed-data.tar.gz" | awk '{print $1}'`
	gdrive update ${FILE_ID} preprocessed-data/preprocessed-data.tar.gz
fi

cd ${ORI_DIR}
echo "`date` daily task finished"