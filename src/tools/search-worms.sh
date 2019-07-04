#!/bin/bash

if [ $# -lt 3 ]; then
  echo "usage: ./search-worms.sh stock_name start_day_index end_day_index"
  exit -1
fi

RUN=1
while [ 1 ]
do
  DATE=`date -u`
  echo "$DATE Starting $RUN runs"
  python3 -u ./search-worms.py $1 $2 $3 $4
  echo "$DATE Finishing $RUN runs"
  RUN=`expr $RUN + 1`
done
