#!/bin/bash

FILE_NAME=preprocessed-data.tar.gz
FILE_ID=`gdrive list | grep "preprocessed-data.tar.gz" | awk '{print $1}'`

gdrive download ${FILE_ID}
tar -zxvf ${FILE_NAME} > /dev/null
rm -rf ../preprocessed-data
mv ./preprocessed-data ../
rm -rf ${FILE_NAME}
echo "preprocessed data downloaded successfully"
