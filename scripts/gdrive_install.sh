#!/bin/bash
set -e


# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG

# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT


wget "https://drive.google.com/uc?id=1Ej8VgsW5RgK66Btb9p74tSdHMH3p4UNb&export=download"
mv 'uc?id=1Ej8VgsW5RgK66Btb9p74tSdHMH3p4UNb&export=download' gdrive
chmod +x gdrive

mv gdrive ../bin/
../bin/gdrive about


