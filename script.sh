#!/bin/bash

mkdir emilemale 
cd emilemale
rsync -az hp3:/mnt/rds/redhen/gallina/home/pas193/EmileMaleV1 .

./run.sh

mv out1.csv ../. 
mv out2/ ../out2
mv out3.csv ../

cd ..
rm -rf emilmale