#!/bin/bash

module load singularity/3.8.1

mkdir emilemale
cp -r data/ emilemale/
cd emilemale
rsync -az pas193@rider.case.edu:/mnt/rds/redhen/gallina/home/pas193/EmileMaleV4/ .
rsync -az pas193@rider.case.edu:/mnt/rds/redhen/gallina/home/pas193/singularity/emilemale.sif .

singularity exec -e -B /mnt/rds/redhen/gallina/home/pas193/test4/emilemale emilemale.sif ./run.sh

mv out1.csv ../.
mv out2/ ../out2
mv out3.csv ../
mv class_prediction.csv ../

cd ..
rm -rf emilemale

