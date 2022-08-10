#!/bin/bash

module load singularity/3.8.1

mkdir emilemale
cp -r data/ emilemale/
cd emilemale
rsync -az pas193@rider.case.edu:/mnt/rds/redhen/gallina/home/pas193/EmileMaleV2/ .
rsync -az pas193@rider.case.edu:/mnt/rds/redhen/gallina/home/pas193/singularity/emilemalev1.sif .

singularity exec -e -B /mnt/rds/redhen/gallina/home/pas193/test/emilemale emilemalev1.sif ./run.sh

mv out1.csv ../.
mv out2/ ../out2
mv final.csv ../

cd ..
rm -rf emilemale

