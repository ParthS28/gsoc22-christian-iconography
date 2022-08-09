# EmileMaleV1

Version 1 of the Emile Male pipeline. This document explains the input format, how to use and output format of this version of the pipeline. This version of the pipeline is not recommended to be used as it comes with the data already stored in the files could be hassle to transfer data, instead check version 2.

## Input 

When running the pipeline, the data should be available in this format.

```
data/
|_____images/
|     |____xyz.jpg
|          .
|          .
|_____info.csv
```

images will contain all the images in .jpg format and info.csv contains the data for the images(if data is not available please check version 2 of the pipeline).

## How to run

The pipeline will be executed in the singularity format. 
```
singularity pull emilemalev1.sif docker://ghcr.io/parths28/emilemale:latest
```

After that run the script
```
#!/bin/bash

module load singularity/3.8.1

mkdir emilemale
cd emilemale
rsync -az pas193@rider.case.edu:/mnt/rds/redhen/gallina/home/pas193/EmileMaleV1/ .
rsync -az pas193@rider.case.edu:/mnt/rds/redhen/gallina/home/pas193/singularity/emilemalev1.sif .

singularity exec -e -B /mnt/rds/redhen/gallina/home/pas193/test/emilemale emilemalev1.sif ./run.sh

mv out1.csv ../.
mv out2/ ../out2
mv final.csv ../

cd ..
rm -rf emilemale
```

## Output

All three stages of the pipeline provide their output separately. 

out1.csv - output from the baseline classifier. CSV format with columns - item, label, predicted.

out2 - Directory containing all the annotations produced by YOLO. Text file and images included.

final.csv - Final output of the pipeline. CSV format with columns - item, label, final_prediction.
