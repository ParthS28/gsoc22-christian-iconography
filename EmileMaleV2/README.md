# EmileMaleV2

Version 2 of the Émile Mâle pipeline. This document explains the input format, how to use and output format of this version of the pipeline. To use this pipeline make sure that you have a directory of the name data/ with a subdirectory images/ which contains all the images. Labels of the images are not required.

## Input 

When running the pipeline, the data should be available in this format.

```
data/
|_____images/
|     |____xyz.jpg
|          .
|          .
```

images will contain all the images in .jpg format.

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
cp -r data/ emilemale/
cd emilemale
rsync -az <yourid>@rider.case.edu:/mnt/rds/redhen/gallina/home/pas193/EmileMaleV2/ .
rsync -az <yourid>@rider.case.edu:/mnt/rds/redhen/gallina/home/pas193/singularity/emilemalev1.sif .

singularity exec -e -B /mnt/rds/redhen/gallina/home/pas193/test/emilemale emilemalev1.sif ./run.sh

mv out1.csv ../.
mv out2/ ../out2
mv final.csv ../

cd ..
rm -rf emilemale

```

## Output

All three stages of the pipeline provide their output separately. 

out1.csv - output from the baseline classifier. CSV format with columns - item, predicted.

out2 - Directory containing all the annotations produced by YOLO. Text file and images included.

final.csv - Final output of the pipeline. CSV format with columns - item, stage1_prediction, final_prediction.
