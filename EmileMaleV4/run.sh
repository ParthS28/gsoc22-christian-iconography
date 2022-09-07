#!/bin/bash

source venv/bin/activate
python3 stage1.py
echo stage 1 done

cd YOLOv6
python3 tools/infer.py --yaml data/dataset.yaml --img-size 224 --weights runs/train/exp2/weights/best_ckpt.pt --source ../data/images
mkdir ../out2
cp -r runs/inference/exp/ ../out2/
mkdir out2
cp -r runs/inference/exp/ out2/
cd ..
echo stage 2 done

python3 stage3.py
rm -rf YOLOv6/runs/inference/exp/

python3 stage4.py

echo task completed
deactivate