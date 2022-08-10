#!/bin/bash

python3 stage1.py
echo stage 1 done

cd YOLOv6
python3 tools/infer.py --yaml data/dataset.yaml --img-size 224 --weights runs/train/exp1/weights/best_ckpt.pt --source ../data/images
mkdir ../out2
cp -r runs/inference/exp/ ../out2/
cd ..
echo stage 2 done

python3 stage3.py
rm -rf YOLOv6/runs/inference/exp/
echo task completed