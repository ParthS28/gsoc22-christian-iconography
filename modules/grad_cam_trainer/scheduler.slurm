#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH -p gpu -C gpup100 --gres=gpu:1 --mem=8gb

echo hello
module swap intel gcc
module load python/3.8.10
rm -rf /tmp/$USER
rsync -az hpc3:/mnt/rds/redhen/gallina/home/pas193/gradcam/ /tmp/$USER/

cd /tmp/$USER
realpath .
mkdir models
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
python3 trainer.py
deactivate
rsync -az /tmp/$USER/models hpc3:/mnt/rds/redhen/gallina/home/pas193/gradcam_results/
echo done
rm -rf .
