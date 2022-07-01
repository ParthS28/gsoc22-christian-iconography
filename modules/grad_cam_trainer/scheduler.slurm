#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH -p gpu -C gpup100 --gres=gpu:1 --mem=8gb

unset PYTHONPATH
module swap intel gcc
module load python/3.6.6
rsync -az hpc3:/home/pas193/pas193gallinahome/gradcam /tmp/$USER/gradcam_trainer
python3 -m venv venv 
source venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
python trainer.py
deactivate
rsync -az hp3:/tmp/$USER/gradcam_trainer /home/pas193/pas193gallinahome/gradcam_results