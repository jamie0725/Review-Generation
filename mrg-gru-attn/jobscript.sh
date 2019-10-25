#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared_course 
#SBATCH --gres=gpu:2

module purge

module load 2019
module load Python/3.6.6-foss-2018b
module load cuDNN/7.6.3-CUDA-10.0.130  
module load NCCL/2.4.7-CUDA-10.0.130

cp -r $HOME/mrg-gru-attn $TMPDIR
cd $TMPDIR/mrg-gru-attn

virtualenv env -p `which python3`
source env/bin/activate
pip install -r requirements.txt

python train.py

cp log.txt $HOME/mrg-gru-attn
mkdir -p $HOME/mrg-gru-attn/results
cp -r tmp/* $HOME/mrg-gru-attn/results