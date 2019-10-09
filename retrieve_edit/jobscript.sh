#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:10:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared_course 
#SBATCH --gres=gpu:1

module purge

module load 2019
module load Python/2.7.15-foss-2018b
module load cuDNN/7.6.3-CUDA-10.0.130  
module load NCCL/2.4.7-CUDA-10.0.130

cd $TMPDIR

virtualenv env -p `which python2`
source env/bin/activate

git clone --single-branch --branch retrieve_edit https://github.com/s2948044/Review-Generation.git
cd Review-Generation/retrieve_edit
pip install -r requirements.txt

wget -O datasets.tar.gz https://worksheets.codalab.org/rest/bundles/0xfa69890526c04899a1eb286afb17d37a/contents/blob/
mkdir -p datasets
tar -C datasets -zxvf datasets.tar.gz
wget -O word_vectors.tar.gz https://worksheets.codalab.org/rest/bundles/0x512544e3af5c4a738dd6e57e02d0e4ba/contents/blob/
mkdir -p word_vectors
tar -C word_vectors -zxvf word_vectors.tar.gz
ls
export COPY_EDIT_DATA=$(pwd); export PYTHONIOENCODING=utf8; cd cond-editor-codalab; python train_ctx_vae.py

deactivate
