#!/bin/bash
#SBATCH --job-name=bloom-job        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=100G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:29:05          # total run time limit (HH:MM:SS)
#SBATCH --constraint=gpu80

module purge
module load anaconda3/2022.10
module load cudatoolkit/11.7
conda activate bloom


HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 nsys profile --trace=cuda,nvtx,osrt -o myprofile_bloom7b1_${SLURM_JOBID} python test_bloom_7b1.py