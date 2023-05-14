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


HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python test_bloom_7b1.py
# deepspeed --num_gpus 4 transformers-bloom-inference/bloom-inference-scripts/bloom-ds-inference.py --name microsoft/bloom-deepspeed-inference-int8 --dtype int8
# python transformers-bloom-inference/bloom-inference-scripts/bloom-accelerate-inference.py --name bigscience/bloom --dtype int8 --batch_size 1 --benchmark 2>&1 | tee bloom-int8-accelerate-inference_bs=1.txt
# python bloom-accelerate-inference.py --name bigscience/bloom --dtype int8 --batch_size 1 --benchmark 2>&1 | tee bloom-int8-accelerate-inference_bs=1.txt