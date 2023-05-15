# How to run Bloom on Della

## Setup

A conda virtual environment named 'bloom' will be created with the needed dependencies:

```shell
conda env create -f environment.yml
conda activate bloom
```

## bloom-7b1
This is a small model(~10GB) and can fit into one GPU: https://huggingface.co/bigscience/bloom-7b1.

### Download the model
Specify the path where you want to store the model in line 3 and 4 in `download_model.py`. To download the model, run this on the home node as the computing nodes do not have Internet access:

```shell
python download_model.py
```

### Test

Change line 8 and 9 in `test_bloom_7b1.py` to the path your specified in `download_model.py`. You can modify the prompt in line 12 and 13 of `test_bloom_7b1.py`.

Run bloom-7b1 on one GPU with:
```shell
sbatch test_bloom_7b1.slurm
```
Check the slurm output for the result.

## bloom-int8
This is the quantized version of the original bloom-176B version: https://huggingface.co/microsoft/bloom-deepspeed-inference-int8. The model is around 170GB and can fit into 4 A100s (80GB) on Della. As Della doesn't have nodes with 8 A100s, this is the largest model we can run within one node.

### Download the model

The downloading is a bit tricky. First, set a soft link to link the `~/.cache` path to a path with enough disc space, e.g., `gpfs`, because `~/.cache` is where the model will be automatically stored. 

Run this on `della-gpu` to download the model:

```shell
deepspeed --num_gpus 1 bloom-ds-inference.py --name microsoft/bloom-deepspeed-inference-int8 --dtype int8 --benchmark
```

It may fail due to Internet connection timeout. Try several times until succeed. If you have downloaded successfully, the script will throw out a CUDA out-of-memory error. That's because we are on the home node and only have one GPU. The model should be stored inside `~/.cache/huggingface/hub/models--microsoft--bloom-deepspeed-inference-int8`. You may see some `temp` files at `~/.cache/huggingface/hub` if you have encountered Internet connection error during downloading. Feel free to remove them.

### Test

To test the model on 4 GPUs, run
```shell
sbatch test_bloom_ds.slurm
```

This should take ~10 minutes.

## Profile with Nsight System

To profile, run
```shell
sbatch test_bloom_7b1_profile.slurm
```

which will generate a trace file `myprofile_bloom7b1_${SLURM_JOBID}`.

Go to https://developer.nvidia.com/gameworksdownload#?dn=nsight-systems-2023-2 to download and install the profiler on your laptop. I use `Nsight Systems 2023.2.1 (macOS Host)`. Download the trace and open it with Nsight System.

You can also use nsys-ui to view the results directly on the machine: https://github.com/PrincetonUniversity/gpu_programming_intro/blob/master/04_gpu_tools/README.md.
