# Calcul Québec AI Benchmark Suite

CQ's AI benchmark suite is a set of Machine Learning codes meant to be representative of common tasks launched by users of Calcul Québec's compute clusters. 

## Read before you consider opening a PR:

1. Here we are only concerned with collecting metrics pertaining to GPU performance during training. No attention is given to model performance/quality.
2. Due to #1 above, we set default batch sizes very high, and we pre-load datasets in GPU memory whenever possible to keep usage high at all times. Zero GPU idle time is what we're looking for here.
3. We wrote individual benchmark python scripts in a monolithic way on purpose. The scripts are pretty simple, so it's easier to read this way and you can immediately see everything that is going on inside just one file.

## Running the benchmark

Here are the pre-requisites for running this benchmark suite:

1. Your OS is a 64 bit Linux distro;
2. You have one or more Nvidia or AMD GPUs available (AMD GPUs **must** be compatible with ROCm 6);
3. You have [Rust](https://www.rust-lang.org/tools/install) installed in your environment;
4. You have [Arrow](https://arrow.apache.org/install/) installed in your environment.

With all these pre-requisites met, you can run ```prepare.sh```. This script will ask you what type of GPU you have (Nvidia or AMD) and install python dependencies accordingly. Then it will download all the data, model checkpoints and tokenizers used in the benchmarks to the appropriate locations (always inside your clone of this repo, so you can clean things up afterwards without leaving unwanted stuff in your $HOME).

Once ```prepare.sh``` finishes running, you can execute ```run.sh```. This script will loop through the '''benchmarks''' folder, run them with an increasing number of GPUs (if applicable) and print a performance report at the end.

## Troubleshooting

The following issues might come up when you run this benchmark:

### 1. ```rocm-smi``` or ```amd-smi``` is showing 100% usage of my AMD GPU, but the benchmark doesn't seem to be running at all.

- Make sure the environment variable ```MIOPEN_USER_DB_PATH``` points to a location where the current user can write. The default is ```$HOME/.config/miopen```, which should be ok.

- Make sure you can read/write at ```$HOME/.cache```. If you can't, set the environment variable ```MIOPEN_DISABLE_CACHE=1```

- If you are asking yourself "Why wouldn't I be able to read/write in my own $HOME", the last two points are not for you. But also do check in case you're trying to run this inside a container.

- If using multiple GPUs and they are **not configured** in a topology that allows direct GPU-to-GPU communication, set ```NCCL_P2P_DISABLE=1```.

### 2. The Large Language Model benchmark keeps failing, why?

The model used in that benchmark is [Meta's Llama2](). We are not using quantization, nor PEFT, or anything else that saves up memory. This means your system needs to be able load the whole thing. It should take around 80GB RAM of host memory. On the GPU side, we split this model across multiple GPUs using Pytorch [Fully Sharded Data Parallel](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html).  
