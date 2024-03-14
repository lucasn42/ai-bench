#!/bin/bash

SETUP_DIR=$PWD/setup

read -p "What GPU environment will be used to run this benchmark suite? [Nvidia/AMD] " GPU_ENVIRONMENT;

case $GPU_ENVIRONMENT in
    AMD) echo "AMD environment selected.";;
    Nvidia) echo "Nvidia environment selected.";;
    *) echo "Invalid environment. Please select one of: [Nvidia/AMD]" && exit;;

esac

echo "Installing benchmark python dependencies..."

if [ $GPU_ENVIRONMENT == "Nvidia" ]; then
    pip install -r $SETUP_DIR/requirements_cuda.txt

else
    pip install -r $SETUP_DIR/requirements_rocm.txt

fi

wait

echo "Switching Huggingface Cache to local directory..."

mkdir ./hf_cache

export HF_HOME=./hf_cache
export HF_DATASETS_CACHE=./hf_cache
export TRANSFORMERS_CACHE=./hf_cache

echo "Downloading benchmark artifacts..."

python $SETUP_DIR/get_data_and_models.py

echo "Done downloading artifacts."
echo "Please execute run.sh to launch the benchmark."

