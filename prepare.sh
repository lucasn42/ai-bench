#!/bin/bash


read -p "What GPU environment will be used to run this benchmark suite? [Nvidia/AMD] " GPU_ENVIRONMENT;

case $GPU_ENVIRONMENT in
    AMD) echo "AMD environment selected.";;
    Nvidia) echo "Nvidia environment selected.";;
    *) echo "Invalid environment. Please select one of: [Nvidia/AMD]" && exit;;

esac

echo "Installing benchmark python dependencies..."

if [ $GPU_ENVIRONMENT -eq "Nvidia" ] then;

    pip install -r requirements_cuda.txt

else
  
    pip install -r requirements_rocm.txt

fi

wait

echo "Switching Huggingface Cache to local directory..."

mkdir ./hf_cache

export HF_HOME=./hf_cache
export HF_DATASETS_CACHE=./hf_cache
export TRANSFORMERS_CACHE=./hf_cache

echo "Downloading benchmark artifacts..."

python get_data_and_models.py

