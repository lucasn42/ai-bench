#!/bin/bash

BASE_DIR=$PWD

export HF_HOME=$BASE_DIR/hf_cache
export HF_DATASETS_CACHE=$BASE_DIR/hf_cache
export TRANSFORMERS_CACHE=$BASE_DIR/hf_cache

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export CUBLAS_WORKSPACE_CONFIG=:4096:8

read -p "How many GPUs will be used to run this benchmark?" N_GPUS;

run_bench(){

   for n_proc in $(seq 1 $N_GPUS)
   do  
       cd $BASE_DIR/benchmarks/$1
       echo "Running $1 benchmark with $n_proc device(s)..."

       if [ "$1" == "large_language_model" ]; then

          accelerate launch --mixed_precision=fp16 --num_machines=1 --num_processes=$n_proc --config_file="${BASE_DIR}/configs/fsdp_llama.yaml"  main.py

       else

          accelerate launch --mixed_precision=fp16 --num_machines=1 --num_processes=$n_proc  main.py

       fi
       
   done
}


echo "Starting benchmark suite..."

BENCHMARKS=`ls $BASE_DIR/benchmarks`

for bench in $BENCHMARKS
do

   run_bench $bench

done

cd $BASE_DIR

echo "Collecting results and generating final performance report..."

python $BASE_DIR/reporting/make_final_report.py

