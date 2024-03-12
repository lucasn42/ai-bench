#!/bin/bash

BASE_DIR=$PWD

# Setting this here for testing. Remove later.
N_GPUS=4

run_bench(){

   for n_proc in $(seq 1 $N_GPUS)
   do  
       cd $BASE_DIR/benchmarks/$1
       echo "Running $1 benchmark with $n_proc device(s)..."

       if [ "$1" == "large_language_model" && $n_proc -gt 3 ]; then
          accelerate launch --mixed_precision=fp16 --num_machines=1 --num_processes=$n_proc --config_file="../configs/fsdp_llama.yaml" main.py
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




