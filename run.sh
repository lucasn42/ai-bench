#!/bin/bash

BASE_DIR = $PWD

# Setting this here for testing. Remove later.
N_GPUS=3

run_bench(){

   for n_proc in {1..$N_GPUS}
   do  
       cd $BASE_DIR/benchmarks/$1
       echo "Running $1 benchmark with $n_proc device(s)..."

       if [ "$1" -eq "large_language_model" && $n_proc -gt 1 ]; then
          accelerate --mixed_precision=fp16 --num_machines=1 --num_processes=i --config_file="../configs/fsdp_llama.yaml" main.py
       else
          accelerate --mixed_precision=fp16 --num_machines=1 --num_processes=i  main.py
       
       fi

   done
}


echo "Starting benchmark suite..."

BENCHMARKS = ls $BASE_DIR/benchmarks

for bench in $BENCHMARKS
do

   run_bench $bench

done




