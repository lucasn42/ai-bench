#!/bin/bash

BASE_DIR=$PWD

export HF_HOME=$BASE_DIR/hf_cache
export HF_DATASETS_CACHE=$BASE_DIR/hf_cache
export TRANSFORMERS_CACHE=$BASE_DIR/hf_cache

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export CUBLAS_WORKSPACE_CONFIG=:4096:8

MAIN_NODE=$1
MAIN_PORT=$2
N_NODES=$3
N_GPUS=$4
MACHINE_RANK=$5

run_bench_parallel(){

   cd $BASE_DIR/benchmarks/$1
   echo "Running $1 benchmark with ${N_NODES} nodes and ${N_GPUS} device(s) per node..."

   N_PROCS=$((${N_NODES} * ${N_GPUS}))

       if [ "$1" == "large_language_model" ]; then

          accelerate launch --mixed_precision=fp16 --num_machines=${N_NODES} --num_processes=${N_PROCS} --main_process_ip=${MAIN_NODE} --main_process_port=${MAIN_PORT} --machine_rank=${MACHINE_RANK} --config_file="${BASE_DIR}/configs/fsdp_llama.yaml"  main.py

       else

          accelerate launch --mixed_precision=fp16 --num_machines=${N_NODES} --num_processes=${N_PROCS} --main_process_ip=${MAIN_NODE} --main_process_port=${MAIN_PORT} --machine_rank=${MACHINE_RANK} main.py

       fi

}


echo "Starting benchmark suite..."

BENCHMARKS=`ls $BASE_DIR/benchmarks`

for bench in $BENCHMARKS
do

   run_bench_parallel $bench

done

cd $BASE_DIR

echo "Collecting results and generating final performance report..."

python $BASE_DIR/reporting/make_final_report.py

