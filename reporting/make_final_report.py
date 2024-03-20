import os
import json

import pandas as pd
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='GPU perf reporting script')
parser.add_argument('--score_with', default='train_samples_per_second', help='')

args = parser.parse_args()

def load_json(report_file):

    with open(report_file,'rb') as f:
        results = json.load(f)

    return results


bench_data = [[benchmark, load_json(f"./benchmarks/{benchmark}/results/{results_file}")] for benchmark in os.listdir("./benchmarks") for results_file in os.listdir(f"./benchmarks/{benchmark}/results/")]

index, results = zip(*bench_data)

results_df = pd.DataFrame(results, index=pd.Index(index,name='Benchmark'))
results_df.reset_index(inplace=True)

weights_data = load_json("./reporting/weights.json")
weights_data = [[benchmark["benchmark"],weight] for benchmark in weights_data["benchmarks"] for weight in benchmark["weights"]]

index, weights = zip(*weights_data)

weights_df = pd.DataFrame(weights, index=pd.Index(index, name='Benchmark'))
weights_df.reset_index(inplace=True)

report_df = pd.merge(results_df,weights_df, on=['Benchmark','num_gpus'], how='left').dropna()

score_metric = report_df[args.score_with]

weights = report_df["weight"]

final_score = np.exp(np.sum(np.log(score_metric) * weights) / np.sum(weights))

print("Benchmark results: \n")
print(report_df.to_string(index=False))

print(f" \n Final score: {final_score}")

