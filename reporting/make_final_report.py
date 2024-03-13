import os
import json

import pandas as pd
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='GPU perf reporting script')
parser.add_argument('--score_with', default='train_samples_per_second', help='')

args = parser.parse_args()

def load_score(report_file):

    with open(report_file,'rb') as f:
        results = json.load(f)

    return results

#weights = TBD

bench_data = [[benchmark, load_score(f"./benchmarks/{benchmark}/results/{results_file}")] for benchmark in os.listdir("./benchmarks") for results_file in os.listdir(f"./benchmarks/{benchmark}/results/")]

index, results = zip(*bench_data)

results_df = pd.DataFrame(results,index=index)

score_metric = results_df[args.score_with]

#final_score = np.exp(np.sum(np.log(score_metric) * weights) / np.sum(weights))

print("Benchmark results:")
print(results_df)

print("Final score: {final_score}")


