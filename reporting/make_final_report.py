import os
import json

import pandas as pd
import numpy as np

def load_score(report_file):
    
    with open(report_file,'rb') as f:
        results = json.load(f)

    return results

#weights = TBD

bench_data = [[benchmark, load_score(f"./benchmarks/{benchmark}/{results_file}")] for results_file in os.listdir("./{benchmark}/results/") for benchmark in os.listdir("./benchmarks")]

index, results = zip(*bench_data)

results_df = pd.DataFrame(results,index=index)

score_metric = results[args.score_with]

final_score = np.exp(np.sum(np.log(score_metric) * weights) / np.sum(weights))

print("Benchmark results:")
print(results)

print("Final score: {final_score}")



