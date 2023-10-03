import csv
import argparse
from collections import defaultdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, required=True)
    args = parser.parse_args()

    # Load results
    dataset_name = None
    results = defaultdict(lambda: defaultdict(float))
    with open(args.result_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Dataset:"):
                dataset_name = line.strip().split(": ")[-1]
            if dataset_name is not None:
                if line.startswith("MRR:"):
                    results[dataset_name]["MRR"] = float(line.strip().split(": ")[-1])
                elif line.startswith("R@1"):
                    results[dataset_name]["R@1"] = float(line.strip().split(": ")[-1])
                elif line.startswith("R@5"):
                    results[dataset_name]["R@5"] = float(line.strip().split(": ")[-1])
                elif line.startswith("R@1000"):
                    results[dataset_name]["R@1000"] = float(line.strip().split(": ")[-1])

    # Average results
    datasets_num = len(results)
    results["AVERAGE"]["MRR"] = round(sum([results[dataset_name]["MRR"] for dataset_name in results]) / datasets_num, 1)
    results["AVERAGE"]["R@1"] = round(sum([results[dataset_name]["R@1"] for dataset_name in results]) / datasets_num, 1)
    results["AVERAGE"]["R@5"] = round(sum([results[dataset_name]["R@5"] for dataset_name in results]) / datasets_num, 1)
    results["AVERAGE"]["R@1000"] = round(sum([results[dataset_name]["R@1000"] for dataset_name in results]) / datasets_num, 1)
    
    # Save results to csv
    flatten_results = {}
    for dataset_name in results:
        for metric_name in results[dataset_name]:
            flatten_results[f"{dataset_name}_{metric_name}"] = results[dataset_name][metric_name]

    print(flatten_results)
    with open("./results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(list(flatten_results.values()))
    