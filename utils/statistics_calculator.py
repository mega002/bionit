import os
import csv
from collections import defaultdict
import statistics
import argparse

def save_results_statistics_to_file(results_file_path, output_dir):
    results = defaultdict(list)

    # Read the results
    with open(results_file_path) as results_file:
        csv_reader = csv.reader(results_file, delimiter="\t")
        line_num = 0
        for row in csv_reader:
            if line_num > 0:
                key1 = row[1]
                key2 = row[2]
                score = float(row[3])
                results[(key1,key2)].append(score)
            line_num += 1

    # Calculate the statistics
    results_statistics = list()
    results_statistics.append(("key1", "key2", "SD", "Mean"))
    for key, results in results.items():
        if key[1] == "Features":
            stdev = statistics.stdev(results)
            mean = statistics.mean(results)
            results_statistics.append((key[0], key[1], stdev, mean))

    # Save the statistics to a tsv file
    statistics_file_name = \
        os.path.splitext(os.path.basename(results_file_path))[0] + "_results.tsv"
    with open(os.path.join(output_dir, statistics_file_name), 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(results_statistics)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    save_results_statistics_to_file(args.results_file_path, args.output_dir)

if __name__ == '__main__':
    main()