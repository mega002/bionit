
import argparse
import json
import os
import time
from pathlib import Path
import csv
from collections import defaultdict
import statistics

from trainer_bionic import TrainerBionic
from trainer_bionit import TrainerBionit
from utils.common import create_time_taken_string

from eval.main import evaluate


base_eval_config_path = "eval/config/costanzo_hu_krogan.json"
base_eval_config_dir = "eval/config/"
base_eval_results_dir = "eval/results/"

def aggregate_results(results_file_path):
    results = defaultdict(list)

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

    aggregated_res = list()
    aggregated_res.append(("key1", "key2", "SD", "Mean"))
    for key, results in results.items():
        stdev = statistics.stdev(results)
        mean = statistics.mean(results)
        aggregated_res.append((key[0], key[1], stdev, mean))

    results_dir = os.path.dirname(results_file_path)
    aggregated_res_file_name = \
        os.path.splitext(os.path.basename(results_file_path))[0] + "_results.tsv"
    with open(os.path.join(results_dir, aggregated_res_file_name), 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(aggregated_res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=['bionic', 'bionit'])
    args = parser.parse_args()

    #
    # Training
    #
    config_path = Path(args.config_path)
    time_start = time.time()
    if args.model == 'bionic':
        trainer = TrainerBionic(config_path)
    elif args.model == 'bionit':
        trainer = TrainerBionit(config_path)
    else:
        raise ValueError()
    trainer.train()
    trainer.forward()
    time_end = time.time()
    print(f"Done training, it took {round(time_end - time_start, 2)} seconds.")

    #
    # Evaluation
    #
    # get the training output from the input configuration.
    with open(args.config_path, "r") as fd:
        train_config = json.load(fd)
        features_path = train_config["out_name"] + "_features.tsv"

    # load the default evaluation configuration, and modify the features path.
    with open(base_eval_config_path, "r") as fd:
        eval_config = json.load(fd)
        assert len(eval_config["features"]) == 1
        eval_config["features"][0]["path"] = features_path

    # save the new evaluation configuration.
    out_name = Path(train_config["out_name"]).name
    config_filename = out_name + ".json"
    eval_config_path = os.path.join(base_eval_config_dir, config_filename)
    with open(eval_config_path, "w") as fd:
        json.dump(eval_config, fd, indent=4)

    # run evaluation
    time_start = time.time()
    evaluate(Path(eval_config_path),
             exclude_tasks=["coannotation", "function_prediction"])
    time_end = time.time()
    print(f"Done evaluating, it took {round(time_end - time_start, 2)} seconds.")

    aggregate_results(os.path.join(base_eval_results_dir, out_name + "_module_detection.tsv"))


if __name__ == '__main__':
    main()