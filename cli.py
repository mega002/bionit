
import argparse
import json
import os
import time
from pathlib import Path

from trainer_bionic import TrainerBionic
from trainer_bionit import TrainerBionit
from utils.statistics_calculator import save_results_statistics_to_file
from utils.common import create_time_taken_string

from eval.main import evaluate


base_eval_config_path = "eval/config/costanzo_hu_krogan.json"
base_eval_config_dir = "eval/config/"
base_eval_results_dir = "eval/results/"


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

    module_detection_results_file_name = \
        os.path.join(base_eval_results_dir, out_name + "_module_detection.tsv")
    save_results_statistics_to_file(module_detection_results_file_name)


if __name__ == '__main__':
    main()