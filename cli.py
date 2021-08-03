
import argparse
import json
import os
import time
from pathlib import Path

from trainer_bionic import TrainerBionic
from trainer_bionit import TrainerBionit
from utils.common import create_time_taken_string

from eval.main import evaluate


base_eval_config_path = "eval/config/costanzo_hu_krogan.json"
base_eval_config_dir = "eval/config/"


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
    print(f"Done training, it took {time_start-time_end} time.")

    #
    # Evaluation
    #
    time_start = time.time()
    config_filename = config_path.name
    eval_config_path = os.path.join(base_eval_config_dir, config_filename)
    with open(args.config_path, "r") as fd:
        train_config = json.load(fd)
        features_path = train_config["out_name"] + "_features.tsv"
    with open(base_eval_config_path, "r") as fd:
        eval_config = json.load(fd)
        assert len(eval_config["features"]) == 1
        eval_config["features"][0]["path"] = features_path

    evaluate(Path(eval_config_path),
             exclude_tasks=["coannotation", "function_prediction"])
    time_end = time.time()
    print(f"Done evaluating, it took {time_start - time_end} time.")


if __name__ == '__main__':
    main()
