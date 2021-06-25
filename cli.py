
import argparse
import time
from pathlib import Path

from trainer_bionic import TrainerBionic
from trainer_bionit import TrainerBionit
from utils.common import create_time_taken_string


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=['bionic', 'bionit'])
    args = parser.parse_args()

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


if __name__ == '__main__':
    main()
