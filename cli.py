
import argparse
import time
from pathlib import Path
from train import Trainer
from utils.common import create_time_taken_string


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    config_path = Path(args.config_path)
    time_start = time.time()
    trainer = Trainer(config_path)
    trainer.train()
    trainer.forward()
    time_end = time.time()


if __name__ == '__main__':
    main()
