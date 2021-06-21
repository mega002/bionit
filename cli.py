import time
from pathlib import Path
from train import Trainer
from utils.common import create_time_taken_string


config_path = Path("config/costanzo_hu_krogan.json")
time_start = time.time()
trainer = Trainer(config_path)
trainer.train()
trainer.forward()
time_end = time.time()
