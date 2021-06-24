import argparse

from pathlib import Path
from typing import Optional, List

from eval.utils.resolvers import resolve_config_path, resolve_tasks
from eval.utils.process_config import process_config
from eval.utils.file_utils import import_datasets

from eval.evals.coannotation import coannotation_eval
from eval.evals.module_detection import module_detection_eval
from eval.evals.function_prediction import function_prediction_eval


def evaluate(
    config_path: Path,
    exclude_tasks: Optional[List[str]] = [],
    exclude_standards: Optional[List[str]] = [],
):
    resolve_config_path(config_path)
    process_config(exclude_tasks, exclude_standards)
    import_datasets()
    tasks = resolve_tasks()
    for task in tasks:
        print(task)
        evaluate_task(task)


def evaluate_task(task: str):
    if task == "coannotation":
        coannotation_eval()
    elif task == "module_detection":
        module_detection_eval()
    elif task == "function_prediction":
        function_prediction_eval()
    else:
        raise NotImplementedError(f"Task '{task}' has not been implemented.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--exclude_tasks", nargs='+')
    parser.add_argument("--exclude_standards", nargs='+')
    args = parser.parse_args()

    evaluate(Path(args.config_path), args.exclude_tasks, args.exclude_standards)


if __name__ == '__main__':
    main()
