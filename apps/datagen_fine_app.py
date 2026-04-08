"""
SIM1-DataGen (Fine mode): Fine-grained data generation entry point.
"""

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from components.datagen.datagen_core import DataGenerator
from components.datagen.selector import Selector, SelectorFine
from components.datagen.splitter import Splitter, SplitterFine
from components.randomization import Randomizer
from envs import LiftShortShirt


def get_project_root():
    """Return the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="SIM1-DataGen Fine: fine-grained trajectory data generation")
    parser.add_argument(
        "--data_folder",
        type=str,
        default=None,
        help="Root folder for input/output data (default: ./dataset/example)",
    )
    parser.add_argument(
        "--task_split_cfg",
        type=str,
        default=None,
        help="Path to task split config JSON (default: ./components/datagen/configs/lift_cloth_manip.json)",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=1000,
        help="Number of trajectories to generate (default: 1000)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fine",
        choices=["fine", "baseline"],
        help="Splitter/Selector mode: 'fine' or 'baseline'",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = get_project_root()

    if args.data_folder is None:
        args.data_folder = os.path.join(project_root, "dataset", "example")

    if not os.path.isabs(args.data_folder):
        args.data_folder = os.path.join(project_root, args.data_folder)

    if args.task_split_cfg is None:
        args.task_split_cfg = os.path.join(
            project_root, "components", "datagen", "configs", "lift_cloth_manip.json"
        )

    data_folder = args.data_folder

    if args.mode == "fine":
        splitter = SplitterFine(data_folder)
        selector = SelectorFine(data_folder)
    else:
        splitter = Splitter(data_folder, args.task_split_cfg)
        selector = Selector(data_folder, args.task_split_cfg)

    env = LiftShortShirt(Randomizer())
    datagen = DataGenerator(env, splitter, selector, data_folder)
    datagen.data_gen(args.num)


if __name__ == "__main__":
    main()
