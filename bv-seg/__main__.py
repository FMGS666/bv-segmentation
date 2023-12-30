f"""

"""
import os
import gc
import torch
import sys

from .core.argument_parser import BVSegArgumentParser
from .core.sample import sample
from .core.train_merged import train as train_merged
from .core.train_sequential import train as train_sequential

supported_commands = [
    "sample",
    "train-merged",
    "train-sequential"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # defining arguments parser
    arg_parser = BVSegArgumentParser(
        supported_commands
    )        
    args = arg_parser.parse_args()
    print(f"{args=}")
    if args.command == "sample":
        sample(
            args
        )
    if args.command == "train-merged":
        train_merged(
            args,
            device
        )
    if args.command == "train-sequential":
        train_sequential(
            args,
            device
        )
    