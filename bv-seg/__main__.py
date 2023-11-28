f"""

"""
import os
import gc
import torch
import sys

from .core.argument_parser import BVSegArgumentParser
from .core.save_volumes import save_volumes
from .core.train import train

supported_commands = [
    "sample",
    "train"
]

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    # defining arguments parser
    arg_parser = BVSegArgumentParser(
        supported_commands
    )        
    args = arg_parser.parse_args()
    if args.command == "sample":
        save_volumes(
            args
        )
    if args.command == "train":
        train(
            args,
            device
        )
    