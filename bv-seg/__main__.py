f"""

"""
import os
import gc
import torch
import sys

from .core.argument_parser import BVSegArgumentParser
from .core.write_volumes import write_volumes
from .core.train_merged import train as train_merged
from .core.train_sequential import train as train_sequential
from .core.predict import predict

supported_commands = [
    "write-volumes",
    "train-merged",
    "train-sequential",
    "predict"
]

datasets = [
    "kidney_1_dense",
    "kidney_1_voi",
    "kidney_2",
    "kidney_3_dense",
    "kidney_3_sparse"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # defining arguments parser
    arg_parser = BVSegArgumentParser(
        supported_commands
    )        
    args = arg_parser.parse_args()
    print(f"{args=}")
    if args.command == "write-volumes":
        write_volumes(
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
            device,
            datasets
        )
    if args.command == "predict":
        predict(
            args,
            device,
        )
    