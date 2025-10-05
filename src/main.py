#!/usr/bin/python


# Author : 
#    Mahanth Yalla,
#    M.Tech - Artificial Intelligence 
#    Vision \& AI Lab, CDS Dept.,
#    Indian Institute of Science.

from dotenv import load_dotenv
load_dotenv() 


import warnings
# warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


import os
import yaml
import time
import json
import math
import pytz
import argparse
import platform
from datetime import datetime as dt

import torch


from rich.box import HEAVY
from rich.panel import Panel
from rich.console import Console
console = Console()



def parse_args():
    parser = argparse.ArgumentParser(description="Model training config overrides")
    parser.add_argument('-c',"--config_file", type=str, default=None, help="Optional additional YAML config file to override defaults")
    parser.add_argument('-z',"--model_type", type=str, default='DEBUG', help="Optional additional name of YAML config file to override defaults")
    parser.add_argument('-d',"--data_list", type=str, default='DEBUG', help="Optional additional name of YAML config file to override defaults")
    parser.add_argument('-t',"--train", action="store_true", help="starts training")
    parser.add_argument('-g',"--generate", action="store_false", default=True, help="Toggle on to generate from model")
    parser.add_argument('-b',"--batch_size", type=int, default=None, help="Config: batch_size")
    parser.add_argument('-m',"--max_steps", type=int, default=None, help="Config: max_steps")
    parser.add_argument('-a',"--trainer.accum_steps", type=int, default=None, help="Config: accum_steps ")
    parser.add_argument('-w',"--trainer.train_time_warmup", type=int, default=None, help="Config: train_time_warmup ")
    parser.add_argument('-p',"--trainer.precomputed_train_time", type=float, default=None, help="Config: precomputed ? train_time_warmup ")
    parser.add_argument('-l',"--logging.log_dir", type=str, help="logging dir")
    parser.add_argument('-e',"--trainer.ema_decay", type=float, default=None, help="Config: ema_decay ")
    parser.add_argument('-r',"--trainer.resume_train", action="store_true", default=None, help="Resume ?")
    parser.add_argument("--gpu.compile_approach", type=str, default=None, help="set mode")
    parser.add_argument("--data-folder-list", type=str, default=None, help="list of dirs")
    parser.add_argument("--optimizer.lr", type=float, default=None, help="Learning rate")
    return parser.parse_args()


def main():
    console.print(f"[yellow]🧭 Torch: {torch.__version__}, CUDA: {torch.version.cuda}, Python: {platform.python_version()}[/yellow]")
    console.print('[red]Not Implemented Yet[/red]')


if __name__ == "__main__":
    start = dt.now()
    main()
    console.print('[bold green]✅ Done! Script Completed Successfuly[/bold green]')
    print(f'⌚ Script Time: {dt.now() - start}')