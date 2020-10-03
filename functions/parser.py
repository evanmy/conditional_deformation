import torch
import numpy as np
from argparse import ArgumentParser

def train_parser():

    parser = ArgumentParser()

    parser.add_argument("gpus", type=str,
                        help="Which gpus to use?")
    
    parser.add_argument("ver",
                        type=str,
                        help="Additional string for the name of the file")

    parser.add_argument("--lr",
                        type=float,
                        dest="lr",
                        default=1e-4,
                        help="Encoder learning rate, default: 1e-4")
    
    parser.add_argument("--seed",
                        type=int,
                        dest="seed",
                        default=232,
                        help="Random seed to sort data, default: 232")
    
    parser.add_argument("--epochs",
                        type=int,
                        dest="epochs",
                        default=800,
                        help="Training epochs, default: 800")
    
    parser.add_argument("--save_dir",
                        type=str,
                        dest="save_dir",
                        default='./',
                        help="Path to the folder where data is saved")
    
    parser.add_argument("--use_dice",
                        action='store_true',
                        help='use dice loss instead of cross-entropy')   
    
    args = parser.parse_args()

    return args
