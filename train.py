############ Import Libraries ############
from torch.nn import Module, ReLU, Conv2d, Linear, MaxPool2d, LogSoftmax, NLLLoss, Dropout, BatchNorm2d, LeakyReLU, GELU, SELU, Mish
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import flatten, float, no_grad
from torch.optim import Adam
import torch
import wandb
import math
import matplotlib.pyplot as plt
import numpy as np
import sys, argparse


################################# Parse Arguments #################################

def parse_arguments():
	parser = argparse.ArgumentParser(description='Training Arguments')
	parser.add_argument('-wp', '--wandb_project', type=str, default='dl_assignment_2', help='Project name used to track experiments in Weights & Biases dashboard')
	parser.add_argument('-we', '--wandb_entity', type=str, default='dl_assignment_2', help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
	parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train neural network.')
	parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size used to train neural network.')
	parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate used to optimize model parameters')
	parser.add_argument('-a', '--activation', type=str, default='relu', choices=["selu", "mish", "relu"], help='Activation function choice: ["selu", "mish", "relu"]')
	parser.add_argument('-f', '--filters', type=int, default=64, help='Number of filters in 1st layer')
	parser.add_argument('-fo', '--filter_org', type=str, default='half', choices=["same", "half", "double"], help='Number of Filters down the network')
	parser.add_argument('-da', '--data_augmentation', type=bool, default=True, choices=[True, False], help='Augment data randomly.')
	parser.add_argument('-dr', '--dropout', type=float, default=0.2, help='Dropout probability to be used.')
	return parser.parse_args()