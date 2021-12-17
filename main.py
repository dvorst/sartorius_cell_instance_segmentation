import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.resnet import BasicBlock
import PIL
import matplotlib.pyplot as plt
import pathlib
import zipfile

import sartorius_cell_instance_segmentation as scis

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float
batch_size = 128
split_pct = .8


