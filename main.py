import time
# from options.test_options import TestOptions
# from data.data_loader_test import CreateDataLoader
from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM
import torch.nn as nn
import os
import numpy as np
import torch
import cv2
import torch.nn.functional as F



## model initialization
from model_init import VTryOnModel
import preprocess
## preprocessing (not as a dataloader, just an img data)



if __name__ == "__main__":
    vton = VTryOnModel()
    preprocess.generate_edge()

    