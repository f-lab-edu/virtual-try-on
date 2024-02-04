# import time
# from PIL import Image
# from models.networks import ResUnetGenerator, load_checkpoint
# from models.afwm import AFWM
# import torch.nn as nn
# import os
# import numpy as np
# import torch
# from torchvision.transforms import transforms
# import cv2
# import torch.nn.functional as F
# import random


from model import VTryOnModel
import preprocess


if __name__ == "__main__":
    
    vton = VTryOnModel()

    preprocess.generate_edge()
    preprocess.resize()
    c_tensor, e_tensor, p_tensor = preprocess.img_to_tensor("dataset/service_cloth/019119_1.jpg",
                                                            "dataset/service_edge/019119_1.jpg",
                                                            "dataset/service_img/005510_0.jpg")
    vton.infer(c_tensor, e_tensor, p_tensor)


    



    