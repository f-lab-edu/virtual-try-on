import time
from PIL import Image
# from options.test_options import TestOptions
# from data.data_loader_test import CreateDataLoader
from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM
import torch.nn as nn
import os
import numpy as np
import torch
from torchvision import transforms
import cv2
import torch.nn.functional as F



## model initialization
from model_init import VTryOnModel
import preprocess

## preprocessing (not as a dataloader, just an img data)



if __name__ == "__main__":
    
    vton = VTryOnModel()

    preprocess.generate_edge()
    preprocess.resize()



    cloth = Image.open("dataset/service_cloth/019119_1.jpg")
    edge =  Image.open("dataset/service_edge/019119_1.jpg")
    person = Image.open("dataset/service_img/005510_0.jpg")

    tf = transforms.ToTensor()
    img_c = torch.unsqueeze(tf(cloth), 0)
    img_e = torch.unsqueeze(tf(edge), 0)
    img_p = torch.unsqueeze(tf(person), 0)
    
    edge = torch.FloatTensor((img_e.detach().numpy() > 0.5).astype(np.int))
    clothes = img_c * edge

    flow_out = vton.warp_model(img_c.cuda(), clothes.cuda())
    warped_cloth, last_flow = flow_out
    warped_edge = F.grid_sample(edge.cuda(), last_flow.permute(0, 2, 3, 1),
                    mode='bilinear', padding_mode='zeros')

    gen_inputs = torch.cat([img_p.cuda(), warped_cloth, warped_edge], 1)
    gen_outputs = vton.gen_model(gen_inputs)
    p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
    p_rendered = torch.tanh(p_rendered)
    m_composite = torch.sigmoid(m_composite)
    p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

    path = 'results/'
    os.makedirs(path, exist_ok=True)
    # sub_path = path + '/PFAFN'
    # os.makedirs(sub_path,exist_ok=True)

    a = img_p.float().cuda()
    b= clothes.cuda()
    c = p_tryon
    combine = torch.cat([a[0],b[0],c[0]], 2).squeeze()
    cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
    rgb=(cv_img*255).astype(np.uint8)
    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    cv2.imwrite(path + "output" +'.jpg',bgr)


    



    