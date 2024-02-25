import os
import logging
import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM
from torchsummary import summary


class VTryOnModel:

    def __init__(self, device):
        self.device = device
        self.warp_model = AFWM(self.device, input_nc=3)
        self.warp_model.eval()
        if self.device=="gpu":self.warp_model.cuda() 
        load_checkpoint(model=self.warp_model, checkpoint_path='checkpoints\PFAFN\warp_model_final.pth')

        self.gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
        self.gen_model.eval()
        if self.device=="gpu" : self.gen_model.cuda() 
        load_checkpoint(model=self.gen_model, checkpoint_path='checkpoints\PFAFN\gen_model_final.pth')
        print('Model Initialized')
    
    def infer(self, c_tensor, e_tensor, p_tensor):
        start = time.time()
        edge = torch.FloatTensor((e_tensor.detach().numpy() > 0.5).astype(np.int))
        clothes = c_tensor * edge

        if self.device == "gpu":
            p_tensor = p_tensor.cuda()
            clothes = clothes.cuda()
            edge = edge.cuda()

        # print("CPU AFWM MODEL",summary(self.warp_model(p_tensor, clothes), input_size=()))
        # print("input size", p_tensor.size(), clothes.size())
        flow_out = self.warp_model(p_tensor, clothes)

        warped_cloth, last_flow = flow_out
        warped_edge = F.grid_sample(edge, 
                                    last_flow.permute(0, 2, 3, 1),
                                    mode='bilinear', padding_mode='zeros')

        # print(p_tensor.size(), warped_cloth.size(), warped_edge.size())
        gen_inputs = torch.cat([p_tensor, warped_cloth, warped_edge], 1)
        gen_outputs = self.gen_model(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * warped_edge
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

        path = 'results/'
        os.makedirs(path, exist_ok=True)

        person_figure = p_tensor.float()
        cloth_figure = clothes
        result_figure = p_tryon
        combine = torch.cat([person_figure[0],cloth_figure[0],result_figure[0]], 2).squeeze()
        cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
        rgb=(cv_img*255).astype(np.uint8)
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)

        # cv2.imwrite(path+'debug'+'.jpg', warped_cloth)

        cv2.imwrite(path + "output" +'.jpg',bgr)
        end = time.time()
        print('Inference Complete & Saved!')
        logging.info(f'Inference Time Taken : {end - start: .5f}s')