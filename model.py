from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM

import os
import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VTryOnModel:

    def __init__(self):
        self.warp_model = AFWM(input_nc=3)
        self.warp_model.eval()
        self.warp_model.cuda()
        load_checkpoint(model=self.warp_model, checkpoint_path='checkpoints\PFAFN\warp_model_final.pth')

        self.gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
        self.gen_model.eval()
        self.gen_model.cuda()
        load_checkpoint(model=self.gen_model, checkpoint_path='checkpoints\PFAFN\gen_model_final.pth')
        print('모델 초기화 완료')
    
    def infer(self, c_tensor, e_tensor, p_tensor):
        start = time.time()
        edge = torch.FloatTensor((e_tensor.detach().numpy() > 0.5).astype(np.int))
        clothes = c_tensor * edge

        flow_out = self.warp_model(p_tensor.cuda(), clothes.cuda())
        warped_cloth, last_flow = flow_out
        warped_edge = F.grid_sample(edge.cuda(), 
                                    last_flow.permute(0, 2, 3, 1),
                                    mode='bilinear', padding_mode='zeros')

        gen_inputs = torch.cat([p_tensor.cuda(), warped_cloth, warped_edge], 1)
        gen_outputs = self.gen_model(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * warped_edge
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

        path = 'results/'
        os.makedirs(path, exist_ok=True)

        person_figure = p_tensor.float().cuda()
        cloth_figure = clothes.cuda()
        result_figure = p_tryon
        combine = torch.cat([person_figure[0],cloth_figure[0],result_figure[0]], 2).squeeze()
        cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
        rgb=(cv_img*255).astype(np.uint8)
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        cv2.imwrite(path + "output" +'.jpg',bgr)
        end = time.time()
        print('추론 및 저장 완료!')
        print(f'추론 소요 시간 : {end - start: .5f}s')