import time
from options.test_options import TestOptions
from data.data_loader_test import CreateDataLoader
from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM
import torch.nn as nn
import os
import numpy as np
import torch
import cv2
import torch.nn.functional as F
import gc


class VTryOnModel:

    def __init__(self, start_epoch=1, epoch_iter=0):
        self.opt = TestOptions().parse()
        self.start_epoch = start_epoch
        self.epoch_iter = epoch_iter
        
        self.warp_model = None
        self.gen_model = None

    def load_data(self):
        data_loader = CreateDataLoader(self.opt)
        dataset = data_loader.load_data()
        dataset_size = len(data_loader)
        print(f'dataset size : {dataset_size}')
        return dataset, dataset_size

    def load_model(self):
        self.warp_model = AFWM(input_nc=3) # input number of channels
        print(self.warp_model)
        self.warp_model.eval()
        self.warp_model.cuda()
        load_checkpoint(self.warp_model, self.opt.warp_checkpoint)

        self.gen_model = ResUnetGenerator(input_nc=7, output_nc=4, num_downs=5, ngf=64, norm_layer=nn.BatchNorm2d)
        print(self.gen_model)
        self.gen_model.eval()
        self.gen_model.cuda()
        load_checkpoint(self.gen_model, self.opt.gen_checkpoint)

    def inference(self, dataset, dataset_size):
        total_steps = (self.start_epoch - 1) * dataset_size + self.epoch_iter
        step_per_batch = dataset_size / self.opt.batchSize
        step = 0

        for i, data in enumerate(dataset, start=self.epoch_iter):
            iter_start_time = time.time()
            total_steps += self.opt.batchSize
            self.epoch_iter += self.opt.batchSize

            real_image = data['image']
            clothes = data['clothes']
            ##edge is extracted from the clothes image with the built-in function in python
            edge = data['edge']
            edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
            clothes = clothes * edge

            flow_out = self.warp_model(real_image.cuda(), clothes.cuda()) ## cuda out of memory
            warped_cloth, last_flow, = flow_out
            warped_edge = F.grid_sample(edge.cuda(), last_flow.permute(0, 2, 3, 1),
                            mode='bilinear', padding_mode='zeros')

            gen_inputs = torch.cat([real_image.cuda(), warped_cloth, warped_edge], 1)
            gen_outputs = self.gen_model(gen_inputs)
            p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
            p_rendered = torch.tanh(p_rendered)
            m_composite = torch.sigmoid(m_composite)
            m_composite = m_composite * warped_edge
            p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)


            ## saving img
            path = 'results/' + self.opt.name
            os.makedirs(path, exist_ok=True)
            sub_path = path + '/PFAFN'
            os.makedirs(sub_path,exist_ok=True)

            if step & 1 == 0:
                fig_left = real_image.float().cuda()
                fig_center = clothes.cuda()
                fig_right = p_tryon
                combined_img = torch.cat([fig_left[0], fig_center[0], fig_right[0]], 2).squeeze()
                cv_img = (combined_img.permute(1,2,0).detach().cpu().numpy() + 1) / 2
                rgb = (cv_img * 255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(sub_path + '/' + str(step) + ".jpg", bgr)


            step += 1
            if self.epoch_iter > dataset_size:
                break



if __name__ == "__main__":
    # gc.collect()
    # torch.cuda.empty_cache()
    
    vton = VTryOnModel(start_epoch=1, epoch_iter=0)
    dataset, dataset_size = vton.load_data()
    vton.load_model()
    vton.inference(dataset, dataset_size)
            




