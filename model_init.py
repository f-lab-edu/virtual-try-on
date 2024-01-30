from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM
import torch.nn as nn

class VTryOnModel:

    def __init__(self):
        self.warp_model = AFWM(input_nc=3)
        print(self.warp_model)
        self.warp_model.eval()
        self.warp_model.cuda()
        load_checkpoint(model=self.warp_model, checkpoint_path='checkpoints\PFAFN\warp_model_final.pth')

        self.gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
        print(self.gen_model)
        self.gen_model.eval()
        self.gen_model.cuda()
        load_checkpoint(model=self.gen_model, checkpoint_path='checkpoints\PFAFN\gen_model_final.pth')

