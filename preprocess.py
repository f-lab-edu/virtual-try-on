from PIL import Image
import os
import glob
import sys
sys.path.append("u2_segment/")

import torch
from torchvision.transforms import transforms
import process, network, options


# input_path = '/kaggle/input/custom-data1/color/*.jpg'
# output_path = '/kaggle/working/dataset/VITON-Clean/VITON_test/test_color/'
# file_list = [file for file in glob.glob(input_path)]

# for f in file_list:
#     img = Image.open(f)
#     img_resize = img.resize((192, 256))
#     path = output_path + f.split("/")[-1]
#     img_resize.save(path)


def generate_edge(
                    edge_exist = True,
                    device='cpu', 
                    img_path="u2_segment/input/019119_1.jpg",
                    checkpoint_path='u2_segment\model\cloth_segm.pth',
                    output_path='dataset\service_edge'
                ):
    if edge_exist: 
        pass
    else:
        process.main(device, img_path, checkpoint_path, output_path)

def resize(resize=True):
    if resize:
        pass
    else:
        pass

def img_to_tensor(img_root):
    cloth = Image.open(img_root + 'service_cloth/' + '019119_1.jpg').convert('RGB')
    edge =  Image.open(img_root + 'service_edge/' + '019119_1.jpg').convert('L')
    person = Image.open(img_root + 'service_img/' + '005510_0.jpg').convert('RGB')


    # transform_list += [transforms.Lambda(lambda img: __make_power_2(img, base=float(16), method=Image.BICUBIC))]
    transform_list = [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    # transform_list_edge = [transforms.Lambda(lambda img: __make_power_2(img, base=float(16), method=Image.NEAREST))]
    transform_list_edge = [transforms.ToTensor()]

    c_tensor = transforms.Compose(transform_list)(cloth)
    e_tensor = transforms.Compose(transform_list_edge)(edge)
    p_tensor = transforms.Compose(transform_list)(person)

    c_tensor = torch.unsqueeze(c_tensor, 0)
    e_tensor = torch.unsqueeze(e_tensor, 0)
    p_tensor = torch.unsqueeze(p_tensor, 0)

    return c_tensor, e_tensor, p_tensor









