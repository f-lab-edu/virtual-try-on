import os
import glob
import sys
import torch
from torchvision.transforms import transforms
from PIL import Image

sys.path.append("u2_segment/")
import process

def generate_edge(
                    output_name,
                    edge_exist = True,
                    device='cpu', 
                    img_path="u2_segment/input/019119_1.jpg",
                    checkpoint_path='u2_segment\model\cloth_segm.pth',
                    output_path='dataset\service_edge'
                    
                ):
    if edge_exist: 
        pass
    else:
        process.main(device, img_path, checkpoint_path, output_path, output_name)

def resize(img_path):
    if resize:
        img = Image.open(img_path)
        img_resize = img.resize((192, 256))
        img_resize.save(img_path)

def img_to_tensor(cloth_path, edge_path, person_path):
    cloth = Image.open(cloth_path).convert('RGB')
    edge =  Image.open(edge_path).convert('L')
    person = Image.open(person_path).convert('RGB')


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









