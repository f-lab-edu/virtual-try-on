from PIL import Image
import os
import glob
import sys
sys.path.append("u2_segment/")

import process, network, options


# input_path = '/kaggle/input/custom-data1/color/*.jpg'
# output_path = '/kaggle/working/dataset/VITON-Clean/VITON_test/test_color/'
# file_list = [file for file in glob.glob(input_path)]

# for f in file_list:
#     img = Image.open(f)
#     img_resize = img.resize((192, 256))
#     path = output_path + f.split("/")[-1]
#     img_resize.save(path)

def generate_edge(device='cpu', 
                    img_path="u2_segment/input/019119_1.jpg",
                    checkpoint_path='u2_segment\model\cloth_segm.pth',
                    output_path='dataset\service_edge'):
    
    process.main(device, img_path, checkpoint_path, output_path)

def resize():
    pass







