import sys
import logging

from model import VTryOnModel
import config
import preprocess


if __name__ == "__main__":
    logging.basicConfig(filename='main.py', level=logging.DEBUG)
    env = sys.argv[1] if len(sys.argv) > 2 else 'dev'
    
    if env == 'dev':
        config = config.DevelopmentConfig()
    elif env == 'test':
        config = config.TestConfig()
        
    vton = VTryOnModel()

    preprocess.generate_edge(config.edge_exist)
    preprocess.resize(config.resize)
    c_tensor, e_tensor, p_tensor = preprocess.img_to_tensor(config.img_root)

    vton.infer(c_tensor, e_tensor, p_tensor)



INFO:root:Segmentation Time Taken :  20.58530s (CPU)
INFO:root:Inference Time Taken :  1.24623s
