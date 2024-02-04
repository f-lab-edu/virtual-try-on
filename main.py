import sys

from model import VTryOnModel
import config
import preprocess


if __name__ == "__main__":
    env = sys.argv[1] if len(sys.argv) > 2 else 'dev'
    
    if env == 'dev':
        config = config.DevelopmentConfig()
        
    vton = VTryOnModel()

    preprocess.generate_edge(config.edge_exist)
    preprocess.resize(config.resize)
    c_tensor, e_tensor, p_tensor = preprocess.img_to_tensor(config.img_root)

    vton.infer(c_tensor, e_tensor, p_tensor)


    



    