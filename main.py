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
    c_tensor, e_tensor, p_tensor = preprocess.img_to_tensor(config.path_c, config.path_e, config.path_p)

    vton.infer(c_tensor, e_tensor, p_tensor)


    



    