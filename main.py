import sys
import logging

from model import VTryOnModel
import yaml
# import config
import preprocess


if __name__ == "__main__":
    logging.basicConfig( level=logging.DEBUG)
    # env = sys.argv[1] if len(sys.argv) > 2 else 'dev'
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # if env == 'dev':
    #     config = config.DevelopmentConfig()
    # elif env == 'test':
    #     config = config.TestConfig()

    device = "gpu" if config["device"] else "cpu"
    edge_exist = config["data"]["edge_exist"]
    resize = config["data"]["resize"]
    img_root = config["data"]["root"]

        
    vton = VTryOnModel(device)

    preprocess.generate_edge(edge_exist)
    preprocess.resize(resize)
    c_tensor, e_tensor, p_tensor = preprocess.img_to_tensor(img_root)

    vton.infer(c_tensor, e_tensor, p_tensor)


