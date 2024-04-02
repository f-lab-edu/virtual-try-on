
import os
import yaml

class Configuration(object):
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(Configuration, cls).__new__(cls)
            cls._instance.load_config()
        return cls._instance

    def load_config(self, config_file="config/config.yaml"):
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
            self.device = "gpu" if cfg["device"] else "cpu"
            self.root = cfg["data"]["root"]
            self.edge_exist = cfg["data"]["edge_exist"]
            self.cloth_path = cfg["data"]["cloth_path"]
            self.edge_path = cfg["data"]["edge_path"]
            self.person_path = cfg["data"]["person_path"]

opt = Configuration()
