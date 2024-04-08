
import os
import yaml

class Configuration(object):
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(Configuration, cls).__new__(cls)
            cls._instance.load_config()
            cls._instance.load_credential()
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
            self.output_path = cfg["data"]["output_path"]

    def load_credential(self, config_file="config/ncpconfig.yaml"):
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
            self.service_name = cfg["service_name"]
            self.endpoint_url = cfg["endpoint_url"]
            self.region_name = cfg["region_name"]
            self.access_key = cfg["access_key"]
            self.secret_key = cfg["secret_key"]


opt = Configuration()
