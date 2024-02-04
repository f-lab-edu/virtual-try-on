class Config():
    host = 'localhost'
    # dbname = 'dbname'
    # user = 'user'
    # password = 'password'
    # port = 'port' 

class DevelopmentConfig(Config):
    img_root = "dataset/"
    edge_exist = False
    name = 'dev'
    resize = False
    device = 'cpu'

    def __init__(self):
        print(f'img_root ={self.img_root}')
        print(f'edge_exist ={self.edge_exist}')
        print(f'name ={self.name}')
        print(f'resize ={self.resize}')
        print(f'device = {self.device}')

class TestConfig(Config):
    DATAROOT = 'dataset/'
    NAME = 'test'
    # # GPU = 0



