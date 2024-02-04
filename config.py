class Config():
    host = 'localhost'
    dbname = 'dbname'
    user = 'user'
    password = 'password'
    port = 'port' 

class DevelopmentConfig(Config):
    path_c = "dataset/service_cloth/019119_1.jpg"
    path_e = "dataset/service_edge/019119_1.jpg"
    path_p = "dataset/service_img/005510_0.jpg"
    edge_exist = True
    name = 'dev'
    resize = False
    device = 'cpu'

    def __init__(self):
        print(f'path_c ={self.path_c}')
        print(f'path_e ={self.path_e}')
        print(f'path_p ={self.path_p}')
        print(f'edge_exist ={self.edge_exist}')
        print(f'name ={self.name}')
        print(f'resize ={self.resize}')
        print(f'device = {self.device}')

class TestConfig(Config):
    DATAROOT = 'dataset/'
    NAME = 'test'
    GPU = 0



