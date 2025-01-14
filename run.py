import argparse
import os
import random
import json
import numpy as np
import torch
from dataset import load_data, load_test_data     # 修改
from pretrain import train_model        # 修改

result_log_dir = './log'
result_weight_dir = './weight'

def getfileindex():
    try:
        files = os.listdir(result_log_dir)
        fileindex = sorted([int(x.split('.')[0]) for x in files ])[-1]
    except:
        fileindex = 0

    return fileindex+1

class Cfg:
    def __init__(self, a, b, clus_temperature, n_clusters, temperature, tau_plus, p1, p2):
        self.a = a
        self.b = b
        self.clus_temperature = clus_temperature
        self.n_clusters = n_clusters
        self.temperature = temperature
        self.tau_plus = tau_plus
        self.p1 = p1
        self.p2 = p2
        

def run():

    device = 'cuda:0' 
    data_set_config = './Json/Anet.json'
    num_workers = 8
    max_iter = [100, 300]
    lr = 1e-4
    seed = 3346
    batch_size = 128
    hidden_size = 1024
    decoder_size = 1024
    hashcode_size = 64
    
    weight_path = None

    cfg = Cfg(0.02, 1.0, 0.2, 800, 0.5, 0.1, 0.9, 0.6)


    # Set seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    with open(data_set_config, 'r') as f:
        config = json.load(f)
        f.close()
    train_val_data_path = (config['train'],config['test'],config['query'])
    input_size = config['hidden_size']

    print("train device:", device)
    print("data_path:", train_val_data_path)
    print("num_workers: ", num_workers)
    print("lr:",lr)
    print("hashcode_size:",hashcode_size)
    print("batch_size:", batch_size)
    print(vars(cfg))

    # Load dataset
    dataloader = load_data(
        data_set_config = data_set_config
        ,batch_size = batch_size
        ,num_workers = num_workers                
    )

    qr_dataloader = load_test_data( 
        data_set_config = data_set_config
        ,batch_size = batch_size
        ,num_workers = num_workers
    )

    fileindex = str(getfileindex())
    print(fileindex)

    with open(os.path.join(result_log_dir, fileindex + '.txt'), mode='a+', encoding='utf-8') as f:
        f.write('train device: {}\n'.format(device))
        f.write('data_path: {}\n'.format(train_val_data_path))
        f.write('seed = {}\n'.format(seed))
        f.write('batch_size = {}\n'.format(batch_size))
        f.write('num_workers = {}\n'.format(num_workers))
        f.close()

    # Training
    train_model(
        train_loader=dataloader
        , test_loader=qr_dataloader
        , input_size=input_size
        , hidden_size=hidden_size
        , decoder_size = decoder_size
        , hashembeding_dim=hashcode_size
        , device=device
        , max_iter= max_iter
        , lr= lr
        , fileindex=fileindex
        , result_log_dir=result_log_dir
        , result_weight_dir=result_weight_dir
        , weight_path = weight_path
        , data_config = config
        , cfg = cfg
    )

if __name__ == '__main__':

    run()

