import os
import scipy.io as sio
from model import WSCH
import torch.optim as optim
import time
import torch.nn as nn
import torch
import math
import numpy as np
from torch.autograd import grad
import torch.nn.functional as F
from functools import partial
from sklearn.cluster import KMeans
import random


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

def cosine_similarity_loss(output, target):

    cos_sim = F.cosine_similarity(output, target, dim=-1)
    loss = 1 - cos_sim.mean()
    return loss


def dcl(out_1, out_2, batch_size, temperature=0.5, tau_plus=0.1):

    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = get_negative_mask(batch_size).to(out_1.device)
    neg = neg * mask

    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    if True:
        N = batch_size * 2 - 2
        Ng = (-tau_plus * N * pos + neg.sum(dim = -1)) / (1 - tau_plus)
        Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
    else:
        Ng = neg.sum(dim=-1)

    loss = (- torch.log(pos / (pos + Ng) )).mean()
    return loss

    
def load_pretrained(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                      if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def train_model(
        train_loader=None
        , test_loader=None
        , input_size= 2048
        , hidden_size= 1024
        , decoder_size = 1024
        , hashembeding_dim=64
        , device='cuda:0'
        , max_iter=[100,300]
        , lr=1e-4
        , fileindex='0'
        , result_log_dir = None
        , result_weight_dir = None
        , weight_path = None
        , data_config = None
        , cfg = None
):
    with open(os.path.join(result_log_dir, fileindex + '.txt'), mode='a+', encoding='utf-8') as f:
        f.write('hashembeding_dim = {}\n'.format(hashembeding_dim))
        f.write('lr = {}\n'.format(lr))
        f.close()

    # Device
    use_device = torch.device(device)
        
    model = WSCH(hashembeding_dim, input_size, hidden_size, decoder_size, cfg).to(use_device)
    for name, param in model.named_parameters():
        print(name)
    total_params = sum(p.numel() for p in model.parameters())
    print("number of params: ", total_params)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if weight_path:
        state_dict = torch.load(weight_path,map_location=device)
        load_pretrained(model, state_dict['berthash'])
        optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer.load_state_dict(state_dict['optimizer'])

    # Create label
    if data_config['dataset'] == 'activitynet':
        labels = sio.loadmat(data_config['test_label'])['re_label']
        q_labels = sio.loadmat(data_config['query_label'])['q_label']
    else:
        labels = sio.loadmat(data_config['test_label'])['labels']

    labels = torch.from_numpy(labels).int()

    model.eval()

    with torch.no_grad():
        hashcode = generate_code(model, test_loader, hashembeding_dim, use_device)
        if data_config['dataset'] == 'activitynet':
            r_hashcode = hashcode[0:len(labels),:]
            q_hashcode = hashcode[len(labels):,:]
            map = mAP(
                r_hashcode.to(use_device),
                labels.to(use_device),
                use_device,
                [5, 20, 40, 60, 80, 100]
                q_hashcode.to(use_device),
                q_labels.to(use_device)
            )
        else:
            map = mAP(
                hashcode.to(use_device),
                labels.to(use_device),
                use_device,
                [5, 20, 40, 60, 80, 100]
            )
        print(map)



def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code
    Args
        dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.
    Returns
        code(torch.Tensor): Hash code.
    """
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        index = 0
        code = torch.zeros([N, code_length])
        for image_features in dataloader:
            image_features = image_features.to(device)
            hash_code = model.inference(image_features)
            batch_size = hash_code.size(0)
            code[index:index+batch_size, :] = hash_code.sign().cpu()
            index += batch_size

    return code



def mAP(hashcode, labels, device, k, q_hashcode=None, q_labels=None):

    nbits = hashcode.shape[1]
    if (q_hashcode==None)
        q_hashcode=hashcode
    if (q_labels==None)
        q_labels = labels
    IX = -torch.mm(hashcode, q_hashcode.t())
    del hashcode
    IX = IX + nbits
    IX = IX * 0.5

    _, IX = torch.topk(IX, k=k[-1], dim=0, largest=False)

    numTrain, numTest = IX.shape
    #print(IX.shape)
    
    res = []
    for i in k:
        range_tensor = torch.arange(1, i+1, dtype=torch.float32, device=device)

        retrieved_labels = labels[IX[:i]]
        retrieved_labels = retrieved_labels.permute(1, 0, 2)
    
        relevance = torch.sum(retrieved_labels * q_labels.unsqueeze(1), dim=2)
        relevance = torch.clamp(relevance, max=1)

        cumsum_relevance = torch.cumsum(relevance, dim=1) * relevance

        precision = cumsum_relevance / range_tensor.unsqueeze(0)

        ap = torch.sum(precision, dim=1)

        res.append('{:.5f}'.format(ap.mean().item() / i))

    torch.cuda.empty_cache()
    return res
