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


    since_time = time.time()

    for epoch in range(0, max_iter[1]):
        # Logger
        print('Stage 1 Epoch {}/{}'.format(epoch + 1, max_iter[1]))
        print('-' * 20)
        with open(os.path.join(result_log_dir, fileindex + '.txt'), mode='a+', encoding='utf-8') as f:
            f.write('Epoch {}/{}\n'.format(epoch + 1, max_iter[1]))
            f.write('-' * 20)
            f.write('\n')
            f.close()

        # Training

        time_start = time.time()
        model.eval()

        t_loss = [0.0, 0.0, 0.0]

        print("Initializing cluster centers...")

        with torch.no_grad():
            features = []
            train_loader.dataset._set_maskprob(cfg.p1)

            for index, data in enumerate(train_loader):
                I, mask = data
                I = I.to(device)
                mask2 = mask[:,1,:].to(device).flatten(1).to(torch.bool)
                mask1 = mask[:,0,:].to(device).flatten(1).to(torch.bool)
                
                I_ = model.inference(I, mask1)
                features.append(I_)
                I_ = model.inference(I, mask2)
                features.append(I_)
        
            features = torch.cat(features, dim=0)

            if (model.initialized == False):
                model.initialized = True
                kmeans = KMeans(n_clusters=cfg.n_clusters, random_state=0, algorithm='elkan')
            else:
                kmeans = KMeans(n_clusters=cfg.n_clusters, random_state=0, init=model.cluster_centers.detach().cpu().numpy(), algorithm='elkan')
            kmeans.fit(features.sign().cpu().numpy())
            new_centers = torch.tensor(kmeans.cluster_centers_, device=use_device).sign()
            model.cluster_centers = new_centers

            print("Cluster centers initialized.")
            del features

        model.train()

        # step2
        train_loader.dataset._set_maskprob(cfg.p2)

        for index, data in enumerate(train_loader):

            if index % 50 == 0:
                print(f'{epoch} {index}')

            I, mask = data

            bool_masked_pos_1 = mask[:,0,:].to(use_device, non_blocking=True).flatten(1).to(torch.bool)
            bool_masked_pos_2 = mask[:,1,:].to(use_device, non_blocking=True).flatten(1).to(torch.bool)

            I = I.to(use_device)

            optimizer.zero_grad()
            
            batch, L, D = I.shape

            frame_1, vh1 = model.forward(I, bool_masked_pos_1)
            frame_2, vh2 = model.forward(I, bool_masked_pos_2)
            
            labels_1 = I[bool_masked_pos_1].reshape(batch, -1, D)
            labels_2 = I[bool_masked_pos_2].reshape(batch, -1, D)

            recon_loss = cosine_similarity_loss(frame_1, labels_1) + cosine_similarity_loss(frame_2, labels_2) 
            
            cluster_loss1 = model.compute_cluster_probability(vh1)
            cluster_loss2 = model.compute_cluster_probability(vh2)
        
            cluster_loss = cluster_loss1 + cluster_loss2
            
            loss = recon_loss + cluster_loss * cfg.a
            t_loss[2] += cluster_loss.item()

            loss.backward()
            optimizer.step()

        # step3
        train_loader.dataset._set_maskprob(cfg.p2)
        
        for index, data in enumerate(train_loader):

            if index % 50 == 0:
                print(f'{epoch} {index}')

            I, mask = data

            bool_masked_pos_1 = mask[:,0,:].to(use_device, non_blocking=True).flatten(1).to(torch.bool)
            bool_masked_pos_2 = mask[:,1,:].to(use_device, non_blocking=True).flatten(1).to(torch.bool)

            I = I.to(use_device)

            optimizer.zero_grad()
            
            batch, L, D = I.shape

            frame_1, vh1 = model.forward(I, bool_masked_pos_1)
            frame_2, vh2 = model.forward(I, bool_masked_pos_2)

            labels_1 = I[bool_masked_pos_1].reshape(batch, -1, D)
            labels_2 = I[bool_masked_pos_2].reshape(batch, -1, D)

            recon_loss = cosine_similarity_loss(frame_1, labels_1) + cosine_similarity_loss(frame_2, labels_2) 

            contra_loss = dcl(vh1, vh2, batch, temperature=cfg.temperature, tau_plus=cfg.tau_plus) 
            
            loss = recon_loss + contra_loss * cfg.b

            t_loss[0] += recon_loss.item()
            t_loss[1] += contra_loss.item()
            loss.backward()
            optimizer.step()


        epoch_loss = ['{:.6f}'.format(x / (len(train_loader.dataset) / 128)) for x in t_loss]

        print('Train Loss: ' + str(epoch_loss))
        time_end = time.time() - time_start
        print('Epoch training in {:.0f}m {:.0f}s'.format(time_end // 60, time_end % 60))

        with open(os.path.join(result_log_dir, fileindex + '.txt'), mode='a+', encoding='utf-8') as f:
            f.write('Train Loss: ' + str(epoch_loss))
            f.close()

        
        if epoch == max_iter[1]-1:
            model_state_dict = model.state_dict()
            dir = os.path.join(result_weight_dir, fileindex + '_' + str(hashembeding_dim) + '.pth')
            state = {
                'berthash': model_state_dict,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1
            }
            torch.save(state, dir)
            print(str(epoch + 1) + 'saved')
        
    time_elapsed = time.time() - since_time
    print('Stage 2 training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
