import json
import h5py
import torch

from torch.utils.data import Dataset

from torch.utils.data.dataloader import DataLoader
import numpy as np
import random
import multiprocessing as mp
from multiprocessing import Queue, Process
import pdb


class RandomMaskingGenerator:
    def __init__(self, max_frames, mask_ratio):
        self.num_patches = max_frames
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        idx = [i for i in range(self.num_patches)]
        random.shuffle(idx)
        idx1 = idx[:(self.num_patches - self.num_mask)]
        idx2 = idx[-(self.num_patches - self.num_mask):]
        mask1 = np.ones(self.num_patches)
        mask2 = np.ones(self.num_patches)
        mask1[idx1] = 0.
        mask2[idx2] = 0.
        return [mask1, mask2]


class OverMaskingGenerator:
    def __init__(self, max_frames, mask_ratio):
        self.num_patches = max_frames
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        idx = [i for i in range(self.num_patches)]
        random.shuffle(idx)
        idx1 = idx[:(self.num_patches - self.num_mask)]
        idx2 = idx[-(self.num_patches - self.num_mask):]
        mask1 = np.ones(self.num_patches)
        mask2 = np.ones(self.num_patches)
        mask1[idx1] = 0.
        mask2[idx2] = 0.
        return [mask1, mask2]


class TrainDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.maskgenerator = RandomMaskingGenerator(cfg['max_frames'], 0.9)

        with h5py.File(cfg['train'], 'r') as h5_file:
            self.video_feats = h5_file['feats'][:]

        print(len(self.video_feats))


    def __getitem__(self, item):
        t1 = self.video_feats[item]
        mask = np.array(self.maskgenerator())
        
        return torch.tensor(t1), torch.as_tensor(mask)

    def __len__(self):
        return len(self.video_feats)

    def _set_maskprob(self, prob):
        self.maskgenerator = RandomMaskingGenerator(self.cfg['max_frames'], prob)


class TestDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg

        if cfg['dataset'] == 'activitynet':
            with h5py.File(cfg['test'], 'r') as h5_file:
                self.video_feats = h5_file['feats'][:]
            with h5py.File(cfg['query'], 'r') as h5_file:
                self.video_feats = np.concatenate((self.video_feats, h5_file['feats'][:]), \
                                                        axis=0) 
        else:
            with h5py.File(cfg['test'], 'r') as h5_file:
                self.video_feats = h5_file['feats'][:]

        print(len(self.video_feats))


    def __getitem__(self, item):
        return torch.tensor(self.video_feats[item])

    def __len__(self):
        return len(self.video_feats)




def load_data(data_set_config, batch_size, num_workers, shuffle=True, pin_memory=True):

    with open(data_set_config, 'r') as f:
        config = json.load(f)
        f.close()

    v = TrainDataset(config)
    data_loader = DataLoader(dataset=v,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return data_loader


def load_test_data(data_set_config, batch_size, num_workers, shuffle=False, pin_memory=False):

    with open(data_set_config, 'r') as f:
        config = json.load(f)
        f.close()
    
    vd = TestDataset(config)
    data_loader = DataLoader(dataset=vd,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return data_loader

