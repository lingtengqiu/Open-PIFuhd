'''
file to help you build dataloader, especially for distributed dataloader
'''
import torch.distributed as dist
import torch.utils.data.distributed as data_dist
import torch
import os 
import  engineer.datasets.loader.build_loader  as loader
import numpy as np 
import random

#enum variable
COLLECT_FN={
    'train_loader_collate_fn':loader.train_loader_collate_fn,
    'test_loader_collate_fn':loader.test_loader_collate_fn,
    'carton_test_loader_collate_fn':loader.carton_test_loader_collate_fn,
    'train_fine_pifu_loader_collate_fn':loader.train_fine_pifu_loader_collate_fn,
    'test_fine_pifu_loader_collate_fn':loader.test_fine_pifu_loader_collate_fn,
}

def build_dataloader(dataset,cfg,args,phase = 'train'):
    
    batch_size = cfg.batch_size if 'train' == phase else cfg.test_batch_size
    collect_fn = COLLECT_FN[cfg.train_collect_fn] if 'train'==phase else COLLECT_FN[cfg.test_collect_fn]
    worker_init_function = worker_init_fn if 'train' == phase else worker_init_test_fn
    shuffle = 'train'==phase
    num_workers = args.num_workers
    if args.local_rank != -1:
        sampler = data_dist.DistributedSampler(dataset)
        num_workers = args.num_workers// dist.get_world_size()
        return get_dpp_loader(dataset,sampler,batch_size,num_workers,collect_fn,worker_init_function)
    else:
        return get_loader(dataset,batch_size = batch_size,num_workers = num_workers,collect_fn = collect_fn,shuffle=shuffle,worker_init_function=worker_init_function)



def get_dpp_loader(dataset,sampler, batch_size,num_workers,collect_fn,worker_init_function):
    assert sampler is not None
    return torch.utils.data.DataLoader(
        dataset,
        batch_size= batch_size,
        sampler = sampler,
        num_workers = num_workers,
        worker_init_fn = worker_init_function,
        collate_fn= collect_fn
    )

def get_loader(dataset,batch_size,num_workers,collect_fn,worker_init_function,shuffle =True):

    return torch.utils.data.DataLoader(
        dataset,
        batch_size= batch_size, num_workers= num_workers, shuffle=shuffle,collate_fn=collect_fn,
        worker_init_fn=worker_init_function)

def worker_init_fn(worker_id):
    #this step aims at avoid the same data sampled by dataloader. Therefore it gives different seeds.
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)


def worker_init_test_fn(worker_id):
    #because test of PIFu has some property, so we need fix seed.
    random.seed(1991)
    np.random.seed(1991)
    torch.manual_seed(1991)