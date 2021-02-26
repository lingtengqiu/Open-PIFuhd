'''
@author:lingteng qiu
@name:Open-Pifu
'''
import sys
sys.path.append("./")
from opt import opt
from mmcv import Config
import os
from torch.utils.data import DataLoader
import torch.nn as nn
from engineer.datasets.loader.build_loader import train_loader_collate_fn,test_loader_collate_fn
from engineer.datasets.builder import build_dataset
from engineer.models.builder import build_model
from engineer.core.train import  train_epochs
from utils import group_weight
from utils.logger import setup_logger,info_cfg,get_experiments_id
import torch
import torch.optim as optim
from utils.dataloader import build_dataloader

from  utils.distributed import set_up_ddp,build_dpp_net,load_checkpoints
import torch.distributed as dist

from tqdm import tqdm 





if __name__ == "__main__":

    args = opt
    assert args.config is not None,"you must give your model config"
    cfg = Config.fromfile(args.config)
    
    if cfg.logger :
        logger=setup_logger(cfg.name,rank= args.local_rank)


    if args.dist:
        logger.info("Using Distributed Training!")
        # env setup
        set_up_ddp(cfg,args)
    
    #print cfg env
    info_cfg(logger,cfg)

    #build dataset
    train_data_set = build_dataset(cfg.data.train)

    train_dataloader = build_dataloader(train_data_set,cfg,args)
    test_data_set = build_dataset(cfg.data.test)

    #debug datasets
    for i in range(len(train_data_set)):
        train_data_set[i]
    for i in range(len(test_data_set)):
        test_data_set[i]

    test_dataloader = build_dataloader(test_data_set,cfg,args,phase='test')
    logger.info("train data size:{}".format(len(train_data_set)))
    logger.info("test data size:{}".format(len(test_data_set)))

    #build model
    model = build_model(cfg.model)


    if args.dist == True:
        #build distributed network
        model = build_dpp_net(model)
    else:
        model = model.cuda()

    #optimal methods 
    optim_dict = cfg.optim_para['optimizer']
    cfg.LR =  optim_dict['lr']
    if 'RMSprop' == optim_dict['type']:
        optimizer = optim.RMSprop(model.parameters(),lr=optim_dict['lr'],momentum=optim_dict['momentum'],weight_decay=optim_dict['weight_decay'])
    elif 'adam' == optim_dict['type']:
            optimizer = optim.Adam(model.parameters(),lr = optim_dict['lr'])
    num_epoch = cfg.num_epoch
    logger.info(optimizer)

    #resume 
    checkpoints_path,gallery_id = get_experiments_id(cfg)
    if True == args.resume:
        resume_path = os.path.join(checkpoints_path,"epoch_{:03d}.tar".format(args.current))
        if not os.path.exists(resume_path):
            raise LookupError("we could not find {}".format(resume_path))
        else:
            epoch = load_checkpoints(model,optimizer,resume_path,args)
        epoch+=1
    else:
        epoch = 0
    gallery_id['save_path'] = checkpoints_path
    #train model, details see in engineer/core/train.py
    train_epochs(model, optimizer, cfg, args, train_dataloader,test_dataloader,epoch,gallery_id)


