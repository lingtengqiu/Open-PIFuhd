'''
@author:lingteng qiu
@name:Open-Pifu inference
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
from engineer.core.eval import  test_epoch,inference
from utils import group_weight
from utils.logger import info_cfg,get_experiments_id,setup_test_logger
import torch
import torch.optim as optim
from utils.dataloader import build_dataloader
from  utils.distributed import set_up_ddp,build_dpp_net,load_checkpoints
import torch.distributed as dist
from tqdm import tqdm 
import logging 

logger = logging.getLogger("logger.trainer")



if __name__ == "__main__":
    args = opt
    assert args.config is not None,"you must give your model config"
    cfg = Config.fromfile(args.config)


    if cfg.logger :
        logger=setup_test_logger(cfg.name,rank= args.local_rank)
    if args.dist:
        logger.info("Using Distributed test!")
        # env setup
        set_up_ddp(cfg,args)
    
    info_cfg(logger,cfg)  


    test_data_set = build_dataset(cfg.data.test)
    test_dataloader = build_dataloader(test_data_set,cfg,args,phase='test')
    logger.info("test data size:{}".format(len(test_data_set)))

    #build model
    model = build_model(cfg.model)
    if args.dist == True:
        #build distributed network
        model = build_dpp_net(model)
    else:
        model = model.cuda()
    #resume
    checkpoints_path,gallery_id = get_experiments_id(cfg)
    resume_path = os.path.join(checkpoints_path,"epoch_best.tar".format(args.current))
    
    epoch = load_checkpoints(model,None,resume_path,args)

    inference(model, cfg, args, test_dataloader, epoch,gallery_id['test'],len(test_data_set.subjects))



