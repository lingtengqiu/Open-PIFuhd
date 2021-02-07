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
import glob 
from engineer.utils.metrics import computer_metrics
from utils.structure import AverageMeter
from engineer.render.builder import build_render

logger = logging.getLogger("logger.trainer")



import engineer.render.face3d as face3d
from engineer.render.face3d import mesh
import trimesh
import cv2
from PIL import Image
import numpy as np

def generate(dataset):
    for i in range(0,len(dataset.subjects)):
        obj = dataset[i]
        target_mesh_path = obj['mesh_path']
        name = target_mesh_path.split("/")[-1]
        target = glob.glob(os.path.join(target_mesh_path,"*.obj"))[-1]
        tgt_mesh = trimesh.load(target)
        for j in tqdm(range(0,360)):
            obj = dataset[i+j*len(dataset.subjects)]
            normal_path = target_mesh_path.replace('GEO/OBJ/','RENDER_NORMAL/')
            if not os.path.exists(normal_path):
                os.makedirs(normal_path,exist_ok =True)
            img_mask = np.transpose(obj['mask'][0].numpy(), (1, 2, 0))[...,0]
            # tutorial for how to use our normal render class 
            render.camera = obj['calib']  
            render.set_mesh(tgt_mesh.vertices, tgt_mesh.faces)
            render.set_world_view_point(0)
            render.world2uv()
            render.set_attribute()
            render.draw()
            tgt_img,tgt_mask = render.get_render()
            tgt_img = np.asarray(tgt_img).copy()
            tgt_img[img_mask==0] = 0
            img = Image.fromarray(tgt_img)
            img.save(os.path.join(normal_path,'{}_0_00.jpg'.format(j)))



if __name__ == "__main__":
    args = opt
    assert args.config is not None,"you must give your model config"
    cfg = Config.fromfile(args.config)
    render = build_render(cfg.render_cfg)
    if cfg.logger :
        logger=setup_test_logger(cfg.name,rank= args.local_rank)
    if args.dist:
        logger.info("Using Distributed test!")
        # env setup
        set_up_ddp(cfg,args)
    
    info_cfg(logger,cfg)  

    train_dataset = build_dataset(cfg.data.train) 
    test_data_set = build_dataset(cfg.data.test)
    
    # generate(train_dataset)
    generate(test_data_set)