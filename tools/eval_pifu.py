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


    test_data_set = build_dataset(cfg.data.test)



    checkpoints_path,gallery_id = get_experiments_id(cfg)
    test_save_path = gallery_id['test']

    chamfer_distance = AverageMeter()
    normal = AverageMeter()
    p2s = AverageMeter()
    for i in range(len(test_data_set.subjects)):
        obj = test_data_set[i]
        target_mesh_path = test_data_set[i]['mesh_path']
        name = target_mesh_path.split("/")[-1]
        pred_path = os.path.join(test_save_path,name)
        target = glob.glob(os.path.join(target_mesh_path,"*.obj"))[-1]
        pred = glob.glob(os.path.join(pred_path,"*.obj"))[-1]
        




        #tutorial for how to use our normal render class 
        # render.camera = obj['calib']  
        # tgt_mesh = trimesh.load(target)
        # pred_mesh = trimesh.load(pred)
        # render.set_mesh(tgt_mesh.vertices, tgt_mesh.faces)
        # render.set_world_view_point(90)
        # render.world2uv()
        # render.set_attribute()
        # render.draw()
        # tgt_img = render.get_render()


        # render.set_mesh(pred_mesh.vertices, pred_mesh.faces)
        # render.set_world_view_point(90)
        # render.world2uv()
        # render.set_attribute()
        # render.draw()
        # pred_img = render.get_render() 
        # tgt_img.show()
        # pred_img.show()



        normal_loss,chamfer_loss,p2s_loss = computer_metrics(pred,target)
        chamfer_distance.update(chamfer_loss,1)
        normal.update(normal_loss,1)
        p2s.update(p2s_loss,1)
        logger.info("{}/{}\tchamfer_loss:{:.4f}, normal_loss:{:.4f}, p2s:{:.4f}".format(i,len(test_data_set.subjects),
        chamfer_distance.avg,normal.avg,p2s.avg))


