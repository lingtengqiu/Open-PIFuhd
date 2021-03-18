import  numpy as np
from collections import defaultdict
import json
from tqdm import tqdm
import cv2
import os
import torch
import copy
from engineer.utils.metrics import *
from utils.structure import AverageMeter
from utils.distributed import reduce_tensor
import time
import logging 
import torch.distributed as dist
from engineer.utils.mesh_utils import gen_mesh
import torch.nn.functional as F
logger = logging.getLogger('logger.trainer')
import numpy as np 

torch.nn.NLLLoss

def inference(model, cfg, args, test_loader, epoch,gallery_id,gallery_time=73):
    model.eval()
    watched = 0
    for batch in test_loader:
        watched+=1
        if watched>gallery_time:
            break
        logger.info("time {}/{}".format(watched,gallery_time))
        B_MIN = np.array([-1, -1, -1])
        B_MAX = np.array([1, 1, 1])
        projection_matrix = np.identity(4)
        # projection_matrix[1, 1] = -1
        calib = torch.Tensor(projection_matrix).float().cuda()

        name = batch['name'][0]
        img = batch['img']


        if cfg.use_front:
            front_normal = batch['front_normal']
            img = torch.cat([img,front_normal],dim = 1)
        if cfg.use_back:
            back_normal = batch['back_normal']
            img = torch.cat([img,back_normal],dim = 1)
        
        if cfg.fine_pifu:
            #collect fine pifu datasets
            crop_query_points = batch['crop_query_points'].cuda()
            crop_img = batch['crop_img']
            crop_front_normal = batch['crop_front_normal']
            crop_back_normal = batch['crop_back_normal']
            crop_imgs = torch.cat([crop_img,crop_front_normal,crop_back_normal],dim=1).cuda()
        else:
            crop_imgs = None
            crop_query_points = None

        try:
            origin_calib = batch['calib'][0]
        except:
            #there is not origina calib matrix
            origin_calib = None
            projection_matrix[1, 1] = -1
            calib = torch.Tensor(projection_matrix).float().cuda()
        '''***************************************'''
        #crop_upper_body
        # h0 = int(np.random.rand()*20)
        # h1 = int(h0+128 + np.random.rand()* 256)   
        # w0 = int(64+np.random.rand()*128)
        # w1 = int(w0+128 + np.random.rand()* 256)
        # h_scale = (h1 - h0)/512
        # w_scale = (w1 - w0)/512
        # h0 = h0/512
        # w0 = w0/512
        # h0 = int(h0*1024)
        # w0 = int(w0*1024)
        # h_scale = int(h_scale*1024)
        # w_scale = int(w_scale*1024)
        # h1 = h0+h_scale
        # w1 = w0+w_scale
        # crop_imgs = crop_imgs[...,h0:h1,w0:w1]
        # diff = abs(h_scale - w_scale)//2
        # crop_imgs = F.pad(crop_imgs,(diff,diff,0,0),'constant',0.) if h_scale>w_scale else F.pad(crop_imgs,(0,0,diff,diff),'constant',0.)
        # crop_imgs_ori = crop_imgs.clone()
        # crop_imgs = F.interpolate(crop_imgs_ori,(1024,1024)).cuda()
        # img = F.interpolate(crop_imgs_ori,(512,512)).cuda()
        # origin_calib = None
        # projection_matrix[1, 1] = -1
        # calib = torch.Tensor(projection_matrix).float().cuda()
        '''***************************************'''

        save_gallery_path = os.path.join(gallery_id,name.split('/')[-2])
        os.makedirs(save_gallery_path,exist_ok=True)

        data=    {'name': name,
            'img': img,
            'calib': calib.unsqueeze(0),
            'mask': None,
            'b_min': B_MIN,
            'b_max': B_MAX,
            'origin_calib':origin_calib,
            'crop_img':crop_imgs,
            'crop_query_points':crop_query_points
        }
        with torch.no_grad():
            gen_mesh(cfg,model,data,save_gallery_path)


def test_epoch(model, cfg, args, test_loader, epoch,gallery_id):
    '''test epoch
    Parameters:
        model:
        cfg:
        args:
        test_loader:
        epoch: current epoch
        gallery_id: gallery save path
    Return:
        test_metrics
    '''
    model.eval()
    
    iou_metrics = AverageMeter()
    prec_metrics = AverageMeter()
    recall_metrics = AverageMeter()
    error_metrics =AverageMeter()
    epoch_start_time = time.time()

    
    with torch.no_grad():
        for idx,data in enumerate(test_loader):  
            image_tensor = data['img'].cuda()
            calib_tensor = data['calib'].cuda()
            sample_tensor = data['samples'].cuda()
            label_tensor = data['labels'].cuda()
            if cfg.use_front:
                front_normal = data['front_normal'].cuda()
                image_tensor = torch.cat([image_tensor,front_normal],dim = 1)
            if cfg.use_back:
                back_normal = data['back_normal'].cuda()
                image_tensor = torch.cat([image_tensor,back_normal],dim = 1)



            if cfg.fine_pifu:
                #collect fine pifu datasets
                crop_query_points = data['crop_query_points'].cuda()
                crop_img = data['crop_img']
                crop_front_normal = data['crop_front_normal']
                crop_back_normal = data['crop_back_normal']
                crop_imgs = torch.cat([crop_img,crop_front_normal,crop_back_normal],dim=1).cuda()

            bs = image_tensor.shape[0]
            if not cfg.fine_pifu:
                res, error = model(image_tensor, sample_tensor, calib_tensor, labels=label_tensor)
            else:
                #debug use
                #res, error = model.global_net(image_tensor, sample_tensor, calib_tensor, labels=label_tensor)

                #inference use
                res,error = model(images = image_tensor,calibs=calib_tensor,points=sample_tensor,labels=label_tensor,crop_imgs = crop_imgs, crop_points_query = crop_query_points)
            IOU, prec, recall = compute_acc(res,label_tensor)
            if args.dist:
                error = reduce_tensor(error)
                IOU = reduce_tensor(IOU)
                prec =  reduce_tensor(prec)
                recall =  reduce_tensor(recall) 
            error_metrics.update(error.item(),bs)
            iou_metrics.update(IOU.item(),bs)
            prec_metrics.update(prec.item(),bs)
            recall_metrics.update(recall.item(),bs)

            iter_net_time = time.time()
            eta = int(((iter_net_time - epoch_start_time) / (idx + 1)) * len(test_loader) - (
                    iter_net_time - epoch_start_time))

            word_handler = 'Test: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} | IOU: {5:.06f}  | prec: {6:.05f} | recall: {7:.05f} | ETA: {8:02d}:{9:02d}:{10:02d}'.format( 
                cfg.name,epoch,idx,len(test_loader),error_metrics.avg,iou_metrics.avg, 
                prec_metrics.avg,recall_metrics.avg, 
                int(eta//3600),int((eta%3600)//60),eta%60)
            if (not args.dist) or dist.get_rank() == 0:
                logger.info(word_handler)
        logger.info("Test Final result | Epoch: {0:d} | Err: {1:.06f} | IOU: {2:.06f}  | prec: {3:.05f} | recall: {4:.05f} |".format(
            epoch, error_metrics.avg, iou_metrics.avg, prec_metrics.avg, recall_metrics.avg
            ))
    return dict(error=error_metrics.avg,iou=iou_metrics.avg,recall = recall_metrics.avg,pre =prec_metrics.avg)

            

            


    


