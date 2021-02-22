'''
@author:lingteng qiu
@name:training method

'''
import os
from utils.structure import AverageMeter
import time
import torch.nn as nn
import torch
import utils.optimizer as optim
import logging
import torch.distributed as dist
import time
from utils.distributed import reduce_tensor,save_checkpoints
from engineer.utils.gallery import save_gallery
from engineer.core.eval import test_epoch

logger = logging.getLogger('logger.trainer')

def train_epochs(model, optimizer, cfg, args, train_loader,test_loader,resume_epoch,gallery_id):
    '''Training epoch method based on open mmlab
    Parameters:
        model: training model
        optimizer: optimizer for training
        cfg (CfgNode): configs. Details can be found in
            configs/PIFu_Render_People_HG.py
        args: option parameters
        train_loader: train dataloader iterator        
        test_loader: test dataloader file
        resume_epoch: resume epoch
        gallery_id: dir you save your gallery results
    Return:
        None
    '''
    best_iou = float('-inf')
    
    for epoch in range(resume_epoch,cfg.num_epoch):
        epoch_start_time = time.time()
        logger.info("training epoch {}".format(epoch))
        model.train()
        #define train_loss
        train_loss = AverageMeter()
        iter_data_time = time.time()
        for idx,data in enumerate(train_loader): 
            iter_start_time = time.time()
            #adjust learning rate
            lr_epoch = epoch+idx/len(train_loader)
            optim.adjust_learning_rate(optimizer,lr_epoch,cfg)
            '''For PIFu,
            name: [B] img_name_list
            img: [B, C, H, W]
            calib: [B, 4, 4]
            samples: [B, C(x,y,z), N]
            labels: [B, 1, N]
            '''
            names = data['name']
            img = data['img'].cuda()
            calib = data['calib'].cuda()
            samples = data['samples'].cuda()
            labels = data['labels'].cuda()


            if cfg.use_front:
                front_normal = data['front_normal'].cuda()
                img = torch.cat([img,front_normal],dim = 1)
            if cfg.use_back:
                back_normal = data['back_normal'].cuda()
                img = torch.cat([img,back_normal],dim = 1)

            if cfg.fine_pifu:
                #collect fine pifu datasets
                crop_query_points = data['crop_query_points'].cuda()
                crop_img = data['crop_img']
                crop_front_normal = data['crop_front_normal']
                crop_back_normal = data['crop_back_normal']
                crop_imgs = torch.cat([crop_img,crop_front_normal,crop_back_normal],dim=1).cuda()

            bs = img.shape[0]
            if not cfg.fine_pifu:
                preds,loss = model(images = img,calibs=calib,points=samples,labels=labels)
            else:
                preds,loss = model(images = img,calibs=calib,points=samples,labels=labels,crop_imgs = crop_imgs, crop_points_query = crop_query_points)
            #distributed learning
            #optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.dist:
                loss = reduce_tensor(loss)
            train_loss.update(loss.item(),bs)

            #time calculate
            iter_net_time = time.time()
            eta = int(((iter_net_time - epoch_start_time) / (idx + 1)) * len(train_loader) - (
                    iter_net_time - epoch_start_time))

            #training visible
            if idx % args.freq_plot==0:
                word_handler = 'Name: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} | LR: {5:.06f}  | dataT: {6:.05f} | netT: {7:.05f} | ETA: {8:02d}:{9:02d}:{10:02d}'.format( 
                    cfg.name,epoch,idx,len(train_loader),train_loss.avg,optim.get_lr_at_epoch(cfg,lr_epoch), 
                    iter_start_time - iter_data_time,iter_net_time - iter_start_time, 
                    int(eta//3600),int((eta%3600)//60),eta%60)
                if (not args.dist) or dist.get_rank() == 0:
                    logger.info(word_handler)
            if idx!=0 and idx % args.freq_gallery ==0:
                #gallery save
                #points [1, N]
                save_gallery(preds,samples,names,gallery_id['train'],epoch)
            iter_data_time = time.time()
        if epoch>0 and epoch % cfg.save_fre_epoch ==0:
            logger.info("save model: epoch {}!".format(epoch))
            save_checkpoints(model,epoch,optimizer,gallery_id['save_path'],args)  
        #test 
        if epoch>= cfg.start_val_epoch and epoch % cfg.val_epoch ==0:
            test_metric = test_epoch(model, cfg, args, test_loader, epoch,gallery_id['test'])
            if best_iou<test_metric['iou']:
                best_iou = test_metric['iou']
                save_checkpoints(model,epoch,optimizer,gallery_id['save_path'],args,best=True)
            
