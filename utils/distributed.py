'''
author: Lingteng Qiu
data: 2021-1-26
file to help you set up distributed environments,reduce tensor and build dataloader...
'''
import os
import torch.distributed as dist
import logging
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger("logger.trainer")

def set_up_ddp(cfg,args):
    ''' set up Distributed Data Parallel env
    Parameters:
        cfg: 
        config 
    '''
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    logger.info(env_dict)
    logger.info(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    logger.info(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )
    cfg.num_gpu = dist.get_world_size()
    torch.cuda.set_device(args.local_rank)



def build_dpp_net(net):
    '''build dpp network
    first you need change all of bn layers to sybn layer
    then, you need transfer your model to ddp
    '''
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net).cuda()
    net = DDP(net,device_ids=[dist.get_rank()],find_unused_parameters=True)
    return net

def load_checkpoints(net,optimizer,resume_path,args):
    '''load checkpoints from *.pth, which includes distributed training types.
    the checkpoints includes, optimizer, epoch, net_parameters, logger, and so on.
    Parameters:
        net: model
        resume_path: .pth file, e.g. ./checkpoints/PIFu_Render_People_HG<<LR=0.001<<batch_size=2<<schedule=stoneLR/000.pth
    return:
        resume epoch
    '''
    logger.info("Loading model from {}".format(resume_path))
    if args.dist:
        #because we save local_rank = 0 into .tar, therefore you need map cuda:{id} to each rank
        map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
        checkpoints = torch.load(resume_path,map_location=map_location)
    else:
        checkpoints = torch.load(resume_path)
    net.load_state_dict(checkpoints['model_state_dict'])
    if optimizer is not None:
        #train
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    current_epoch = checkpoints['epoch']
    logger.info("Loading end!")
    logger.info("resume model from epoch {}".format(current_epoch))

    return current_epoch


def save_checkpoints(model,epoch,optimizer,checkpoint_path,args,best =False):
    '''save_checkpoints ----> *.tar
    Parameters:
        model: e.g. PIFunet
        epoch: current epoch
        optimizer: optimizer you use to solver
        path: save path
        args: option which identify whether you use distributed or not
    '''
    if best == True:
        path = os.path.join(checkpoint_path,"epoch_best.tar")
    else:
        path = os.path.join(checkpoint_path,"epoch_{:03d}.tar".format(epoch))
    if args.dist:
        if dist.get_rank() == 0:
            torch.save({'epoch':epoch,'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, path)
        dist.barrier()
    else:
        torch.save({'epoch':epoch,'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, path)
        

def reduce_tensor(tensor: torch.Tensor):
    value = tensor.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)

    value /= dist.get_world_size()
    return value
