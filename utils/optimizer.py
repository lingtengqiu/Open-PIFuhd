'''
@author:lingteng Qiu
@data: 2021-1-26
'''
import math
import numpy as np

def adjust_learning_rate(optimizer, epoch:float,cfg:dict):
    """Sets the learning rate to the initial LR decayed by schedule

    Parameters:
        optimizer: optim machine
        epoch: current epoch
        configs/PIFu_Render_People_HG.py
        
    """
    lr  = get_epoch_lr(epoch,cfg)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



def get_lr_at_epoch(cfg, cur_epoch):
    '''
    Retrieve the learning rate of the current epoch with the option to perform
    warm up in the beginning of the training stage.
    Args:
        cfg (CfgNode): configs. Details can be found in
            configs/PIFu_Render_People_HG.py
        cur_epoch (float): the number of epoch of the current training stage.
    '''
    lr = get_lr_func(cfg.lr_policy)(cfg, cur_epoch)
    # Perform warm up.
    if cur_epoch < cfg.warm_epoch:
        lr_start = cfg.lr_warm_up
        lr_end = get_lr_func(cfg.lr_policy)(
            cfg, cfg.warm_epoch
        )

        alpha = (lr_end - lr_start) / cfg.warm_epoch
        lr = cur_epoch * alpha + lr_start
    return lr


def lr_func_cosine(cfg, cur_epoch):
    '''
    Retrieve the learning rate to specified values at specified epoch with the
    cosine learning rate schedule. Details can be found in:
    Ilya Loshchilov, and  Frank Hutter
    SGDR: Stochastic Gradient Descent With Warm Restarts.
    Args:
        cfg (CfgNode): configs. Details can be found in
            configs/PIFu_Render_People_HG.py
        cur_epoch (float): the number of epoch of the current training stage.
    '''
    return max(
        cfg.LR
        * (math.cos(math.pi * cur_epoch / cfg.num_epoch) + 1.0)
        * 0.5,1e-5
    )


def lr_func_stoneLR(cfg,cur_epoch):
    '''
    Retrieve the learning rate to specified values at specified epoch with the
    stone learning rate schedule. Details can be found in:
    Pytorch version, stoneLR schedule
    Args:
        cfg (CfgNode): configs. Details can be found in
            configs/PIFu_Render_People_HG.py
        cur_epoch (float): the number of epoch of the current training stage.
        
        there exists two main keys. 'stone' indicates change steps while 'gamma' means change ratio.
    ''' 
    schedule = cfg.scheduler
    stone = schedule['stone']
    gamma = schedule['gamma']
    lr = cfg.LR
    dp = np.zeros(cfg.num_epoch)
    dp[stone]=1
    dp = np.cumsum(dp)
    return lr*(gamma**dp[int(cur_epoch)])
    

def get_lr_func(lr_policy):
    '''
    Given the configs, retrieve the specified lr policy function.
    Args:
        lr_policy (string): the learning rate policy to use for the job.
    '''
    policy = "lr_func_" + lr_policy
    if policy not in globals():
        raise NotImplementedError("Unknown LR policy: {}".format(lr_policy))
    else:
        return globals()[policy]


def get_epoch_lr(cur_epoch, cfg):
    '''
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decays.
        cur_epoch (float): the number of epoch of the current training stage.
    '''
    return get_lr_at_epoch(cfg, cur_epoch)


