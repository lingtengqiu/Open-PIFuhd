'''
author: lingteng qiu
date: 2021-1-21
'''
import logging
import time
import os 

def setup_logger(path,rank=-1):
    '''set up env logger

    Parameters:
        path: where you save logger handler
    
    return:
        logger object
    '''
    save_path = os.path.join(os.getcwd(),'checkpoints',path,'logger')
    rq = time.strftime('%Y-%m-%d-%H', time.localtime(time.time()))
    current_env = os.path.join(save_path,rq)
    os.system('mkdir -p {}'.format(current_env))
    logger_name = os.path.join(current_env,'logger.trainer')
    logger = logging.getLogger('logger.trainer')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s - %(filename)s line:%(lineno)d - %(levelname)s]: %(message)s",datefmt='%Y-%m-%d %H:%M:%S')

    if rank == -1 or rank == 0:

        fh = logging.FileHandler(logger_name, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info("Set up logger\n:{}".format(logger_name))

    return logger



def setup_test_logger(path,rank=-1):
    '''set up env test logger

    Parameters:
        path: where you save logger handler
    
    return:
        logger object
    '''
    logger = logging.getLogger('logger.trainer')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s - %(filename)s line:%(lineno)d - %(levelname)s]: %(message)s",datefmt='%Y-%m-%d %H:%M:%S')

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


    return logger

def info_cfg(logger,cfg):
    '''inshow trainer parameters
    
    Parameters:
        logger: logger obj
        cfg: input parameters, generally, dict
    
    return:
        None
    '''
    cfg_dict = cfg._cfg_dict.to_dict()
    __repr__ ="Parameters:\n"
    for key in cfg_dict.keys():
        __repr__+="{}: {}\n".format(key,cfg_dict[key])
    logger.info(__repr__[:-1])
    
def get_experiments_id(cfg):
    '''
    build checkpoints file folder
    '''
    exp_id = os.path.join(cfg.checkpoints,cfg.name,'weight/')+'LR={LR}-batch_size={batch_size}-schedule={schedule}'.format(LR=cfg.LR,batch_size=cfg.batch_size,schedule=cfg.lr_policy)
    gallery_train_path = os.path.join(cfg.checkpoints,cfg.name,'gallery/train/')+'LR={LR}-batch_size={batch_size}-schedule={schedule}'.format(LR=cfg.LR,batch_size=cfg.batch_size,schedule=cfg.lr_policy)
    gallery_test_path = os.path.join(cfg.checkpoints,cfg.name,'gallery/test/')+'LR={LR}-batch_size={batch_size}-schedule={schedule}'.format(LR=cfg.LR,batch_size=cfg.batch_size,schedule=cfg.lr_policy)
    if not os.path.exists(exp_id):
        os.system("mkdir -p {}".format(exp_id))

    if not os.path.exists(gallery_train_path):
        os.system("mkdir -p {}".format(gallery_train_path))
    if not os.path.exists(gallery_test_path):
        os.system("mkdir -p {}".format(gallery_test_path))

    return exp_id,{"train":gallery_train_path,'test':gallery_test_path}