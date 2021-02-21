'''
@author: lingteng qiu
@data  : 2021-1-21
RenderPeople Dataset:https://renderpeople.com/
'''
import cv2
import sys
sys.path.append("./")
from torch.utils.data import Dataset
import json
import os
import numpy as np
import random
import torch
import scipy.sparse as sp
from .pipelines import Compose
from .registry import DATASETS
import warnings
from PIL import Image,ImageOps
import torchvision.transforms as transforms
import trimesh
import numpy as np
from tqdm import tqdm
import logging
import glob
logger = logging.getLogger('logger.trainer')
from skimage import draw
import json


@DATASETS.register_module
class Carton_Dataset(Dataset):
    #Note that __B_MIN and __B_max is means that bbox of valid sample zone for all images, unit cm
    __B_MIN = np.array([-128, -28, -128])
    __B_MAX = np.array([128, 228, 128])
    def __init__(self,input_dir,cache,pipeline=None,is_train=True,projection_mode='orthogonal',random_multiview=False,img_size = 512,num_views = 1,num_sample_points = 5000, \
        num_sample_color = 0,sample_sigma=5.,check_occ='trimesh',debug=False):
        '''
        Render People Dataset
        Parameters:
            input_dir: file direction e.g. Garmant/render_people_gen， in this file you have some subfile direction e.g. rp_kai_posed_019_BLD
            caceh: memeory cache which is employed to save sample points from mesh. Of course, we use it to speed up data loaded. 
            pipeline: the method which process datasets, like crop, ColorJitter and so on.
            is_train: phase the datasets' state
            projection_mode: orthogonal or perspective
            num_sample_points: the number of sample clounds from mesh 
            num_sample_color: the number of sample colors from mesh, default 0, means train shape model
            sample_sigma: the distance we disturb points sampled from surface. unit: cm e.g you wanna get 5cm, you need input 5
            check_occ: which method, you use it to check whether sample points are inside or outside of mesh. option: trimesh |
            debug: debug the dataset like project the points into img_space scape
        Return:
            None
        '''
        super(Carton_Dataset,self).__init__()
        self.is_train = is_train
        self.projection_mode = projection_mode
        self.input_dir=input_dir
        self.__name="Carton People"
        self.img_size = img_size
        self.num_views = num_views
        self.num_sample_points = num_sample_points
        self.num_sample_color = num_sample_color
        self.sigma = sample_sigma
        
        #view from render
        self.__yaw_list = [0]
        self.__pitch_list = [0]
        self._get_infos()
        self.subjects = self.get_subjects()
        self.random_multiview = random_multiview
        self.cache = cache
        self.check_occ =check_occ
        self.debug = debug


        if not pipeline == None:
            #color ColorJitter,blur,crop,resize,totensor,normalize .....
            self.transformer  = Compose(pipeline)
        else:
            self.transformer = None

        self.input_para=dict(
            input_dir=input_dir,
            is_train=is_train,
            projection_mode = projection_mode,
            pipeline = self.transformer,
            img_size = img_size,
            num_views = num_views,
            num_sample_points = num_sample_points,
            num_sample_color = num_sample_color,
            random_multiview = random_multiview,
            sample_sigma=sample_sigma,
            cache=cache,
            check_occ=check_occ,
            debug = debug
        )

        # sampling joints
        None

        #transform method or pipeline method

        self.to_tensor = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    

    def get_index(self,index):
        sid = index % len(self.subjects)
        tmp = index // len(self.subjects)
        yid = tmp % len(self.yaw)
        pid = tmp // len(self.yaw)
        return sid,yid,pid 

    def _get_infos(self):
        '''
        prepare for images-preprocessed
        '''
        img_list  = [glob.glob(os.path.join(self.input_dir,"*.{}".format(item))) for item in ["jpeg",'jpg']]
        self.img_list = []
        for _ in img_list:
            self.img_list.extend(_)
        self.img_list = sorted(self.img_list)
        self.json_list = sorted(glob.glob(os.path.join(self.input_dir,"*.json")))
    def get_subjects(self):
        subjects = []
        for name in self.img_list:
            subjects.append(name.split('/')[-1][:-5])
        return subjects


    #*********************property********************#
    @property 
    def yaw(self):
        return self.__yaw_list
    @property
    def pitch(self):
        return self.__pitch_list

    @property
    def B_MAX(self):
        return self.__B_MAX
    @property
    def B_MIN(self):
        return self.__B_MIN
    
    #******************magic method*****************#
    def __repr__(self):
        __repr = "{}(Parameters: ".format(self.__name)
        for key in self.input_para.keys():
            __repr+="{}:{}, ".format(key,self.input_para[key])
        __repr=__repr[:-2]
        return __repr+')'
    def __getitem__(self,index):
        '''
        Capturing data from datasets according to input index
        Parameters:
            index: type(int)(0-len(self))
        return:
        '''
        sid,yid,pid = self.get_index(index)
        
        img_name = self.img_list[sid]
        annotation = self.json_list[sid]
        with open(annotation) as reader:
            an = json.load(reader)
        points = (an['shapes'][0]['points'])
        points = np.asarray(points)
        w, h = an['imageWidth'],an['imageHeight']
        points = points[:,::-1]
        mask = draw.polygon2mask((h,w),points)
        mask = np.asarray(mask,dtype=np.uint8)
        mask[mask>0] =255
        img = cv2.imread(img_name)
        img[mask==0]=0
        mask = mask[...,None]
        mask = np.repeat(mask,3,axis=2)
        
        tmp_keys = img_name.split("/")
        keys = []
        keys+=tmp_keys[:-1]
        keys.append(tmp_keys[-1].replace('.jpeg',''))
        keys.append(tmp_keys[-1])
        img_name = os.path.join(*keys)

        val =  dict(
            name = img_name,
            img=img,mask=mask
        )

        val = self.transformer(val)
        val['img'] = val['mask'].expand_as(val['img']) * val['img']
        
        return val

    def __len__(self):   
        return len(self.subjects) * len(self.yaw) * len(self.pitch)