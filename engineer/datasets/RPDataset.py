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
logger = logging.getLogger('logger.trainer')



@DATASETS.register_module
class RPDataset(Dataset):
    #Note that __B_MIN and __B_max is means that bbox of valid sample zone for all images, unit cm
    __B_MIN = np.array([-128, -28, -128])
    __B_MAX = np.array([128, 228, 128])
    def __init__(self,input_dir,cache,pipeline=None,is_train=True,projection_mode='orthogonal',random_multiview=False,img_size = 512,num_views = 1,num_sample_points = 5000, \
        num_sample_color = 0,sample_sigma=5.,check_occ='trimesh',debug=False, span = 1,normal = False,sample_aim = 5):
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
            span: span step from 0 deg to 359 deg, e.g. if span == 2, deg: 0 2 4 6 ...,
            normal: whether, you want to use normal map, default False, if you want to train pifuhd, you need set it to 'True'
            sample_aim: set sample distance from mesh, according to PIFu it use 5 cm while PIFuhd choose 5 cm for coarse PIFu and 3 cm
                to fine PIFu
        Return:
            None
        '''
        super(RPDataset,self).__init__()
        self.is_train = is_train
        self.projection_mode = projection_mode
        self.input_dir=input_dir
        self.__name="Render People"
        self.img_size = img_size
        self.num_views = num_views
        self.num_sample_points = num_sample_points
        self.num_sample_color = num_sample_color
        self.sigma = sample_sigma if type(sample_sigma) == list else [sample_sigma]
        
        #view from render
        self.__yaw_list = list(range(0,360,span))
        self.__pitch_list = [0]
        self.normal = normal
        self._get_infos()
        self.subjects = self.get_subjects()
        self.random_multiview = random_multiview
        self.cache = cache
        self.check_occ =check_occ
        self.debug = debug
        self.span = span
        self.sample_aim = sample_aim


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
            debug = debug,
            span = span,
            normal = normal,
            sample_aim = sample_aim
        )

        # sampling joints
        self.sample_cloud_points()
        logger.info("sample points is already loaded!")

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
    def __build_sigma_path(self,sigma_list,subject,obj_name):
        '''build_sigma_path 
        Parameters:
            sigma_list: the number of sigmas you want to set to sample points from mesh
            subject: subject class 
            obj_name: obj's name
        return:
            [len(sigma)] list of path of file saved according to sigma_list
        '''
        if type(self.sigma) == list:
            cache_sigma_path = []
            for sigma in self.sigma:
                cache_sigma_path.append(os.path.join(self.cache,subject,obj_name.replace('.obj','_sigma_{}cm.npz'.format(sigma))))
        else:
            cache_sigma_path = [os.path.join(self.cache,subject,obj_name.replace('.obj','_sigma_{}cm.npz'.format(self.sigma)))]
        return cache_sigma_path
    def check_sigma_path(self,sigma_path_list):
        '''check whether all sigma_path exites 
        '''
        for sigma_path in sigma_path_list:
            if not os.path.exists(sigma_path):
                return False
        return True


    def sample_cloud_points(self):
        for index in tqdm(range(len(self.subjects))):
            sid,yid,pid = self.get_index(index)
            subject = self.subjects[sid]
            obj = self.obj[sid]
            path = os.path.join(obj, subject)
            obj_name = os.listdir(path)[0]
            obj_path = os.path.join(path,obj_name)
            if not os.path.exists(os.path.join(self.cache,subject)):
                os.makedirs(os.path.join(self.cache,subject),exist_ok =  True)
            
            cache_sigma_path = self.__build_sigma_path(self.sigma,subject,obj_name)
            cache_random_path = os.path.join(self.cache,subject,obj_name.replace('.obj','_random.npz'))

            if self.check_sigma_path(cache_sigma_path) and os.path.exists(cache_random_path):
                continue

            #obtain object state
            mesh = trimesh.load(obj_path)
            #like if-net, we sample points first, default, we sample 400000 points from here
            surface_points, _ = trimesh.sample.sample_surface(mesh, 400000)

            sample_points_list = []
            sample_occ_v_list = []

            for sigma in self.sigma:
                sample_points = surface_points + np.random.normal(scale=sigma, size=surface_points.shape)
                sample_points_list.append(sample_points)
            length = self.B_MAX - self.B_MIN
            random_points = np.random.rand(400000, 3) * length + self.B_MIN
            if self.check_occ =='trimesh':
                for sample_points in sample_points_list:
                    sample_occ_v = mesh.contains(sample_points)
                    sample_occ_v_list.append(sample_occ_v)
                random_occ_v = mesh.contains(random_points)
            else:
                NotImplementedError
            #save points
            for cache_path,sample_points,sample_occ_v in zip(cache_sigma_path, sample_points_list,sample_occ_v_list):
                np.savez(cache_path,points=sample_points,label=sample_occ_v)
            np.savez(cache_random_path,points=random_points,label=random_occ_v)




    
    def select_sampling_method(self, subject,path):

        '''
        sample points from mesh
        Note that original code processed data synchronized with training 
        which leads to very slow dataloader. Therefore, I use cache-idea to speed up dataloader. 
        Parameters:
            subject: the object belongs to which subjects
            path: obj_file path
        return:
            dict value where it have :
            samples: points you sample from mesh
            labels: occ value which means that whether points sampled from mesh inside or outside that 
        '''

        def load_trimesh(path):
            mesh = np.load(path)
            return mesh['points'],mesh['label']
        def _sample(points,labels,num):
            subsample_indices = np.random.randint(0, len(points), num)
            return points[subsample_indices],labels[subsample_indices]

        def _shuffle(sample_points,inside):
            index  = list(range(len(sample_points)))
            np.random.shuffle(index)
        
            return sample_points[index],inside[index]    



        
        obj_name = os.listdir(path)[0]
        cache_sigma_path = os.path.join(self.cache,subject,obj_name.replace('.obj','_sigma_{}cm.npz'.format(self.sample_aim)))
        cache_random_path = os.path.join(self.cache,subject,obj_name.replace('.obj','_random.npz'))

        assert os.path.exists(cache_sigma_path) and os.path.exists(cache_random_path)

        sample_points,sample_inside = load_trimesh(cache_sigma_path)
        random_points,random_inside = load_trimesh(cache_random_path)
        # the surface points and background surface, we have 16:1, 16 from surface and 1 from background
        #the range of surface_points 

        sample_points,sample_label = _sample(sample_points,sample_inside,4*self.num_sample_points)
        random_points,random_label = _sample(random_points,random_inside,self.num_sample_points//4)
        
        sample_points = np.concatenate([sample_points, random_points], 0)
        inside = np.concatenate([sample_label, random_label], 0)

        #shuffle two different sample data
        sample_points,inside = _shuffle(sample_points,inside)


        #ray judge outside or inside
        inside_points = sample_points[inside]
        outside_points = sample_points[np.logical_not(inside)]
        #the number of inside points
        number_inside = inside_points.shape[0]
        inside_points = inside_points[:self.num_sample_points // 2] if number_inside > self.num_sample_points // 2 else inside_points
        outside_points = outside_points[:self.num_sample_points // 2] if number_inside > self.num_sample_points // 2 \
        else outside_points[:(self.num_sample_points - number_inside)]


        samples = np.concatenate([inside_points, outside_points], 0).T
        labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1)

        samples = torch.Tensor(samples).float()
        labels = torch.Tensor(labels).float()
        return {
            'samples': samples,
            'labels': labels
        }
    def _get_infos(self):
        '''
        prepare for images-preprocessed
        '''
        input_list = os.listdir(self.input_dir)
        input_list = sorted(input_list)
        input_list = [os.path.join(self.input_dir, name) for name in input_list]
    
        self.root = []
        self.render = []
        self.mask = []
        self.param = []

        self.uv_mask = []
        self.uv_noraml = []
        self.uv_render = []
        self.uv_pos = []
        self.obj = []
        self.render_normal = []


        for root in input_list:
            render = os.path.join(root, 'RENDER')
            mask = os.path.join(root, 'MASK')
            param =  os.path.join(root, 'PARAM')
            uv_mask =  os.path.join(root, 'UV_MASK')
            uv_normal = os.path.join(root, 'UV_NORMAL')
            uv_render = os.path.join(root, 'UV_RENDER')
            uv_pos = os.path.join(root, 'UV_POS')
            obj = os.path.join(root, 'GEO', 'OBJ')    
            if self.normal:
                render_normal = os.path.join(root,'RENDER_NORMAL')     
            try:
                assert os.path.exists(render)
                assert os.path.exists(mask)
                assert os.path.exists(param)
                assert os.path.exists(uv_mask)
                assert os.path.exists(uv_normal)
                assert os.path.exists(uv_render)
                assert os.path.exists(uv_pos)
                assert os.path.exists(obj)

                if self.normal:
                    os.path.exists(render_normal)
            except:
                continue
            self.render.append(render)
            self.mask.append(mask)
            self.param.append(param)
            self.uv_mask.append(uv_mask)
            self.uv_noraml.append(uv_normal)
            self.uv_render.append(uv_render)
            self.uv_pos.append(uv_pos)
            self.obj.append(obj)
            self.root.append(root)

            if self.normal:
                self.render_normal.append(render_normal)

    def get_subjects(self):
        all_subjects = []
        var_subjects = []
        for render,root in zip(self.render,self.root):
            all_subjects +=os.listdir(render)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                var = np.loadtxt(os.path.join(root, 'val.txt'), dtype=str)
                if not len(var) ==0:
                    var_subjects+=var

        if len(var_subjects) == 0:
            return all_subjects

        if self.is_train:
            return sorted(list(set(all_subjects) - set(var_subjects)))
        else:
            return sorted(list(var_subjects))
    

    def __get_render(self, subject, num_views, yid=0, pid=0,sid=0, random_sample=False):
        '''
        Gaining the render data
        Parameters:
            subject: subject name
            num_views: how many views to return
            view_id: the first view_id. If None, select a random one.
            the sequence of pid,pid and sidthe: para and so on 
        Return:
            value that should contain some key as following
            'name': img_name e.g  savepath/rp_fernanda_posed_021/34_0_00.jpg
            'img': [num_views, C, W, H] images
            'calib': [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
            'mask': [num_views, 1, W, H] masks
        '''
        pitch = self.pitch[pid]

        # The ids are an even distribution of num_views around view_id, +1/3,+2/3
        view_ids = [self.yaw[(yid + len(self.yaw) // num_views * offset) % len(self.yaw)]
                    for offset in range(num_views)]
        if random_sample:
            view_ids = np.random.choice(self.yaw, num_views, replace=False)

        calib_list = []
        render_list = []
        mask_list = []
        extrinsic_list = []

        front_normal_list = []
        back_normal_list = []
        back_mask_list = []
        

        for vid in view_ids:
            param_path = os.path.join(self.param[sid], subject, '%d_%d_%02d.npy' % (vid, pitch, 0))
            render_path = os.path.join(self.render[sid], subject, '%d_%d_%02d.jpg' % (vid, pitch, 0))
            mask_path = os.path.join(self.mask[sid], subject, '%d_%d_%02d.png' % (vid, pitch, 0)) 

            if self.normal:
                front_render_normal_path = os.path.join(self.render_normal[sid],subject,'%d_%d_%02d.jpg' % (vid, pitch, 0))
                back_render_normal_path = os.path.join(self.render_normal[sid],subject,'%d_%d_%02d.jpg' % ((vid+180)%360, pitch, 0))
                back_mask_path = os.path.join(self.mask[sid], subject, '%d_%d_%02d.png' % ((vid+180)%360, pitch, 0)) 
                assert os.path.exists(front_render_normal_path)
                assert os.path.exists(back_render_normal_path)
                assert os.path.exists(back_mask_path)

            assert os.path.exists(param_path)
            assert os.path.exists(render_path)
            assert os.path.exists(mask_path)

            # loading calibration data
            param = np.load(param_path, allow_pickle=True)
            # pixel unit / world unit is equal to 1
            # pixel unit / uv unit ---> is ortho_ratio
            ortho_ratio = param.item().get('ortho_ratio')
            # world unit / model unit
            scale = param.item().get('scale')
            # camera center world coordinate
            center = param.item().get('center')
            # model rotation
            R = param.item().get('R')
            #translate the position of camera into world coordinate origin. 
            translate = -np.matmul(R, center).reshape(3, 1)
            extrinsic = np.concatenate([R, translate], axis=1)
            extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)

            # Match camera space to image pixel space
            scale_intrinsic = np.identity(4)
            scale_intrinsic[0, 0] = scale / ortho_ratio
            #render code, this part flip(axis =0),therefore, y need change
            scale_intrinsic[1, 1] = -scale / ortho_ratio
            scale_intrinsic[2, 2] = scale / ortho_ratio

   
            #uv space is [-1,1] we map [-256,255]->[-1,1]
            # Match image pixel space to image uv space
            uv_intrinsic = np.identity(4)
            uv_intrinsic[0, 0] = 1.0 / float(self.img_size// 2)
            uv_intrinsic[1, 1] = 1.0 / float(self.img_size // 2)
            uv_intrinsic[2, 2] = 1.0 / float(self.img_size // 2)

            # Transform under image pixel space
            trans_intrinsic = np.identity(4)


            mask = np.asarray(cv2.imread(mask_path))
            render = np.asarray(cv2.imread(render_path))  
            

            if self.normal:
                front_normal = np.asarray(cv2.imread(front_render_normal_path))
                #you need flip for 
                back_normal = np.asarray(cv2.imread(back_render_normal_path)[:,::-1,:])
                back_mask = np.asarray(cv2.imread(back_mask_path)[:,::-1,:])

            else:
                front_normal = None
                back_normal = None
                back_mask = None

            data = {
                'img':render,
                'mask':mask,
                'scale_intrinsic':scale_intrinsic,
                'trans_intrinsic':trans_intrinsic,
                'uv_intrinsic':uv_intrinsic,
                'extrinsic':extrinsic,
                'front_normal':front_normal,
                'back_normal': back_normal,
                'back_mask': back_mask,
                'flip': False
            }
        
            if self.transformer is not None:
                data = self.transformer(data)
            else:
                raise NotImplementedError
            render = data['img']
            mask = data['mask']
            calib = data['calib']
            extrinsic = data['extrinsic']

            front_normal = data['front_normal']
            back_normal =data['back_normal']
            back_mask =data['back_mask']

            

            mask_list.append(mask)
            if len(mask.shape)!=len(render.shape):
                mask = mask.unsqueeze(-1)
            if len(back_mask.shape)!=len(render.shape):
                back_mask = back_mask.unsqueeze(-1)
            #remove background
            
            render = mask.expand_as(render) * render
            render_list.append(render)
            calib_list.append(calib)
            extrinsic_list.append(extrinsic)

            
            if self.normal:
                front_normal = mask.expand_as(front_normal) * front_normal
                back_normal = back_mask.expand_as(back_normal) * back_normal
                front_normal_list.append(front_normal)
                back_normal_list.append(back_normal)
                back_mask_list.append(back_mask)
        return {
            'name':render_path,
            'img': torch.stack(render_list, dim=0),
            'calib': torch.stack(calib_list, dim=0),
            'extrinsic': torch.stack(extrinsic_list, dim=0),
            'mask': torch.stack(mask_list, dim=0),
            'front_normal':torch.stack(front_normal_list,dim=0),
            'back_normal':torch.stack(back_normal_list,dim=0),
            'back_mask': torch.stack(back_mask_list,dim=0)
        }

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
    

    def __debug(self,render_data,sample_data):
        orimg = np.uint8((np.transpose(render_data['img'][0].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, :] * 255.0)

        if self.normal:
            
            front = render_data['front_normal'][0]
            back = render_data['back_normal'][0]
            

            front = np.uint8(((np.transpose(render_data['front_normal'][0].numpy(), (1, 2, 0))+1)/2)[:, :, :] * 255.0)
            back = np.uint8(((np.transpose(render_data['back_normal'][0].numpy(), (1, 2, 0))+1)/2)[:, :, :] * 255.0)
            cv2.imshow("front",front)
            cv2.imshow("back",back)
        rot = render_data['calib'][0,:3, :3]
        trans = render_data['calib'][0,:3, 3:4]
        
        pts = torch.addmm(trans, rot, sample_data['samples'][:, sample_data['labels'][0] > 0.5])  # [3, N]
  
        pts = 0.5 * (pts.numpy().T + 1.0) * render_data['img'].size(2)
        img=orimg.copy()
        for p in pts:
            
            img = cv2.circle(img,(p[0],p[1]),2,(0,255,0),-1)
        cv2.imshow('inside', img)

        pts = torch.addmm(trans, rot, sample_data['samples'][:, sample_data['labels'][0] < 0.5])  # [3, N]
        pts = 0.5 * (pts.numpy().T + 1.0) * render_data['img'].size(2)
        img=orimg.copy()
        for p in pts:
            img = cv2.circle(img,(p[0],p[1]),2,(0,0,255),-1)
        cv2.imshow('outside', img)
        cv2.waitKey()

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
        #datasets composed from(subjects,yaw, pitch)
        #which index is not like array in c++ or python 
        #it likes arr[pitch][yaw][subjects]
        sid,yid,pid = self.get_index(index)
        
        #sequence is following:
        #pid, yid ,sid
        subject = self.subjects[sid]
        obj = self.obj[sid]
        res = {
        'name': subject,
        'mesh_path': os.path.join(obj, subject),
        'sid': sid,
        'yid': yid,
        'pid': pid,
        'b_min': self.B_MIN,
        'b_max': self.B_MAX,
        }

        #get render image
        render_data = self.__get_render(subject, num_views=self.num_views, yid=yid, pid=pid,sid=sid,
                                        random_sample=self.random_multiview)
        res.update(render_data)

        if self.num_sample_points:
            sample_data = self.select_sampling_method(subject,res['mesh_path'])
            res.update(sample_data)

        if self.debug:
            self.__debug(render_data,sample_data)
        
        return res


    def __len__(self):   
        return len(self.subjects) * len(self.yaw) * len(self.pitch)

























