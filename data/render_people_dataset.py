import os.path
import os 
import glob
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch

class RenderPeopleDataset(BaseDataset):
    def find_root_path(self,root,span = 6, key='RENDER/*/*.jpg'):
        files = os.listdir(root)
        files = [os.path.join(root,file) for file in files]
        heads = []
        for file in files:
            head = glob.glob(os.path.join(file,key))
            head =sorted(head)[::span]
            heads.extend(head)
        return heads
    def __check_A_B(self):
        for A,B in zip(self.A_paths,self.B_paths):
            if os.path.join(*A.split("/")[-2:]) != os.path.join(*B.split("/")[-2:]):
                return False
        return True
    def __check_B_exists(self):
        for B in self.B_paths:
            if not os.path.exists(B):
                return False
        return True

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot  

        ### input A (label maps)
        self.A_paths = sorted(self.find_root_path(self.root,span=1))

        ### input B (real images)
        if opt.isTrain and not opt.no_front:        
            self.B_paths = sorted(self.find_root_path(self.root,key='RENDER_NORMAL/*/*.jpg',span=1))
            assert self.__check_A_B()
        elif opt.isTrain:
            self.B_paths = [a.replace("RENDER",'RENDER_NORMAL') for a in self.A_paths]
            for _index in range(len(self.B_paths)):
                B_key = self.B_paths[_index].split("/")[-1]
                pitch = B_key.split("_")[0]
                back_pitch = (int(pitch)+180)%360
                back_key = "{}_0_00.jpg".format(back_pitch)
                self.B_paths[_index] = self.B_paths[_index].replace(B_key,back_key)
            self.__check_B_exists()
            

        
        ### mask maps
        if not opt.no_instance:
            self.inst_paths  = [b.replace("RENDER_NORMAL",'MASK').replace(".jpg",'.png') for b in self.B_paths]


        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]             
        A = Image.open(A_path)  
        params = get_params(self.opt, A.size)

        if self.opt.label_nc == 0:

            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain and not self.opt.no_front:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)
        elif self.opt.isTrain:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            mask_path = A_path.replace("RENDER","MASK").replace('.jpg','.png')
            mask = Image.open(mask_path).convert('RGB')
            mask = np.asarray(mask,dtype=np.uint8)
            B = np.asarray(B,dtype=np.float).copy()
            B = B[:,::-1,:]
            B /= 255
            B = B*2 -1
            B[...,2] = -B[...,2]
            B[...,0] = -B[...,0]

            B = (B+1)/2
            B*=255
            B = np.asarray(B,np.uint8)
            B[mask == 0]= 0
            B = Image.fromarray(B)
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)
        ### if using instance maps  
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst = np.asarray(inst,np.bool)
            inst_tensor = torch.from_numpy(inst)
            
            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat)) 

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'RenderPeopleDataset'