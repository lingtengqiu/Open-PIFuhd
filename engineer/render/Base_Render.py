'''
@author: Lingteng Qiu
@data: 2021-02-04
Base class for render object
'''
import numpy as np
import torch
import trimesh
from engineer.utils.metrics import euler_to_rot_mat
class _Base_Render(object):
    def __init__(self,width:int,height:int,render_lib='face3d'):
        ''' Base render class 
        Parameters
            width: width of render image
            height: height of render image 
            e.g. uv map
        '''
        super(_Base_Render,self).__init__()
        self.width = width
        self.height = height
        self.render_lib = render_lib
        self.__name = 'Base_Render'
        self.verts = None
        self.faces = None
        self.verts_normals = None
        self.faces_normals = None
        self.world_to_view_point =np.eye(4)
        self.__camera = None


        self.input_para = dict(
            width = width,
            height = height,
            render_lib = render_lib,           
        )
    

    def world2uv(self):
        world_to_view_point = torch.from_numpy(self.world_to_view_point).float().unsqueeze(0)
        verts = torch.from_numpy(self.verts).unsqueeze(0).float()

        world_mat = world_to_view_point[:,:3,:3]
        world_trans = world_to_view_point[:,:3,3:4]
        
        verts = torch.baddbmm(world_trans,world_mat,verts.transpose(1,2)).transpose(1,2)

        mat = self.camera[:,:3,:3]
        trans = self.camera[:,:3,3:4]
        #[-1,1]
        verts = torch.baddbmm(trans,mat,verts.transpose(1,2)).transpose(1,2)

        verts = (verts+1)*self.width/2
        verts = verts.squeeze(0).numpy()

        if self.flip_normal == True:
            faces = self.faces[...,[0,2,1]]

        self.render_mesh = trimesh.Trimesh(verts,faces)
        


    def set_world_view_point(self,deg):
        '''setting from world to view homo rotation matrix
        deg: degree to rotation your map
        '''

        rz = deg / 180. * np.pi
        
        self.world_to_view_point[:3, :3] = euler_to_rot_mat(0, rz, 0)

    def set_mesh(self, verts, faces, faces_normals =None, verts_normals = None):
        '''set mesh infos
        Parameters:
            verts float[N, 3]:  vertex of mesh
            faces int[N, 3]:  faces id of mesh
            faces_normals[N, 3]: normals of faces of mesh
            verts_normals[N, 3]: normals of verts of mesh  
        '''
        self.verts = verts
        self.faces = faces
        self.faces_normals =faces_normals
        self.verts_normals = verts_normals

    def set_attribute(self,attribute):
        '''attribute you want to set when you render
        '''
        raise NotImplementedError
    def draw(self):

        raise NotImplementedError
    

    @property
    def camera(self):
        return self.__camera
    @camera.setter
    def camera(self,calib:torch.Tensor):
        '''
        Parameters:
            calib: [B, 4, 4]
        '''
        self.__camera = calib
    
    


    def __repr__(self):
        __repr = "{}(Parameters: ".format(self.__name)
        for key in self.input_para.keys():
            __repr+="{}:{}, ".format(key,self.input_para[key])
        __repr=__repr[:-2]
        return __repr+')'

    @property
    def name(self):
        __repr = "{}(Parameters: ".format(self.__name)
        for key in self.input_para.keys():
            __repr+="{}:{}, ".format(key,self.input_para[key])
        __repr=__repr[:-2]
        return __repr+')'
    
    @name.setter
    def name(self,v):
        self.__name = v
