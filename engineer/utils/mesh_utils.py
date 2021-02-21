import numpy as np
import os 
from PIL import Image
from engineer.utils.sdf import *
from skimage import measure
import torch

def reconstruction(net, calib_tensor,
                   resolution, b_min, b_max,
                   use_octree=False, num_samples=50000, transform=None):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid(resolution, resolution, resolution,
                              b_min, b_max, transform=transform)

    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        points = np.repeat(points, net.num_views, axis=0)
        samples = torch.from_numpy(points).cuda().float()
        net.query(samples, calib_tensor)
        pred = net.get_preds()[0][0]
        return pred.detach().cpu().numpy()
    # Then we evaluate the grid
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples)
    # Finally we do marching cubes
    try:
        verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, 0.5)
        # transform verts into world coordinate system
        verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
        verts = verts.T
        return verts, faces, normals, values
    except:
        print('error cannot marching cubes')
        return -1

def gen_mesh(cfg, net, data, save_path, use_octree=True):
    '''generate mesh by marching cube

    Parameters:
        cfg (CfgNode): configs. Details can be found in
            configs/PIFu_Render_People_HG.py
        net: network model e.g. PIFu
        data: input data, it contains four key variables, 'img','calib', 'b_min', 'b_max'
        save_path: where you save your mesh and single-view image
        use_octree: default, True
    
    return 
        None
    '''
    try:
        #distributed model
        net = net.module
    except:
        pass
    image_tensor = data['img'].cuda()
    calib_tensor = data['calib'].cuda()
    origin_calib_tensor = data['origin_calib']


    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor[None,...]
    net.extract_features(image_tensor)
    b_min = data['b_min']
    b_max = data['b_max']
    save_img_path = os.path.join(save_path,"obj.jpg")
    save_img_list = []
    for v in range(image_tensor.shape[0]):
        save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, :] * 255.0
        save_img_list.append(save_img)
    save_img = np.concatenate(save_img_list, axis=1)
    save_img = save_img[...,:3]

    Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)

    verts, faces, _, _ = reconstruction(net, calib_tensor, cfg.resolution, b_min, b_max, use_octree=use_octree)
    
    verts,flip_face = transfer_uv_to_world(verts,origin_calib_tensor)
    save_obj_mesh(os.path.join(save_path,"mesh.obj"), verts, faces,flip_face)


def transfer_uv_to_world(verts,origin_calib,img_size=512,z_depth=200):

    
    if origin_calib == None:
        return verts,False
    verts[...,2] = verts[...,2]*z_depth/(img_size//2)
    mat = origin_calib.detach().numpy()
    inv_mat = np.linalg.inv(mat)
    homo_verts = np.concatenate([verts,np.ones((verts.shape[0],1))],axis=1)
    ori_verts = np.matmul(inv_mat,homo_verts.T).T
    return ori_verts[...,:3],True

def save_obj_mesh(mesh_path, verts, faces,flip=False):
    '''save mesh, xxx.obj
    
    Parameters:
        mesh_path: save to where
        verts: the vertices of mesh [N, 3]
        faces: face_id [N, 3]->[Int] 
    return None
    '''
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        if flip:           
            file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
        else:
            file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    '''save mesh with color, xxx.obj
    
    Parameters:
        mesh_path: save to where
        verts: the vertices of mesh [N, 3]
        faces: face_id [N, 3]->[Int] 
        colors: face color: [N, 3] rgb

    return None
    '''
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    '''save mesh with uv map, xxx.obj
    
    Parameters:
        mesh_path: save to where
        verts: the vertices of mesh [N, 3]
        faces: face_id [N, 3]->[Int] 
        uvs: face color: [N, 3] rgb

    return None
    '''
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        vt = uvs[idx]
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[2], f_plus[2],
                                              f_plus[1], f_plus[1]))
    file.close()