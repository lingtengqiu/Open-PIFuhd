'''
compute metrics 
'''
import torch
import trimesh
import numpy as np
import math
from PIL import Image


def euler_to_rot_mat(r_x, r_y, r_z):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(r_x), -math.sin(r_x)],
                    [0, math.sin(r_x), math.cos(r_x)]
                    ])

    R_y = np.array([[math.cos(r_y), 0, math.sin(r_y)],
                    [0, 1, 0],
                    [-math.sin(r_y), 0, math.cos(r_y)]
                    ])

    R_z = np.array([[math.cos(r_z), -math.sin(r_z), 0],
                    [math.sin(r_z), math.cos(r_z), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def compute_acc(pred, gt, thresh=0.5):
    '''
    Parameters:
        pred: points you net output [B, 1, N]
        gt: label of points [B, 1, N]
    return:
        IOU, precision, and recall
    '''
    with torch.no_grad():
        vol_pred = pred > thresh
        vol_gt = gt > thresh

        union = vol_pred | vol_gt
        inter = vol_pred & vol_gt

        true_pos = inter.sum().float()

        union = union.sum().float()
        if union == 0:
            union = 1
        vol_pred = vol_pred.sum().float()
        if vol_pred == 0:
            vol_pred = 1
        vol_gt = vol_gt.sum().float()
        if vol_gt == 0:
            vol_gt = 1
        return true_pos / union, true_pos / vol_pred, true_pos / vol_gt




def computer_metrics(pred,target):
    from engineer.render.PIFuhd.gl.normal_render import NormalRender

    normal_render = NormalRender(width=512, height=512)
    pred = trimesh.load(pred)
    target = trimesh.load(target)
    normal_loss = get_reproj_normal_error(normal_render,pred,target)
    chamfer_loss = computer_chamfer_distance(pred,target)
    p2s = computer_surface_dist(pred,target)

    return normal_loss,chamfer_loss,p2s

def computer_surface_dist(src,tgt, num_samples=10000):
    #P2S distance 
    src_surf_pts, _ = trimesh.sample.sample_surface(src, num_samples)

    _, src_tgt_dist, _ = trimesh.proximity.closest_point(tgt, src_surf_pts)

    src_tgt_dist[np.isnan(src_tgt_dist)] = 0

    src_tgt_dist = src_tgt_dist.mean()

    return src_tgt_dist

def computer_chamfer_distance(src,tgt,num_samples = 10000):

    src_surf_pts, _ = trimesh.sample.sample_surface(src, num_samples)
    tgt_surf_pts, _ = trimesh.sample.sample_surface(tgt, num_samples)
    _, src_tgt_dist, _ = trimesh.proximity.closest_point(tgt, src_surf_pts)
    _, tgt_src_dist, _ = trimesh.proximity.closest_point(src, tgt_surf_pts)
    src_tgt_dist[np.isnan(src_tgt_dist)] = 0
    tgt_src_dist[np.isnan(tgt_src_dist)] = 0
    src_tgt_dist = (src_tgt_dist).mean()
    tgt_src_dist = (tgt_src_dist).mean()
    chamfer_dist = (src_tgt_dist + tgt_src_dist) / 2
    return chamfer_dist




def get_reproj_normal_error(normal_render,pred,target,frontal=True, back=True, left=True, right=True):

    def _get_reproj_normal_error(render,pred,target,deg):
        tgt_normal = render_normal(render,target, deg)
        src_normal = render_normal(render,pred, deg)
        mask = tgt_normal[...,3]
        src_normal = src_normal[mask>0]
        tgt_normal = tgt_normal[mask>0]
        


        error = ((src_normal[..., :3] - tgt_normal[..., :3]) ** 2).mean() * 3


        return error, src_normal, tgt_normal


    side_cnt = 0
    total_error = 0
    if frontal:
        side_cnt += 1
        error, src_normal, tgt_normal = _get_reproj_normal_error(normal_render,pred,target,0)
        total_error += error
    if back:
        side_cnt += 1
        error, src_normal, tgt_normal = _get_reproj_normal_error(normal_render,pred,target,180)
        total_error += error
    if left:
        side_cnt += 1
        error, src_normal, tgt_normal = _get_reproj_normal_error(normal_render,pred,target,90)
        total_error += error
    if right:
        side_cnt += 1
        error, src_normal, tgt_normal = _get_reproj_normal_error(normal_render,pred,target,270)
        total_error += error
    return total_error / side_cnt


def render_normal(render,mesh, deg,scale_factor=1.3, offset=-120):
    view_mat = np.identity(4)
    view_mat[:3, :3] *= 2 / 256
    rz = deg / 180. * np.pi
    model_mat = np.identity(4)
    model_mat[:3, :3] = euler_to_rot_mat(0, rz, 0)
    model_mat[1, 3] = offset
    view_mat[2, 2] *= -1

    render.set_matrices(view_mat, model_mat)
    render.set_normal_mesh(scale_factor*mesh.vertices, mesh.faces, mesh.vertex_normals, mesh.faces)
    render.draw()
    normal_img = render.get_color()
    return normal_img
