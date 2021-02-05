''' 
Generate 2d maps representing different attributes(colors, depth, pncc, etc)
: render attributes to image space.
'''
import os, sys
import numpy as np
import scipy.io as sio
from skimage import io
from time import time
import matplotlib.pyplot as plt

sys.path.append('..')
import engineer.render.face3d as face3d
from engineer.render.face3d import mesh

# ------------------------------ load mesh data
C = sio.loadmat('Data/example1.mat')
vertices = C['vertices']; colors = C['colors']; triangles = C['triangles']

colors = colors/np.max(colors)

# ------------------------------ modify vertices(transformation. change position of obj)
# scale. target size=200 for example
s = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1]))
# rotate 30 degree for example
R = mesh.transform.angle2matrix([0, 30, 0]) 
# no translation. center of obj:[0,0]
t = [0, 0, 0]

transformed_vertices = mesh.transform.similarity_transform(vertices, s, R, t)

# ------------------------------ render settings(to 2d image)
# set h, w of rendering
h = w = 256
# change to image coords for rendering
image_vertices = mesh.transform.to_image(transformed_vertices, h, w)

## --- start
save_folder = 'results/image_map'
if not os.path.exists(save_folder):
    os.makedirs(save_folder,exist_ok= True)
## 0. color map
attribute = colors


color_image = mesh.render.render_colors(image_vertices, triangles, attribute, h, w, c=3)
io.imsave('{}/color.jpg'.format(save_folder), np.squeeze(color_image))
## 1. depth map
z = image_vertices[:,2:]
z = z - np.min(z)
z = z/np.max(z)
attribute = z
depth_image = mesh.render.render_colors(image_vertices, triangles, attribute, h, w, c=1)
io.imsave('{}/depth.jpg'.format(save_folder), np.squeeze(depth_image))