import sys
sys.path.append('..')
sys.path.append('.')

import ipdb
import imageio

import torch
import numpy as np

import mcubes
from assignment1.starter.utils import get_device, get_points_renderer
from assignment1.starter.panaroma_render import render_mesh_panaroma

from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.renderer.cameras import look_at_view_transform
import pytorch3d

import pytorch3d.ops.cubify as cubify

device = get_device()
def to_numpy(tensor)->np.ndarray:
    return tensor.detach().cpu().numpy()

def render_mesh_util(vertices=None, triangles=None, output='', mesh=None, dist=32):
    '''
    Given vertices and triangles, render the 3D mesh.
    '''
    # Create a Meshes object
    if mesh is None:
        assert vertices is not None and triangles is not None, "Either mesh or vertices and triangles should be provided."
        vertices = torch.tensor(vertices, device=device, dtype=torch.float32)
        faces = torch.tensor(triangles, device=device, dtype=torch.int32)
        mesh = Meshes(verts=[vertices], faces=[faces], textures=TexturesVertex(verts_features=torch.ones_like(vertices)[None]))
    # ipdb.set_trace()
    render_mesh_panaroma(mesh, dist=dist, output_path=output)

def render_volume(voxel_grid, output, dist=3):
    def cubify_voxel(voxel_grid):
        '''
        [B, 1, D, H, W]
        '''
        if len(voxel_grid.shape) == 5:
            voxel_grid = voxel_grid[:, 0] # [B, D, H, W]
        B = voxel_grid.shape[0]
        mesh = pytorch3d.ops.cubify(voxel_grid, 0.5)
        mesh.textures = TexturesVertex(verts_features=torch.ones_like(mesh.verts_packed()).unsqueeze(0).repeat(B, 1, 1)) # [B, V, C]
        return mesh
         
    def voxel2mesh(voxel_grid):
        '''
        Given a voxel grid, return a mesh using marching cubes algorithm.
        '''
        # shape_x, shape_y, shape_z = voxel_grid.shape
        # x, y, z = np.mgrid[:shape_x, :shape_y, :shape_z]

        # Create the 0-isosurface
        voxel_grid = to_numpy(voxel_grid[0])
        print(voxel_grid.dtype)
        vertices, triangles = mcubes.marching_cubes(voxel_grid, 0)
        vertices = vertices.astype(np.float32)
        vertices -= vertices.mean(axis=0) # Move to center
        triangles = triangles.astype(np.int64)
        return vertices, triangles

    # NOTE: using mcube
    # return render_mesh_util(*voxel2mesh(voxel_grid), output=output, dist=32)
    # NOTE: using cubify
    return render_mesh_util(mesh=cubify_voxel(voxel_grid), output=output, dist=dist)

def render_mesh(mesh, output):
    '''
    - input
        - mesh: pytorch3d.structures.meshes.Meshes
    '''
    # ipdb.set_trace()
    vertices = mesh.verts_packed()
    faces = mesh.faces_packed()
    return render_mesh_util(to_numpy(vertices), to_numpy(faces), output=output, dist=1)

def render_pointcloud(points: torch.tensor, output, dist=0.8, num_views=12):
    '''
    - input
        - points: [1, P, 3]
    '''
    # print("x, y, z mean")
    # print(points[:, 0].mean())
    # print(points[:, 1].mean())
    # print(points[:, 2].mean())

    # print(points.shape)
    points = points.repeat(num_views, 1, 1)

    # print(points.shape)
    features = torch.ones_like(points)

    point_cloud = pytorch3d.structures.Pointclouds(points=points, features=features)
    
    points_renderer = get_points_renderer(
        image_size=256, 
        radius=0.01,
    )
    azims = torch.linspace(0, 360, num_views)
    R, T = look_at_view_transform(dist=dist, elev=0, azim=azims)
    pc_cameras = pytorch3d.renderer.PerspectiveCameras(
        R=R, 
        T=T, 
        device=device
    )
    rend = points_renderer(point_cloud, cameras=pc_cameras)
    # #print(rend.shape) # [B, S, S, 3]
    image_list = to_numpy(rend[..., :3])
    images = (image_list*255).astype(np.uint8)
    # #print(image_list.shape)
    duration = 0.00005  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(output, images, duration=duration, loop=0)