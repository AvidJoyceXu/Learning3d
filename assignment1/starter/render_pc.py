import sys
sys.path.append('.')

from starter.utils import unproject_depth_image
from starter.render_generic import load_rgbd_data
from starter.utils import get_device
import torch, numpy as np
import pytorch3d
device = get_device()
print(device)

data = load_rgbd_data()
print(data.keys())
rgbs = [data['rgb1'], data['rgb2']]
rgbs = [torch.from_numpy(rgb).to(device) for rgb in rgbs]
masks = [data['mask1'], data['mask2']]
masks = [torch.from_numpy(mask).to(device) for mask in masks]
depth = [data['depth1'], data['depth2']]
depth = [torch.from_numpy(depth).to(device) for depth in depth]
cameras = [data['cameras1'], data['cameras2']]
cameras = [camera.to(device) for camera in cameras]
# print(cameras[0]) # PerspectiveCameras()
# print(rgbs[0].shape) # (800, 800, 3)
# print(masks[0].shape) # (800, 800)
# print(depth[0].shape) # (800, 800)

num_views = 12
def wrapper_unproject_depth_image(depth, camera, mask, image, repeat=1):
    points, rgb = unproject_depth_image(depth=depth, camera=camera, mask=mask, image=image)
    points = points.unsqueeze(0).repeat(repeat, 1, 1)
    rgb = rgb[...,:3].unsqueeze(0).repeat(repeat, 1, 1)
    return points, rgb

from pytorch3d.renderer import look_at_view_transform
from starter.utils import get_points_renderer

def render_pc_from_points(points, features):
    print(points.shape, features.shape) # [B, N, 3], [B, N, 3]
    point_cloud = pytorch3d.structures.Pointclouds(points=points, features=features)
    points_renderer = get_points_renderer(
        image_size=256, 
        radius=0.01,
    )
    azims = torch.linspace(0, 360, num_views)
    R, T = look_at_view_transform(6.0, 0, azim=azims)
    pc_cameras = pytorch3d.renderer.PerspectiveCameras(
        R=R, 
        T=T, 
        device=device
    )
    rend = points_renderer(point_cloud, cameras=pc_cameras)
    # print(rend.shape) # [B, S, S, 3]
    image_list = rend[..., :3].cpu().numpy()
    images = (image_list*255).astype(np.uint8)
    # print(image_list.shape)
    return np.flip(images, axis=1)

import imageio
points_list = []
rgb_list = []
for i in range(len(cameras)):
    points, rgb = wrapper_unproject_depth_image(depth=depth[i], camera=cameras[i], mask=masks[i], image=rgbs[i], 
                                                repeat=num_views) 
    points_list.append(points)
    rgb_list.append(rgb)
    images = render_pc_from_points(points, rgb)
    duration = 0.00005  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(f'play/5-rendering-pc/plants-view-{i}.gif', images, duration=duration, loop=0)

points_union = torch.cat(points_list, dim=1)
rgb_union = torch.cat(rgb_list, dim=1)
images = render_pc_from_points(points_union, rgb_union)
duration = 0.00005  # Convert FPS (frames per second) to duration (ms per frame)
imageio.mimsave(f'play/5-rendering-pc/plants-union.gif', images, duration=duration, loop=0)


