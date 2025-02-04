'''
python starter/360d-render.py --mesh_path data/MyMeshes/cube.obj
'''


import sys
sys.path.append('.')
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.io import load_obj
import torch
from pytorch3d.renderer import FoVPerspectiveCameras
import pytorch3d
from pytorch3d.renderer import (
    RasterizationSettings,
    PointLights,
)
from starter.utils import get_mesh_renderer
from PIL import Image
import numpy as np
import imageio

# Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import argparse

def generate_spiral_points(n_points=100):
    """
    Generate points on a sphere in a continuous spiral pattern from top to bottom.
    
    Args:
        n_points: Number of points to generate
    
    Returns:
        Tuple of (elevations, azimuths) in degrees
    """

    # Generate points from top to bottom (90° to -90°)
    elevations = torch.linspace(90, -90, n_points)

    azimuth_step = 360.0 / (n_points / 4)  # base step
    azimuths = (torch.arange(n_points) * azimuth_step) % 360
    
    return elevations, azimuths
    
def main(meshes):
    num_cameras = 100
    elevations, azimuths = generate_spiral_points(num_cameras)
    # Get camera positions using look_at_view_transform
    R, T = look_at_view_transform(
        dist=3,
        elev=elevations,
        azim=azimuths,
        device=device
    )

    # Create a batch of cameras
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Move the mesh to the device
    meshes = meshes.to(device)

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    renderer = get_mesh_renderer(lights=lights, image_size=256, device=device)

    # Render the images
    images = renderer(meshes.extend(num_cameras), cameras=cameras, lights=lights)

    images = images.cpu().numpy()[:, :, :, :3] # [1, H, W, rgb]
    images = [(image*255).astype(np.uint8) for image in images]

    duration = 0.00005  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave('play/loop-circle-cameras-new.gif', images, duration=duration, loop=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render a 3D mesh from different camera angles.')
    parser.add_argument('--mesh_path', type=str, required=True, help='Path to the 3D mesh file')
    args = parser.parse_args()

    vertices, faces, _ = load_obj(args.mesh_path)
    vertices = vertices.unsqueeze(0)
    faces = faces.verts_idx.unsqueeze(0)

    texture_rgb = torch.ones_like(vertices) # N X 3
    texture_rgb = texture_rgb * torch.tensor([0.7, 0.7, 1])
    textures = pytorch3d.renderer.TexturesVertex(texture_rgb) #

    meshes = pytorch3d.structures.Meshes(
        verts=vertices, # batched tensor or a list of tensors
        faces=faces,
        textures=textures,
    )
    main(meshes)
