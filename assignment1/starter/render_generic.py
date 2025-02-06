"""
Sample code to render various representations.

Usage:
    python -m starter.render_generic --render point_cloud  # 5.1
    python -m starter.render_generic --render parametric  --num_samples 100  # 5.2
    python -m starter.render_generic --render implicit  # 5.3
"""
import sys
sys.path.append('.')
import argparse
import pickle

import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer, get_points_renderer
from starter.panaroma_render import generate_spiral_points

def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def render_bridge(
    point_cloud_path="data/bridge_pointcloud.npz",
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    point_cloud = np.load(point_cloud_path)
    verts = torch.Tensor(point_cloud["verts"][::50]).to(device).unsqueeze(0)
    rgb = torch.Tensor(point_cloud["rgb"][::50]).to(device).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    return rend


def render_sphere(image_size=256, num_samples=200, device=None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = torch.sin(Theta) * torch.cos(Phi)
    y = torch.cos(Theta)
    z = torch.sin(Theta) * torch.sin(Phi)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(sphere_point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()

def render_torus(R=1, r=0.25, image_size=256, num_samples=200, device=None, num_cameras=100):
    """
    Renders a torus using parametric sampling. Samples num_samples ** 2 points.
    Parametric curve: https://en.wikipedia.org/wiki/Torus
    """
    
    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = (R + r * torch.cos(Theta)) * torch.cos(Phi)
    y = (R + r * torch.cos(Theta)) * torch.sin(Phi)
    z = r * torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1).to(device) \
            .unsqueeze(0)\
            .repeat(num_cameras, 1, 1)
    print("points shape: ", points.shape)
    color = (points - points.min()) / (points.max() - points.min()).to(device) \
            .unsqueeze(0)\
            .repeat(num_cameras, 1, 1)
    print("color shape: ", color.shape)
    
    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=points, features=color,
    ).to(device)

    elevations, azimuths = generate_spiral_points(num_cameras)
    print("elev: ", elevations.shape)
    print("azim: ", azimuths.shape)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=elevations, azim=azimuths,
                                                     at = torch.tensor([[0, 0, 0]]).to(device), 
                                                     # The parametric curve is defined on the original point
                                                     device=device)
    print("R: ", R.shape)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R = R,
        T = T, 
        device=device
    )
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(torus_point_cloud, cameras=cameras)
    print("rend shape: ", rend.shape)
    return rend[..., :3].cpu().numpy()

def render_klein_bottle(r=1, image_size=256, num_samples=200, num_cameras=100, device=None):
    '''
    Render a Klein bottle using the following parametric equation:
    - x(u, v) = (r + cos(u/2) * sin(v) - sin(u/2) * sin(2v)) * cos(u)
    - y(u, v) = (r + cos(u/2) * sin(v) - sin(u/2) * sin(2v)) * sin(u)
    - z(u, v) = sin(u/2) * sin(v) + cos(u/2) * sin(2v)
    '''
    if device is None:
        device = get_device()
    u = torch.linspace(0, 2 * np.pi, num_samples)
    v = torch.linspace(0, 2 * np.pi, num_samples)
    U, V = torch.meshgrid(u, v)
    x = (r + torch.cos(U / 2) * torch.sin(V) - torch.sin(U / 2) * torch.sin(2 * V)) * torch.cos(U)
    y = (r + torch.cos(U / 2) * torch.sin(V) - torch.sin(U / 2) * torch.sin(2 * V)) * torch.sin(U)
    z = torch.sin(U / 2) * torch.sin(V) + torch.cos(U / 2) * torch.sin(2 * V)
    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1).to(device) \
            .unsqueeze(0)\
            .repeat(num_cameras, 1, 1)
    print("points shape: ", points.shape)
    color = (points - points.min()) / (points.max() - points.min()).to(device) \
            .unsqueeze(0)\
            .repeat(num_cameras, 1, 1)
    print("color shape: ", color.shape)
    
    point_cloud = pytorch3d.structures.Pointclouds(
        points=points, features=color,
    ).to(device)

    elevations, azimuths = generate_spiral_points(num_cameras)
    print("elev: ", elevations.shape)
    print("azim: ", azimuths.shape)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=10, elev=elevations, azim=azimuths,
                                                     at = torch.tensor([[0, 0, 0]]).to(device), 
                                                     # The parametric curve is defined on the original point
                                                     device=device)
    print("R: ", R.shape)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R = R,
        T = T, 
        device=device
    )
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    print("rend shape: ", rend.shape)
    return rend[..., :3].cpu().numpy()

def render_sphere_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    # X.shape: [voxel_size, voxel_size, voxel_size]
    voxels = X ** 2 + Y ** 2 + Z ** 2 - 1
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)

def render_torus_mesh(R=1, r=0.25, image_size=256, voxel_size=64, device=None, num_cameras=200):
    if device is None:
        device = get_device()
    min_value = -1.3
    max_value = 1.3
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    # X.shape: [voxel_size, voxel_size, voxel_size]

    voxels = (torch.sqrt(X**2 + Y**2) - R)**2 + Z**2 - r**2
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))

    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.

    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)

    elev, azim = generate_spiral_points(num_cameras)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=elev, azim=azim, device=device)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh.extend(num_cameras), cameras=cameras, lights=lights)
    return rend[..., :3].detach().cpu().numpy().clip(0, 1)

def render_heart_mesh(image_size=256, voxel_size=64, num_cameras=200, device=None):
    '''
    Render a **heart** using the following implicit function:
    - (x^2 + 9/4 y^2 + z^2 - 1)^3 - x^2 z^3 - 9/80 y^2 z^3 = 0
    And using mcube to render a mesh.
    '''
    if device is None:
        device = get_device()
    min_value = -1.3
    max_value = 1.3
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = (X**2 + 9/4 * Y**2 + Z**2 - 1)**3 - X**2 * Z**3 - 9/80 * Y**2 * Z**3
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    
    textures = torch.ones_like(vertices) * torch.tensor([1.0, 0.75, 0.8])
    textures = pytorch3d.renderer.TexturesVertex(textures.unsqueeze(0))
    
    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(device)

    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)

    renderer = get_mesh_renderer(image_size=image_size, device=device)

    elev, azim = generate_spiral_points(num_cameras)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=elev, azim=azim, device=device)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh.extend(num_cameras), cameras=cameras, lights=lights)
    return rend[..., :3].detach().cpu().numpy().clip(0, 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="point_cloud",
        # choices=["point_cloud", "parametric", "implicit", "torus", "torus_mesh", ],
    )
    parser.add_argument("--output_path", type=str, default="play/5-rendering-pc/bridge.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    if args.render == "point_cloud":
        image = render_bridge(image_size=args.image_size)
    elif args.render == "parametric":
        image = render_sphere(image_size=args.image_size, num_samples=args.num_samples)
    elif args.render == "torus":
        image = render_torus(image_size=args.image_size, num_samples=args.num_samples)
    elif args.render == "implicit":
        image = render_sphere_mesh(image_size=args.image_size)
    elif args.render == "bottle":
        image = render_klein_bottle(image_size=args.image_size)
    elif args.render == "torus_mesh":
        image = render_torus_mesh(image_size=args.image_size)
    elif args.render == "heart_mesh":
        image = render_heart_mesh(image_size=args.image_size)
    else:
        raise Exception("Did not understand {}".format(args.render))
    if image.ndim == 4:
        import imageio
        duration = 1
        imageio.mimsave(args.output_path, (image * 255).astype(np.uint8), duration=duration, loop=0)
    else:  
        plt.imsave(args.output_path, image)