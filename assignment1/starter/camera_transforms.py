"""
Usage:
    python -m starter.camera_transforms --image_size 512
"""
import sys
sys.path.append('.')
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer
from pytorch3d.renderer import look_at_view_transform


def render_textured_cow(
    cow_path="data/cow.obj",
    image_size=256,
    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],
    device=None,
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)
    R_relative = torch.tensor(R_relative).float()
    T_relative = torch.tensor(T_relative).float()
    R_0 = [[1.0, 0, 0], [0, 1, 0], [0, 0, 1]]
    T_0 = [0.0, 0, 3]
    R = R_relative @ torch.tensor(R_0)
    T = R_relative @ torch.tensor(T_0) + T_relative
    renderer = get_mesh_renderer(image_size=256)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.unsqueeze(0), T=T.unsqueeze(0), device=device,
    )

    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    rend = renderer(meshes, cameras=cameras, lights=lights)
    return rend[0, ..., :3].cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--output_path", type=str, default="play/textured_cow")
    args = parser.parse_args()
    R = []
    T = []
    # Default Setting
    R.append([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    T.append([0, 0, 0])
    # Case 1
    R.append([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    T.append([0, 0, 0])
    # Case 2
    R.append([[1, 0 , 0], [0, 1, 0], [0, 0, 1]])
    T.append([0, 0, 3])
    # Case 3
    R.append([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    T.append([0.5, -0.5, 0])
    # Case 4
    R.append([[0, 0, 1], [0, 1.0, 0], [-1.0, 0, 0]]) 
    T.append([-3, 0, 3.0])

    for i in range(len(R)):
        image = render_textured_cow(cow_path=args.cow_path, image_size=args.image_size, \
                        R_relative=R[i], T_relative=T[i])
        plt.imsave(f"play/x_{i}.jpg", image)
    
