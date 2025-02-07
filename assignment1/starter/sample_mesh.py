import sys
sys.path.append('.')

import numpy as np
import torch
import pytorch3d
from utils import get_device, load_cow_mesh, get_points_renderer
from starter.panaroma_render import generate_spiral_points
import imageio

device = get_device()

def set_seed(seed=23):
    np.random.seed(seed)
    torch.manual_seed(seed)

def calculate_face_areas(faces, vertices):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    e0 = v1 - v0
    e1 = v2 - v0
    cross = torch.cross(e0, e1)
    return 0.5 * torch.linalg.norm(cross, axis=1)

def sample_pc_from_mesh(faces, vertices, num_samples):
    '''
    Input: 
    - faces: torch.tensor - (Nf, 3)
    - vertices: torch.tensor - (Nv, 3)
    - num_samples: number of points to sample from the mesh
    Output:
    - pc: a point cloud

    Procedure:
    1. Sample a face with probability proportional to the area of the face
    2. Sample a random barycentric coordinate uniformly
    3. Compute the corresponding point using baricentric coordinates on the selected face.
    '''
    areas = calculate_face_areas(faces, vertices)
    #print(areas.shape)
    probabilities = areas / areas.sum()

    def sample_barycentric_coordinates():
        u = torch.rand(1).item()
        v = torch.rand(1).item()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        w = 1 - u - v
        return u, v, w

    pc = torch.zeros((num_samples, 3))
    for i in range(num_samples):
        face_idx = torch.multinomial(probabilities, 1).item()
        face = faces[face_idx]
        u, v, w = sample_barycentric_coordinates()
        point = u * vertices[face[0]] + v * vertices[face[1]] + w * vertices[face[2]]
        pc[i] = point

    return pc

set_seed()
vertices, faces = load_cow_mesh()
#print("Number of faces: ", faces.shape)
num_sample_list = [10, 100, 1000, 10000]

num_cameras = 100
elevations, azimuths = generate_spiral_points(num_cameras)

for num_samples in num_sample_list:
    points = sample_pc_from_mesh(vertices=vertices, faces=faces, num_samples=num_samples)
    color = (points - points.min()) / (points.max() - points.min())
    

    cow_pc = pytorch3d.structures.Pointclouds(points=[points], features=[color]).to(device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=elevations, azim=azimuths, device=device)
    #print("R shape: ", R.shape)
    #print("T shape: ", T.shape)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

    renderer = get_points_renderer(image_size=256, device=device)

    rend = renderer(cow_pc.extend(num_cameras), cameras=cameras)
    #print("rend shape: ", rend.shape)
    rend = rend[..., :3].cpu().numpy()

    images = ((rend * 255).astype(np.uint8))
    duration = 0.05
    imageio.mimsave(f"play/7-sample-mesh/points_{num_samples}.gif", images, duration=duration, loop=0)