import sys
sys.path.append('.')
import torch
import numpy as np
import mcubes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

def create_sphere_mesh(resolution=64, radius=1.0):
    # Create a grid of points
    x, y, z = np.mgrid[-1.5:1.5:resolution*1j, -1.5:1.5:resolution*1j, -1.5:1.5:resolution*1j]
    
    # Implicit function for sphere
    sphere_function = x**2 + y**2 + z**2 - radius**2
    
    # Extract vertices and faces using marching cubes
    vertices, triangles = mcubes.marching_cubes(sphere_function, 0)
    
    # Normalize vertices
    vertices = vertices / resolution * 3 - 1.5
    
    return torch.tensor(vertices.astype(np.float32)), torch.tensor(triangles.astype(np.int64))

def create_torus_mesh(resolution=64, R=1.2, r=0.1):
    # Create a grid of points
    x, y, z = np.mgrid[-2:2:resolution*1j, -2:2:resolution*1j, -2:2:resolution*1j]
    
    # Implicit function for torus
    torus_function = (np.sqrt(x**2 + z**2) - R)**2 + y**2 - r**2
    
    # Extract vertices and faces using marching cubes
    vertices, triangles = mcubes.marching_cubes(torus_function, 0)
    
    # Normalize vertices
    vertices = vertices / resolution * 4 - 2
    
    return torch.tensor(vertices.astype(np.float32)), torch.tensor(triangles.astype(np.int64))

from PIL import Image

def jpg_to_tensor(image_path):
    """
    Converts a JPG image into a PyTorch tensor of colors.

    Args:
        image_path (str): Path to the JPG image.

    Returns:
        torch.Tensor: A tensor representing the image colors in shape (C, H, W),
                      where C is the number of channels (3 for RGB),
                      H is the height, and W is the width.
    """
    try:
        # Open the image file
        img = Image.open(image_path).convert("RGB")

        # Convert the image to a PyTorch tensor
        img_tensor = torch.from_numpy(np.array(img)).float()

        # Rearrange dimensions to (C, H, W) from (H, W, C)
        img_tensor = img_tensor.permute(2, 0, 1) / 255.0

        return img_tensor

    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def create_texture_map_by_image(vertices, image_path):
    """
    Creates a texture map for a 3D model based on an image.

    Args:
        vertices (torch.Tensor): Tensor of shape (N, 3) representing 3D vertices.
        image_path (str): Path to the texture image.
        type (str): The type of object being textured (e.g., "planet").

    Returns:
        torch.Tensor: A tensor containing the texture colors mapped to vertices.
    """
    try:
        # Load the image as a tensor
        texture_image = jpg_to_tensor(image_path)
        if texture_image is None:
            raise ValueError("Failed to load the texture image.")

        # Normalize vertex positions to the range [0, 1] for texture mapping
        min_vals = vertices.min(dim=0).values
        max_vals = vertices.max(dim=0).values
        normalized_vertices = (vertices - min_vals) / (max_vals - min_vals)

        # Map normalized vertices to texture coordinates (u, v)
        u = (normalized_vertices[:, 0] * (texture_image.shape[2] - 1)).long()
        v = (normalized_vertices[:, 1] * (texture_image.shape[1] - 1)).long()

        # Ensure indices are within bounds
        u = torch.clamp(u, 0, texture_image.shape[2] - 1)
        v = torch.clamp(v, 0, texture_image.shape[1] - 1)

        # Extract colors for each vertex
        texture_colors = texture_image[:, v, u].permute(1, 0)

        return texture_colors

    except Exception as e:
        print(f"Error creating texture map: {e}")
        return None

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create planet mesh
    planet_verts, planet_faces = create_sphere_mesh()
    planet_colors = create_texture_map_by_image(planet_verts, "data/Textures4Fun/sat0fds1.jpg")
    
    # Create ring mesh
    ring_verts, ring_faces = create_torus_mesh()
    ring_colors = create_texture_map_by_image(ring_verts, "data/Textures4Fun/rotated_ring.jpg")

    # Combine meshes
    verts = torch.cat([planet_verts, ring_verts])
    faces = torch.cat([planet_faces, ring_faces + len(planet_verts)])
    colors = torch.cat([planet_colors, ring_colors])
    
    # Create a Meshes object
    textures = TexturesVertex(verts_features=colors[None])
    meshes = Meshes(
        verts=[verts],
        faces=[faces],
        textures=textures
    ).to(device)

    from starter.panaroma_render import render_mesh_panaroma
    render_mesh_panaroma(meshes, output_path='play/6-fun/planet_ring.gif', top=30, bottom=-30)
    return 0

if __name__ == "__main__":
    main()