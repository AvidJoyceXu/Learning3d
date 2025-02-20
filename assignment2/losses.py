import torch
import torch.nn as nn
import ipdb
from pytorch3d.loss import mesh_laplacian_smoothing
# define losses
def voxel_loss(voxel_src,voxel_tgt):
	'''
	voxel_src: b x h x w x d
	voxel_tgt: b x h x w x d
	implement some loss for binary voxel grids
	'''
	# loss = nn.CrossEntropyLoss()
	# return loss(voxel_src,voxel_tgt)

	voxel_src = torch.clamp(voxel_src, 0, 1)
	criterion = nn.BCELoss()
	return criterion(voxel_src,voxel_tgt)

def chamfer_loss(point_cloud_src, point_cloud_tgt):
	'''
	- input: 
		- point_cloud_src, point_cloud_src: b x n_points x 3  
	- output: [1]
		- return chamfer loss between two point clouds, normalized by the number of points
	'''
	dist_matrix = torch.cdist(point_cloud_src, point_cloud_tgt)
	loss_chamfer = torch.min(dist_matrix, dim=1)[0].mean() + torch.min(dist_matrix, dim=2)[0].mean()
	# implement chamfer loss from scratch
	loss_chamfer = loss_chamfer / 2
	return loss_chamfer

def smoothness_loss(mesh_src):
	'''
	- input: 
		- mesh_src: pytorch3d.structures.meshes.Meshes
	- output: laplacian smoothening loss
	'''
	# Compute Laplacian
	return mesh_laplacian_smoothing(mesh_src)

def compute_laplacian(mesh):
	'''
	- input: 
		- mesh: pytorch3d.structures.meshes.Meshes
	- output: 
		- Laplacian **loss** for the mesh: [N, 3]
	'''
	verts = mesh.verts_packed() # [N, 3]
	n_vertices, _ = verts.shape
	L = torch.zeros((n_vertices, 3), device=mesh.device)
	faces = mesh.faces_packed()
	for i in range(n_vertices):
		neighbors = get_neighbors(faces, i) 
		L[i, :] = verts[i, :] - torch.mean(verts[neighbors, :], dim=0)
	return L

def get_neighbors(faces, vertex_index):
	# Convert vertex_index to tensor if it's not already
    vertex_index = torch.tensor(vertex_index, device=faces.device)
    
    # Find faces that contain the vertex
    face_mask = (faces == vertex_index).any(dim=1)
    relevant_faces = faces[face_mask]
    
    # Get all adjacent vertices and remove the vertex itself
    neighbors = relevant_faces.reshape(-1)
    neighbors = neighbors[neighbors != vertex_index]
    
    # Get unique neighbors using torch.unique
    unique_neighbors = torch.unique(neighbors)
    
    return unique_neighbors