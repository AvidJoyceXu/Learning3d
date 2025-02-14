import torch
import torch.nn as nn
# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	loss = nn.CrossEntropyLoss()
	# implement some loss for binary voxel grids
	return loss(voxel_src,voxel_tgt)

def chamfer_loss(point_cloud_src, point_cloud_tgt):
	'''
	- input: 
		- point_cloud_src, point_cloud_src: b x n_points x 3  
	- output: chamfer loss between two point clouds
	'''
	dist_matrix = torch.cdist(point_cloud_src, point_cloud_tgt)
	loss_chamfer = torch.min(dist_matrix, dim=1)[0].sum() + torch.min(dist_matrix, dim=2)[0].sum()
	# implement chamfer loss from scratch
	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
	return loss_laplacian