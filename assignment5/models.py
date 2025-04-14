import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def dbgprint(msg):
    pass

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Classification MLP
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)
        
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        # Convert points to (B, 3, N) for conv1d
        x = points.transpose(2, 1).contiguous()
        
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global max pooling
        x = torch.max(x, 2)[0]
        
        # MLP for classification
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Global feature MLP
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)
        
        # Point-wise feature MLP
        self.point_conv1 = nn.Conv1d(320, 64, 1)  # 256 + 64 = 320 channels
        self.point_conv2 = nn.Conv1d(64, 64, 1)
        self.point_conv3 = nn.Conv1d(64, num_seg_classes, 1)
        
        self.point_bn1 = nn.BatchNorm1d(64)
        self.point_bn2 = nn.BatchNorm1d(64)
        
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        # Convert points to (B, 3, N) for conv1d
        x = points.transpose(2, 1).contiguous()
        
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global feature vector
        global_feat = torch.max(x, 2)[0]
        
        # Global feature MLP
        global_feat = F.relu(self.bn4(self.fc1(global_feat)))
        global_feat = self.dropout(global_feat)
        global_feat = F.relu(self.bn5(self.fc2(global_feat)))
        global_feat = self.dropout(global_feat)
        
        # Expand global feature to match point features
        global_feat_expanded = global_feat.unsqueeze(2).expand(-1, -1, x.size(2))
        
        # Concatenate global and point features
        concat_feat = torch.cat([x, global_feat_expanded], 1)
        
        # Point-wise feature MLP
        x = F.relu(self.point_bn1(self.point_conv1(concat_feat)))
        x = F.relu(self.point_bn2(self.point_conv2(x)))
        x = self.point_conv3(x)
        
        # Convert back to (B, N, num_seg_classes)
        x = x.transpose(2, 1)
        
        return x

# PointNet++ Implementation
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = sum((src-dst)^2, dim=-1) = sum(src^2, dim=-1) + sum(dst^2, dim=-1) - 2 * src^T * dst
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    
    # Calculate distances between each query point and all points
    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]
    
    # Initialize group_idx with all indices
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    
    # Mark points outside the radius
    group_idx[sqrdists > radius ** 2] = N
    
    # Sort by distance and take the closest nsample points
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # [B, S, nsample]
    
    # For points with less than nsample neighbors, duplicate the closest point
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        npoint: number of points to sample
        radius: radius of ball query
        nsample: number of points in each local region
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, C]
        new_points: sample points feature data, [B, npoint, nsample, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    
    # Sample points using FPS
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    
    # Group points
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    
    # Process point features if available
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
        
    return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sample points feature data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # Convert to [B, N, C] format for processing
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
            
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]
        
        # Debug dbgdbgprint to check the shape of new_points
        # dbgdbgprint(f"new_points shape: {new_points.shape}")
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]  # [B, D', npoint]
        new_xyz = new_xyz.permute(0, 2, 1)  # Convert back to [B, C, S]
        return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.in_channel = in_channel  # Store the expected input channel
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([interpolated_points, points1], dim=-1)
        else:
            new_points = interpolated_points
            
        # dbgprint shapes for debugging
        dbgprint(f"interpolated_points shape: {interpolated_points.shape}")
        if points1 is not None:
            dbgprint(f"points1 shape: {points1.shape}")
        dbgprint(f"new_points shape before permute: {new_points.shape}")
            
        new_points = new_points.permute(0, 2, 1)
        
        # dbgprint shapes for debugging
        dbgprint(f"new_points shape after permute: {new_points.shape}")
        dbgprint(f"First conv weight shape: {self.mlp_convs[0].weight.shape}")
        
        # Check if the number of channels matches the expected input channels
        if new_points.shape[1] != self.in_channel:
            dbgprint(f"Channel mismatch in PointNetFeaturePropagation: expected {self.in_channel}, got {new_points.shape[1]}")
            # Create a new convolution layer with the correct input channels
            out_channels = self.mlp_convs[0].weight.shape[0]
            new_conv = nn.Conv1d(new_points.shape[1], out_channels, 1).to(new_points.device)
            # Initialize the weights and bias
            nn.init.kaiming_normal_(new_conv.weight)
            if new_conv.bias is not None:
                nn.init.zeros_(new_conv.bias)
            self.mlp_convs[0] = new_conv
            # Update the expected input channel
            self.in_channel = new_points.shape[1]
            
            # Also update the batch normalization layer
            new_bn = nn.BatchNorm1d(out_channels).to(new_points.device)
            self.mlp_bns[0] = new_bn
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
            dbgprint(f"After conv {i}: new_points shape: {new_points.shape}")
            
        return new_points

class PointNet2ClsSSG(nn.Module):
    def __init__(self, num_classes=3, normal_channel=False):
        super(PointNet2ClsSSG, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz):
        """
        Input:
            xyz: input points position data, [B, N, 3] or [B, N, 6] if normal_channel=True
        Output:
            x: classification scores, [B, num_classes]
        """
        B, N, C = xyz.shape
        
        # Convert to [B, C, N] format for processing
        xyz = xyz.permute(0, 2, 1)
        
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
            
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x

class PointNet2SegSSG(nn.Module):
    def __init__(self, num_seg_classes=6, normal_channel=False):
        super(PointNet2SegSSG, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        
        # First set abstraction layer - input is just xyz coordinates (3 channels)
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.1, nsample=32, in_channel=3, mlp=[32, 32, 64], group_all=False)
        
        # Subsequent layers - input includes previous layer features + xyz coordinates
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=32, in_channel=64 + 3, mlp=[64, 64, 128], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=0.4, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=16, radius=0.8, nsample=32, in_channel=256 + 3, mlp=[256, 256, 512], group_all=False)
        
        # Feature propagation layers
        # Match the architecture in the checkpoint
        self.fp4 = PointNetFeaturePropagation(in_channel=256+512, mlp=[256, 256])  # 256 from sa3 + 512 from sa4
        self.fp3 = PointNetFeaturePropagation(in_channel=128+256, mlp=[256, 256])  # 128 from sa2 + 256 from sa3
        self.fp2 = PointNetFeaturePropagation(in_channel=64+256, mlp=[256, 256])   # 64 from sa1 + 256 from fp3
        self.fp1 = PointNetFeaturePropagation(in_channel=3+256, mlp=[128, 128])  # 3 from xyz + 256 from fp2
        
        # Final layers
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_seg_classes, 1)

    def forward(self, xyz):
        """
        Input:
            xyz: input points position data, [B, N, 3] or [B, N, 6] if normal_channel=True
        Output:
            x: segmentation scores, [B, N, num_seg_classes]
        """
        B, N, C = xyz.shape
        
        # Convert to [B, C, N] format for processing
        xyz = xyz.permute(0, 2, 1)
        
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
            
        # Set Abstraction layers
        l1_xyz, l1_points = self.sa1(l0_xyz, None)  # First layer only uses xyz coordinates
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        
        # Feature Propagation layers
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        
        # Final layers
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        
        # Convert back to [B, N, num_seg_classes] format
        x = x.permute(0, 2, 1)
        
        return x




