import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as torchvision_models
from torchvision import transforms

class ResnetBlockFC(nn.Module):
    """Fully connected ResNet Block"""
    def __init__(self, size_in, size_out=None, hidden_size=None):
        super().__init__()
        if size_out is None:
            size_out = size_in
        if hidden_size is None:
            hidden_size = size_in

        self.fc1 = nn.Linear(size_in, hidden_size)
        self.fc2 = nn.Linear(hidden_size, size_out)
        
        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        
        # Initialization
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x):
        net = F.relu(self.fc1(x))
        dx = self.fc2(net)

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
            
        return F.relu(x_s + dx)

class OccupancyNetwork(nn.Module):
    def __init__(self, args, feat_dim=512, hidden_size=256, n_blocks=5):
        super().__init__()
        
        # Image feature encoder
        self.feat_dim = feat_dim
        vision_model = torchvision_models.__dict__[args.arch](pretrained=True) # NOTE: use resnet18 pretrained encoder
        self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        
        # Coordinate encoder
        self.coord_fc = nn.Linear(3, hidden_size)
        
        # Conditional batch normalization
        self.cbn_fc = nn.Linear(feat_dim, 2*hidden_size)
        
        # ResNet blocks
        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for _ in range(n_blocks)
        ])
        
        # Final layer
        self.fc_out = nn.Linear(hidden_size, 1)
        
    def forward(self, p, c):
        """
        Args:
            p: point coordinates [B, N, 3]
            c: image features [B, feat_dim]
        Returns:
            occupancy values [B, N, 1]
        """
        batch_size, n_pts = p.shape[:2]
        
        # Encode coordinates
        net = self.coord_fc(p)  # [B, N, hidden]
        
        # Get batch norm parameters
        gamma, beta = self.cbn_fc(c).chunk(2, dim=1)  # [B, hidden]
        gamma = gamma.unsqueeze(1).expand(-1, n_pts, -1)  # [B, N, hidden]
        beta = beta.unsqueeze(1).expand(-1, n_pts, -1)   # [B, N, hidden]
        
        # Apply conditional batch norm
        net = gamma * net + beta
        
        # ResNet blocks
        for block in self.blocks:
            net = block(net)
            
        # Output
        out = self.fc_out(net)

        # Normalize to [0, 1]
        out = torch.sigmoid(out)
        return out

    def init_weights(self):
        '''
        Initialize weights other than the image encoder.
        '''
        pass

#####################################################################################

def create_grid_points_from_bounds(min_bound, max_bound, resolution):
    """Create grid points in normalized coordinate space"""
    x = torch.linspace(min_bound, max_bound, resolution)
    y = torch.linspace(min_bound, max_bound, resolution)
    z = torch.linspace(min_bound, max_bound, resolution)
    
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1)
    return grid_points.reshape(-1, 3)

def create_target_occupancy(voxels, resolution=32):
    """
    Creates a function that computes occupancy values for arbitrary points
    given a voxel grid.
    
    Args:
        voxels: Binary voxel grid tensor of shape [B, resolution, resolution, resolution]
        resolution: Voxel grid resolution (default: 32)
    
    Returns:
        function that maps points in [-1, 1]³ to occupancy values
    """
    voxels.squeeze_() # NOTE: debug - [B, 1, 32, 32, 32] -> [B, 32, 32, 32], res=32
    batch_size = voxels.shape[0]
    device = voxels.device

    def target_occupancy(p):
        """
        Args:
            p: Points tensor of shape [B, N, 3] in [-1, 1]³
        Returns:
            occupancy: Binary tensor of shape [B, N, 1]
        """
        # Convert points from [-1, 1]³ to [0, resolution-1]³
        p_vox = (p + 1) * (resolution - 1) / 2
        
        # # Clamp indices to valid range
        # p_vox = torch.clamp(p_vox, 0, resolution - 1)

        # TODO: Use trilinear interpolation to get occupancy values
        # Get integer and fractional parts
        p_floor = torch.floor(p_vox)
        p_ceil = torch.ceil(p_vox)
        p_frac = p_vox - p_floor

        # Clamp indices to valid range
        p_floor = torch.clamp(p_floor.long(), 0, resolution - 1)
        p_ceil = torch.clamp(p_ceil.long(), 0, resolution - 1)

        # Get corner points
        batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1)
        batch_idx = batch_idx.expand(-1, p_vox.shape[1], -1)

        # Get values at eight corners
        c000 = voxels[batch_idx[..., 0], p_floor[..., 0], p_floor[..., 1], p_floor[..., 2]]
        c001 = voxels[batch_idx[..., 0], p_floor[..., 0], p_floor[..., 1], p_ceil[..., 2]]
        c010 = voxels[batch_idx[..., 0], p_floor[..., 0], p_ceil[..., 1], p_floor[..., 2]]
        c011 = voxels[batch_idx[..., 0], p_floor[..., 0], p_ceil[..., 1], p_ceil[..., 2]]
        c100 = voxels[batch_idx[..., 0], p_ceil[..., 0], p_floor[..., 1], p_floor[..., 2]]
        c101 = voxels[batch_idx[..., 0], p_ceil[..., 0], p_floor[..., 1], p_ceil[..., 2]]
        c110 = voxels[batch_idx[..., 0], p_ceil[..., 0], p_ceil[..., 1], p_floor[..., 2]]
        c111 = voxels[batch_idx[..., 0], p_ceil[..., 0], p_ceil[..., 1], p_ceil[..., 2]]

        # Get interpolation weights
        x, y, z = p_frac[..., 0], p_frac[..., 1], p_frac[..., 2]
        
        # Perform trilinear interpolation
        c00 = c000 * (1 - x) + c100 * x
        c01 = c001 * (1 - x) + c101 * x
        c10 = c010 * (1 - x) + c110 * x
        c11 = c011 * (1 - x) + c111 * x

        c0 = c00 * (1 - y) + c10 * y
        c1 = c01 * (1 - y) + c11 * y

        occupancy = c0 * (1 - z) + c1 * z

        return occupancy.unsqueeze(-1).float()

    return target_occupancy

########################################################################################

def train_step(model: OccupancyNetwork, optimizer, scheduler, img_gt, target_occupancy):
    """
    Perform a single training step for occupancy network. 
    Input: 
        model - OccupancyNetwork model
        target_occupancy - function that maps points in [-1, 1]³ to occupancy values
    """
    model.train()
    optimizer.zero_grad()
    
    image_features = model.encoder(model.normalize(img_gt.permute(0,3,1,2))).squeeze(-1).squeeze(-1) # b x 512
    
    # Create random points for training
    device = image_features.device
    batch_size = image_features.shape[0]
    n_points = 2048  # number of points per training step
    
    # Sample random points in unit cube
    p = torch.rand(batch_size, n_points, 3).to(device) * 2 - 1
    
    # Get occupancy values for sampled points
    pred_occupancy = model(p, image_features)
    target = target_occupancy(p)  # assuming target_occupancy is a function
    
    # import ipdb
    # ipdb.set_trace()
    # Binary cross entropy loss
    loss = F.binary_cross_entropy(pred_occupancy, target)
    
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    return loss.detach()


def evaluate_occupancy_network(model, image_features, resolution=32):
    """Evaluate model on regular grid"""
    model.eval()
    device = image_features.device
    
    # Create grid points
    grid_points = create_grid_points_from_bounds(-1, 1, resolution)
    grid_points = grid_points.to(device)
    
    # Split points into chunks to avoid memory overflow
    chunk_size = 32768
    with torch.no_grad():
        occupancy_grid = []
        for i in range(0, grid_points.shape[0], chunk_size):
            p = grid_points[i:i+chunk_size].unsqueeze(0)
            p = p.expand(image_features.shape[0], -1, -1)
            
            pred = model(p, image_features)
            occupancy_grid.append(pred.squeeze(-1))
            
    occupancy_grid = torch.cat(occupancy_grid, dim=1)
    occupancy_grid = occupancy_grid.reshape(-1, 1, resolution, resolution, resolution)
    
    return occupancy_grid