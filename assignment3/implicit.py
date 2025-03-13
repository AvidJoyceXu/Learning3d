import torch
import torch.nn.functional as F
from torch import autograd

from ray_utils import RayBundle


# Sphere SDF class
class SphereSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.radius = torch.nn.Parameter(
            torch.tensor(cfg.radius.val).float(), requires_grad=cfg.radius.opt
        )
        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)

        return torch.linalg.norm(
            points - self.center,
            dim=-1,
            keepdim=True
        ) - self.radius


# Box SDF class
class BoxSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.side_lengths = torch.nn.Parameter(
            torch.tensor(cfg.side_lengths.val).float().unsqueeze(0), requires_grad=cfg.side_lengths.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = torch.abs(points - self.center) - self.side_lengths / 2.0

        signed_distance = torch.linalg.norm(
            torch.maximum(diff, torch.zeros_like(diff)),
            dim=-1
        ) + torch.minimum(torch.max(diff, dim=-1)[0], torch.zeros_like(diff[..., 0]))

        return signed_distance.unsqueeze(-1)

# Torus SDF class
class TorusSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.radii = torch.nn.Parameter(
            torch.tensor(cfg.radii.val).float().unsqueeze(0), requires_grad=cfg.radii.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = points - self.center
        q = torch.stack(
            [
                torch.linalg.norm(diff[..., :2], dim=-1) - self.radii[..., 0],
                diff[..., -1],
            ],
            dim=-1
        )
        return (torch.linalg.norm(q, dim=-1) - self.radii[..., 1]).unsqueeze(-1)

sdf_dict = {
    'sphere': SphereSDF,
    'box': BoxSDF,
    'torus': TorusSDF,
}


# Converts SDF into density/feature volume
class SDFVolume(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )

        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )

        self.alpha = torch.nn.Parameter(
            torch.tensor(cfg.alpha.val).float(), requires_grad=cfg.alpha.opt
        )
        self.beta = torch.nn.Parameter(
            torch.tensor(cfg.beta.val).float(), requires_grad=cfg.beta.opt
        )

    def _sdf_to_density(self, signed_distance):
        # Convert signed distance to density with alpha, beta parameters
        return torch.where(
            signed_distance > 0,
            0.5 * torch.exp(-signed_distance / self.beta),
            1 - 0.5 * torch.exp(signed_distance / self.beta),
        ) * self.alpha

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.view(-1, 3)
        depth_values = ray_bundle.sample_lengths[..., 0]
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        ).view(-1, 1)

        # Transform SDF to density
        signed_distance = self.sdf(ray_bundle.sample_points)
        density = self._sdf_to_density(signed_distance)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(sample_points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        out = {
            'density': -torch.log(1.0 - density) / deltas,
            'feature': base_color * self.feature * density.new_ones(sample_points.shape[0], 1)
        }

        return out


# Converts SDF into density/feature volume
class SDFSurface(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )
        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )
    
    def get_distance(self, points):
        points = points.view(-1, 3)
        return self.sdf(points)

    def get_color(self, points):
        points = points.view(-1, 3)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        return base_color * self.feature * points.new_ones(points.shape[0], 1)
    
    def forward(self, points):
        return self.get_distance(points)

class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.include_input = include_input
        self.output_dim = n_harmonic_functions * 2 * in_channels

        if self.include_input:
            self.output_dim += in_channels

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)

        if self.include_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


class LinearWithRepeat(torch.nn.Linear):
    def forward(self, input):
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)


class MLPWithInputSkips(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips,
    ):
        super().__init__()

        layers = []

        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim

            linear = torch.nn.Linear(dimin, dimout)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))

        layers.append(torch.nn.Linear(dimout, output_dim))

        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = x

        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)

            y = layer(y)

        return y

    def init_weights(self):
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)


# TODO (Q3.1): Implement NeRF MLP
class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        
        # Create embeddings for xyz coordinates and view directions
        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        if hasattr(cfg, 'use_input'):
            self.harmonic_embedding_dir = HarmonicEmbedding(3, cfg.n_harmonic_functions_dir, include_input=cfg.use_input)
        else:
            self.harmonic_embedding_dir = HarmonicEmbedding(3, cfg.n_harmonic_functions_dir)
        
        # Get embedding dimensions
        self.embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        self.embedding_dim_dir = self.harmonic_embedding_dir.output_dim
        
        # Create main MLP for xyz processing
        self.mlp_xyz = MLPWithInputSkips(
            n_layers=cfg.n_layers_xyz,
            input_dim=self.embedding_dim_xyz,
            output_dim=cfg.n_hidden_neurons_xyz,  # Output features for color network
            skip_dim=self.embedding_dim_xyz,
            hidden_dim=cfg.n_hidden_neurons_xyz,  # 256 hidden neurons
            input_skips=cfg.append_xyz,  # Skip at 5th layer (index 4)
        )
        
        # Create density head
        self.density_head = torch.nn.Linear(cfg.n_hidden_neurons_xyz, 1)
    
        # Create color MLP
        self.mlp_color = MLPWithInputSkips(
            n_layers=1,  # Single layer network as shown in paper
            input_dim=cfg.n_hidden_neurons_xyz + self.embedding_dim_dir,  # 256 + view_dir_embedding
            output_dim=3,  # RGB colors
            skip_dim=0,  # No skip connections for color network
            hidden_dim=cfg.n_hidden_neurons_dir,
            input_skips=[],
        )
        
        # Store density noise std
        self.density_noise_std = cfg.density_noise_std
        self.use_view_dirs = cfg.use_view_dirs if hasattr(cfg, 'use_view_dirs') else True

    def init_weights(self):
        self.mlp_xyz.init_weights()
        self.mlp_color.init_weights()

       
    def forward(self, ray_bundle):
        # Get sample points and view directions
        sample_points = ray_bundle.sample_points.view(-1, 3)
        view_dirs = ray_bundle.directions.view(-1, 3)

        # Embed coordinates and view directions
        xyz_embedding = self.harmonic_embedding_xyz(sample_points)
        dir_embedding = self.harmonic_embedding_dir(view_dirs)
        
        # Get features from main MLP
        features = self.mlp_xyz(xyz_embedding, xyz_embedding)
        
        # Get density from features
        density = self.density_head(features) # NOTE: [debug]density is all negative now !!
      
        # Apply ReLU to get final density
        density = torch.relu(density)

        # 0311: Add noise **after** ReLU to avoid zero density
        # NOTE: only add noise during training
        if self.training and self.density_noise_std > 0:
            density = density + torch.randn_like(density) * self.density_noise_std
            density = torch.relu(density)
        if torch.sum(density) == 0:
            print(f"density is 0 during self.training: {self.training}")

        # Get colors by concatenating features and view directions
        if self.use_view_dirs:
            # dir_embedding: [batch_size, 24]
            # features: [batch_size * num_samples, 256]
            # Repeat dir_embedding for each sample
            n_pts_per_ray = features.shape[0]//dir_embedding.shape[0]
            dir_embedding = dir_embedding.repeat_interleave(n_pts_per_ray, dim=0)
            color_input = torch.cat([features, dir_embedding], dim=-1)
        else:
            color_input = features
            
        raw_colors = self.mlp_color(color_input, None)  # No skip connections for color
        colors = torch.sigmoid(raw_colors)  # Ensure colors are in [0,1]
        
        return {
            'density': density,
            'feature': colors,
        }


class NeuralSurface(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        
        # Create positional embedding for xyz coordinates
        self.harmonic_embedding_xyz = HarmonicEmbedding(
            3, cfg.n_harmonic_functions_xyz
        )
        
        # Get embedding dimension
        self.embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        
        # Create MLP for SDF prediction
        self.mlp_sdf = MLPWithInputSkips(
            n_layers=cfg.n_layers_distance,
            input_dim=self.embedding_dim_xyz,
            output_dim=1,  # Single SDF value output
            skip_dim=self.embedding_dim_xyz,
            hidden_dim=cfg.n_hidden_neurons_distance,
            input_skips=cfg.append_distance,  # Skip connections as specified in config
        )

    def get_distance(
        self,
        points
    ):
        '''
        Output:
            distance: N X 1 Tensor, where N is number of input points
        '''
        points = points.view(-1, 3)
        
        # Embed xyz coordinates
        points_embedding = self.harmonic_embedding_xyz(points)
        
        # Pass through MLP to get SDF values
        distances = self.mlp_sdf(points_embedding, points_embedding)
        # NOTE: No need to apply ReLU to **Signed** DF values
        return distances

    def get_color(
        self,
        points
    ):
        '''
        TODO: Q7
        Output:
            distance: N X 3 Tensor, where N is number of input points
        '''
        points = points.view(-1, 3)
        pass
    
    def get_distance_color(
        self,
        points
    ):
        '''
        TODO: Q7
        Output:
            distance, points: N X 1, N X 3 Tensors, where N is number of input points
        You may just implement this by independent calls to get_distance, get_color
            but, depending on your MLP implementation, it maybe more efficient to share some computation
        '''
        
    def forward(self, points):
        return self.get_distance(points)

    def get_distance_and_gradient(
        self,
        points
    ):
        has_grad = torch.is_grad_enabled()
        points = points.view(-1, 3)

        # Calculate gradient with respect to points
        with torch.enable_grad():
            points = points.requires_grad_(True)
            distance = self.get_distance(points)
            gradient = autograd.grad(
                distance,
                points,
                torch.ones_like(distance, device=points.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True
            )[0]
        
        return distance, gradient


implicit_dict = {
    'sdf_volume': SDFVolume,
    'nerf': NeuralRadianceField,
    'sdf_surface': SDFSurface,
    'neural_surface': NeuralSurface,
}
