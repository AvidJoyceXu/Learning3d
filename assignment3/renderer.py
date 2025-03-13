import torch
import torch.nn.functional as F

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False

    def _compute_weights(
        self,
        deltas,
        rays_density: torch.Tensor,
        eps: float = 1e-10
    ):
        # TODO (1.5): Compute transmittance using the equation described in the README
        # Calculate alpha (opacity) from density * delta
        alpha = 1 - torch.exp(-rays_density * deltas) # (N_rays, n_pts)
        
        # Compute transmittance T = exp(-sum(density * delta))
        # For numerical stability, we compute this in log space
        log_transmittance = torch.cumsum( # (N_rays, n_pts)
            torch.log(1 - alpha + eps),
            dim=1
        )
        # Shift the log_transmittance and prepend with 0 (T=1 for first sample)
        log_transmittance = torch.cat([
            torch.zeros_like(log_transmittance[:, :1]),
            log_transmittance[:, :-1]
        ], dim=1)
        transmittance = torch.exp(log_transmittance)

        # TODO (1.5): Compute weight used for rendering from transmittance and alpha
        # weights = T * (1 - exp(-sigma * delta))
        weights = transmittance * alpha
        
        return weights # Weights = T(x, x_ti) * alpha(x, x_ti)
    
    def _aggregate(
        self,
        weights: torch.Tensor,
        rays_feature: torch.Tensor
    ):
        # TODO (1.5): Aggregate (weighted sum of) features using weights
        # Weighted sum of features
        feature = torch.sum(weights * rays_feature, dim=1)
        
        return feature

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            implicit_output = implicit_fn(cur_ray_bundle)
            density = implicit_output['density']
            feature = implicit_output['feature']

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1), #
                density.view(-1, n_pts, 1)
            ) 
            #  ipdb; ipdb.set_trace()
            # TODO (1.5): Render (color) features using weights
            feature = self._aggregate(
                weights,
                feature.view(-1, n_pts, feature.shape[-1])
            )

            # TODO (1.5): Render depth map
            # Depth is the weighted sum of sample distances
            depth = self._aggregate(
                weights,
                depth_values.view(-1, n_pts, 1)
            )

            # Return
            cur_out = {
                'feature': feature,
                'depth': depth,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }
        return out


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class SphereTracingRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self.near = cfg.near
        self.far = cfg.far
        self.max_iters = cfg.max_iters
    
    def sphere_tracing(
        self,
        implicit_fn,
        origins, # Nx3
        directions, # Nx3
    ):
        '''
        Input:
            implicit_fn: a module that computes a SDF at a query point
            origins: N_rays X 3
            directions: N_rays X 3
        Output:
            points: N_rays X 3 points indicating ray-surface intersections. For rays that do not intersect the surface,
                    the point can be arbitrary.
            mask: N_rays X 1 (boolean tensor) denoting which of the input rays intersect the surface.
        '''
        # TODO (Q5): Implement sphere tracing
        # 1) Iteratively update points and distance to the closest surface
        #   in order to compute intersection points of rays with the implicit surface
        # 2) Maintain a mask with the same batch dimension as the ray origins,
        #   indicating which points hit the surface, and which do not
        
        # Initialize points at ray origins
        points = origins.clone()
        
        # Initialize mask for convergence
        mask = torch.zeros(origins.shape[0], 1, dtype=torch.bool, device=origins.device)
        
        # Normalize ray directions
        directions = F.normalize(directions, dim=-1)
        
        # Sphere tracing loop
        eps = 1e-5  # Distance threshold for surface intersection
        max_iters = self.max_iters
        min_dist = self.near
        max_dist = self.far
        total_distance = torch.zeros_like(mask, dtype=torch.float32)
        
        for _ in range(max_iters):
            # Get current SDF values
            distances = implicit_fn(points)
            
            # Check if we hit the surface (SDF value close to 0)
            hit_mask = torch.abs(distances) < eps
            
            # Update convergence mask
            mask = mask | hit_mask
            
            # Break if all rays have hit
            if torch.all(mask):
                break
                
            # Update points for non-converged rays
            not_converged = ~mask.squeeze(-1) 
            
            # Step size is the SDF value (distance to nearest surface)
            step_size = distances[not_converged]
            
            # Update points by stepping along ray
            points[not_converged] = points[not_converged] + directions[not_converged] * step_size
        
        # import ipdb; ipdb.set_trace()
        return points, mask

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir=None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]
            points, mask = self.sphere_tracing(
                implicit_fn,
                cur_ray_bundle.origins,
                cur_ray_bundle.directions
            )
            mask = mask.repeat(1,3)
            isect_points = points[mask].view(-1, 3)

            # Get color from implicit function with intersection points
            isect_color = implicit_fn.get_color(isect_points)

            # Return
            color = torch.zeros_like(cur_ray_bundle.origins)
            color[mask] = isect_color.view(-1)

            cur_out = {
                'color': color.view(-1, 3),
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


def sdf_to_density(signed_distance, alpha, beta):
    # TODO (Q7): Convert signed distance to density with alpha, beta parameters
    # Equation (2) and (3) from the paper:
    # σ(x) = αΨ_β(-d_Ω(x))
    # where Ψ_β(s) = {
    #   1/2 * exp(s/β)        if s ≤ 0
    #   1 - 1/2 * exp(-s/β)   if s > 0
    # }
    
    # Negate the signed distance as per equation (2)
    s = -signed_distance
    
    # Compute Laplace CDF (Ψ_β)
    # Case 1: s ≤ 0
    mask = s <= 0
    density = torch.zeros_like(signed_distance)
    density[mask] = 0.5 * torch.exp(s[mask] / beta)
    
    # Case 2: s > 0
    density[~mask] = 1.0 - 0.5 * torch.exp(-s[~mask] / beta)
    
    # Multiply by alpha to get final density
    density = alpha * density
    
    return density


class VolumeSDFRenderer(VolumeRenderer):
    def __init__(
        self,
        cfg
    ):
        super().__init__(cfg)

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False
        self.alpha = cfg.alpha
        self.beta = cfg.beta

        self.cfg = cfg

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir=None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            distance, color = implicit_fn.get_distance_color(cur_ray_bundle.sample_points)
            density = sdf_to_density(distance, self.alpha, self.beta)

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            geometry_color = torch.zeros_like(color)

            # Compute color
            color = self._aggregate(
                weights,
                color.view(-1, n_pts, color.shape[-1])
            )

            # Return
            cur_out = {
                'color': color,
                "geometry": geometry_color
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'volume': VolumeRenderer,
    'sphere_tracing': SphereTracingRenderer,
    'volume_sdf': VolumeSDFRenderer
}
