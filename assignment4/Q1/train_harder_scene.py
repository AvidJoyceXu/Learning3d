import os
import torch
import imageio
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from model import Scene, Gaussians
from torch.utils.data import DataLoader
from data_utils import visualize_renders
from data_utils_harder_scene import get_nerf_datasets, trivial_collate

from pytorch3d.renderer import PerspectiveCameras
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def make_trainable(gaussians):

    ### YOUR CODE HERE ###
    # HINT: You can access and modify parameters from gaussians
    gaussians.means.requires_grad = True
    gaussians.pre_act_scales.requires_grad = True
    gaussians.colours.requires_grad = True
    gaussians.pre_act_opacities.requires_grad = True
    
    # For isotropic Gaussians, we don't need to make quaternions trainable
    if not gaussians.is_isotropic:
        gaussians.pre_act_quats.requires_grad = True

def setup_optimizer(gaussians):

    gaussians.check_if_trainable()

    ### YOUR CODE HERE ###
    # HINT: Modify the learning rates to reasonable values. We have intentionally
    # set very high learning rates for all parameters.
    # HINT: Consider reducing the learning rates for parameters that seem to vary too
    # fast with the default settings.
    # HINT: Consider setting different learning rates for different sets of parameters.
    if gaussians.is_isotropic:
        print("Using isotropic gaussians to train baselines")
        parameters = [
        # Means should have a smaller learning rate since they control position
        {'params': [gaussians.means], 'lr': 0.00005, "name": "means"},
        # Scales affect the size/shape of Gaussians - moderate learning rate
        {'params': [gaussians.pre_act_scales], 'lr': 0.001, "name": "scales"},
        # Colors and opacities can change more rapidly
        {'params': [gaussians.colours], 'lr': 0.005, "name": "colours"},
        {'params': [gaussians.pre_act_opacities], 'lr': 0.005, "name": "opacities"},
        ]
    else:
        print("Using anisotropic gaussians to train harder scene")
        parameters = [
            {'params': [gaussians.pre_act_opacities], 'lr': 0.01, "name": "opacities"},
            {'params': [gaussians.pre_act_scales], 'lr': 0.005, "name": "scales"},
            {'params': [gaussians.colours], 'lr': 0.02, "name": "colours"},
            {'params': [gaussians.means], 'lr': 0.001, "name": "means"},
            {'params': [gaussians.pre_act_quats], 'lr': 0.0001, "name": "quats"},
        ]
    optimizer = torch.optim.Adam(parameters, lr=0.0, eps=1e-15)
    
    # Creating a learning rate scheduler to gradually decrease learning rate
    # This helps with convergence in the later stages of training
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    return optimizer, scheduler

def ndc_to_screen_camera(camera, img_size = (128, 128)):

    min_size = min(img_size[0], img_size[1])

    screen_focal = camera.focal_length * min_size / 2.0
    screen_principal = torch.tensor([[img_size[0]/2, img_size[1]/2]]).to(torch.float32)

    return PerspectiveCameras(
        R=camera.R, T=camera.T, in_ndc=False,
        focal_length=screen_focal, principal_point=screen_principal,
        image_size=(img_size,),
    )

def run_training(args):

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)

    train_dataset, val_dataset, _ = get_nerf_datasets(
        dataset_name="materials", data_root=args.data_path,
        image_size=[128, 128],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    train_itr = iter(train_loader)

    # Preparing some code for visualization
    viz_gif_path_1 = os.path.join(args.out_path, "q1_harder_training_progress.gif")
    viz_gif_path_2 = os.path.join(args.out_path, "q1_harder_training_final_renders.gif")
    viz_idxs = np.linspace(0, len(train_dataset)-1, 5).astype(np.int32)[:4]

    gt_viz_imgs = [(train_dataset[i]["image"]*255.0).numpy().astype(np.uint8) for i in viz_idxs]
    gt_viz_imgs = [np.array(Image.fromarray(x).resize((256, 256))) for x in gt_viz_imgs]
    gt_viz_img = np.concatenate(gt_viz_imgs, axis=1)

    viz_cameras = [ndc_to_screen_camera(train_dataset[i]["camera"]).cuda() for i in viz_idxs]

    # Init gaussians and scene
    gaussians = Gaussians(
        num_points=10000, init_type=args.init_type,
        device=args.device, isotropic=args.isotropic  # Using anisotropic Gaussians for better modeling
    )
    scene = Scene(gaussians)

    # Making gaussians trainable and setting up optimizer
    make_trainable(gaussians)
    optimizer, scheduler = setup_optimizer(gaussians)

    # Training loop
    viz_frames = []
    prev_loss = float('inf')
    patience = 0
    max_patience = 10
    loss_history = []
    
    for itr in range(args.num_itrs):

        # Fetching data
        try:
            data = next(train_itr)
        except StopIteration:
            train_itr = iter(train_loader)
            data = next(train_itr)

        gt_img = data[0]["image"].cuda()
        camera = ndc_to_screen_camera(data[0]["camera"]).cuda()

        # Rendering scene using gaussian splatting
        ### YOUR CODE HERE ###
        # HINT: Can any function from the Scene class help?
        # HINT: Set bg_colour to (0.0, 0.0, 0.0)
        # HINT: Set img_size to (128, 128)
        # HINT: Get per_splat from args.gaussians_per_splat
        # HINT: camera is available above
        pred_img, _, _ = scene.render(
            camera=camera,
            per_splat=args.gaussians_per_splat,
            img_size=(128, 128),
            bg_colour=(0.0, 0.0, 0.0)
        )

        # Compute loss
        ### YOUR CODE HERE ###
        # Using a combination of L1 loss and SSIM loss for better performance
        # L1 loss helps with general reconstruction
        l1_loss = torch.nn.functional.l1_loss(pred_img, gt_img)
        
        # Add SSIM loss calculation - convert tensors to proper format for SSIM calculation
        pred_img_batch = pred_img.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        gt_img_batch = gt_img.permute(2, 0, 1).unsqueeze(0)      # [1, 3, H, W]
        
        # Using pytorch-msssim library if available
        # If not available, can be approximated with structural similarity metrics 
        try:
            from pytorch_msssim import ssim
            ssim_loss = 1.0 - ssim(pred_img_batch, gt_img_batch, data_range=1.0, size_average=True)
        except ImportError:
            # Fallback to just L1 loss if pytorch-msssim is not available
            ssim_loss = 0.0
        
        # Adding a small L2 regularization on the means to prevent them from moving too far
        # This helps with the randomly initialized points
        reg_loss = 0.0001 * torch.mean(torch.sum(gaussians.means ** 2, dim=1))
        
        # Weight for each loss component
        w_l1 = 1.0
        w_ssim = 0.2
        w_reg = 1.0
        
        loss = w_l1 * l1_loss + w_ssim * ssim_loss + w_reg * reg_loss

        loss.backward()
        optimizer.step()
        
        # Adjust learning rate based on loss trends
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        # If loss isn't decreasing, adjust patience counter
        if loss_val >= prev_loss:
            patience += 1
        else:
            patience = 0
            
        # If we've been stuck for a while, adjust learning rate
        if patience >= max_patience:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.8
            patience = 0
            print(f"[*] Reducing learning rate for {param_group['name']}")
            
        prev_loss = loss_val
        
        # Apply scheduler every 100 iterations
        if itr % 100 == 0 and itr > 0:
            scheduler.step()
            
        optimizer.zero_grad()

        print(f"[*] Itr: {itr:07d} | Loss: {loss:0.3f}")

        if itr % args.viz_freq == 0:
            # Attempt adaptive density control every visualization step
            # Only after some initial training
            viz_frame = visualize_renders(
                scene, gt_viz_img,
                viz_cameras, (128, 128)
            )
            viz_frames.append(viz_frame)

    print("[*] Training Completed.")

    # Saving training progess GIF
    imageio.mimwrite(viz_gif_path_1, viz_frames, loop=0, duration=(1/10.0)*1000)

    # Creating renderings of the training views after training is completed.
    frames = []
    viz_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=False, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    for viz_data in tqdm(viz_loader, desc="Creating Visualization"):
        gt_img = viz_data[0]["image"].cuda()
        camera = ndc_to_screen_camera(viz_data[0]["camera"]).cuda()

        with torch.no_grad():

            # Rendering scene using gaussian splatting
            ### YOUR CODE HERE ###
            # HINT: Can any function from the Scene class help?
            # HINT: Set bg_colour to (0.0, 0.0, 0.0)
            # HINT: Set img_size to (128, 128)
            # HINT: Get per_splat from args.gaussians_per_splat
            # HINT: camera is available above
            pred_img, _, _ = scene.render(
                camera=camera,
                per_splat=args.gaussians_per_splat,
                img_size=(128, 128),
                bg_colour=(0.0, 0.0, 0.0)
            )

        pred_npy = pred_img.detach().cpu().numpy()
        pred_npy = (np.clip(pred_npy, 0.0, 1.0) * 255.0).astype(np.uint8)
        frames.append(pred_npy)

    # Saving renderings
    imageio.mimwrite(viz_gif_path_2, frames, loop=0, duration=(1/10.0)*1000)

    # Running evaluation using the test dataset
    psnr_vals, ssim_vals = [], []
    for val_data in tqdm(val_loader, desc="Running Evaluation"):

        gt_img = val_data[0]["image"].cuda()
        camera = ndc_to_screen_camera(val_data[0]["camera"]).cuda()

        with torch.no_grad():

            # Rendering scene using gaussian splatting
            # Rendering scene using gaussian splatting
            ### YOUR CODE HERE ###
            # HINT: Can any function from the Scene class help?
            # HINT: Set bg_colour to (0.0, 0.0, 0.0)
            # HINT: Set img_size to (128, 128)
            # HINT: Get per_splat from args.gaussians_per_splat
            # HINT: camera is available above
            pred_img, _, _ = scene.render(
                camera=camera,
                per_splat=args.gaussians_per_splat,
                img_size=(128, 128),
                bg_colour=(0.0, 0.0, 0.0)
            )

            gt_npy = gt_img.detach().cpu().numpy()
            pred_npy = pred_img.detach().cpu().numpy()
            psnr = peak_signal_noise_ratio(gt_npy, pred_npy)
            ssim = structural_similarity(gt_npy, pred_npy, channel_axis=-1, data_range=1.0)

            psnr_vals.append(psnr)
            ssim_vals.append(ssim)

    mean_psnr = np.mean(psnr_vals)
    mean_ssim = np.mean(ssim_vals)
    print(f"[*] Evaluation --- Mean PSNR: {mean_psnr:.3f}")
    print(f"[*] Evaluation --- Mean SSIM: {mean_ssim:.3f}")

def get_args():

    parser = argparse.ArgumentParser()
    # NOTE: Baseline
    parser.add_argument(
        "--isotropic", default=False, type=bool,
        help="Whether to use isotropic gaussians."
    )
    parser.add_argument(
        "--init_type", default="random", type=str,
        help="Whether to use random or points initialization."
    )
    parser.add_argument(
        "--out_path", default="./output", type=str,
        help="Path to the directory where output should be saved to."
    )
    parser.add_argument(
        "--data_path", default="./data/materials", type=str,
        help="Path to the dataset."
    )
    parser.add_argument(
        "--gaussians_per_splat", default=-1, type=int,
        help=(
            "Number of gaussians to splat in one function call. If set to -1, "
            "then all gaussians in the scene are splat in a single function call. "
            "If set to any other positive interger, then it determines the number of "
            "gaussians to splat per function call (the last function call might splat "
            "lesser number of gaussians). In general, the algorithm can run faster "
            "if more gaussians are splat per function call, but at the cost of higher GPU "
            "memory consumption."
        )
    )
    parser.add_argument(
        "--num_itrs", default=20000, type=int,
        help="Number of iterations to train the model."
    )
    parser.add_argument(
        "--viz_freq", default=20, type=int,
        help="Frequency with which visualization should be performed."
    )
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    run_training(args)
