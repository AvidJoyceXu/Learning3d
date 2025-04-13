import numpy as np
import argparse
import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg
import os


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


def compute_segmentation_accuracy(pred_labels, true_labels):
    """Compute segmentation accuracy for a single object."""
    return (pred_labels == true_labels).mean()


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model(num_seg_classes=args.num_seg_class).to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print("successfully loaded checkpoint from {}".format(model_path))

    # Load and prepare data
    ind = np.random.choice(10000, args.num_points, replace=False)
    test_data = np.load(args.test_data)[:,ind,:]
    test_label = np.load(args.test_label)[:,ind]

    # Experiment 1: Rotation Analysis
    print("\n=== Experiment 1: Rotation Analysis ===")
    rotation_angles = [0, 30, 60, 90]  # degrees
    rotation_results = {}
    
    for angle in rotation_angles:
        print(f"\nTesting with {angle} degrees rotation:")
        # Rotate all point clouds
        rotated_data = np.array([rotate_point_cloud(pc, angle_x=angle) for pc in test_data])
        rotated_data = torch.from_numpy(rotated_data).to(args.device)
        test_label_tensor = torch.from_numpy(test_label).to(args.device)
        
        # Make predictions
        with torch.no_grad():
            pred_label = model(rotated_data)
            pred_label = torch.max(pred_label, 2)[1]
        
        # Compute accuracy
        accuracy = pred_label.eq(test_label_tensor).cpu().sum().item() / (test_label_tensor.reshape((-1,1)).size()[0])
        rotation_results[angle] = accuracy
        print(f"Accuracy with {angle}° rotation: {accuracy:.4f}")
        
        # Visualize a few samples
        if angle in [0, 90]:  # Visualize original and 90-degree rotation
            num_samples = 3
            indices = np.random.choice(len(rotated_data), num_samples, replace=False)
            for idx in indices:
                # Save ground truth visualization
                gt_path = os.path.join(args.output_dir, f"rotation_{angle}_gt_{idx}.gif")
                viz_seg(rotated_data[idx], test_label_tensor[idx], gt_path, args.device)
                
                # Save prediction visualization
                pred_path = os.path.join(args.output_dir, f"rotation_{angle}_pred_{idx}.gif")
                viz_seg(rotated_data[idx], pred_label[idx], pred_path, args.device)
                print(f"Saved visualizations for rotation {angle}° object {idx}")

    # Experiment 2: Point Density Analysis
    print("\n=== Experiment 2: Point Density Analysis ===")
    point_densities = [1000, 5000, 10000]  # number of points
    density_results = {}
    
    for num_points in point_densities:
        print(f"\nTesting with {num_points} points:")
        # Sample points
        ind = np.random.choice(10000, num_points, replace=False)
        sampled_data = torch.from_numpy(np.load(args.test_data))[:,ind,:].to(args.device)
        sampled_label = torch.from_numpy(np.load(args.test_label))[:,ind].to(args.device)
        
        # Make predictions
        with torch.no_grad():
            pred_label = model(sampled_data)
            pred_label = torch.max(pred_label, 2)[1]
        
        # Compute accuracy
        accuracy = pred_label.eq(sampled_label).cpu().sum().item() / (sampled_label.reshape((-1,1)).size()[0])
        density_results[num_points] = accuracy
        print(f"Accuracy with {num_points} points: {accuracy:.4f}")
        
        # Visualize a few samples
        if num_points in [1000, 10000]:  # Visualize lowest and highest density
            num_samples = 3
            indices = np.random.choice(len(sampled_data), num_samples, replace=False)
            for idx in indices:
                # Save ground truth visualization
                gt_path = os.path.join(args.output_dir, f"density_{num_points}_gt_{idx}.gif")
                viz_seg(sampled_data[idx], sampled_label[idx], gt_path, args.device)
                
                # Save prediction visualization
                pred_path = os.path.join(args.output_dir, f"density_{num_points}_pred_{idx}.gif")
                viz_seg(sampled_data[idx], pred_label[idx], pred_path, args.device)
                print(f"Saved visualizations for density {num_points} object {idx}")

    # Print summary of results
    print("\n=== Summary of Results ===")
    print("\nRotation Analysis:")
    for angle, acc in rotation_results.items():
        print(f"{angle}° rotation: {acc:.4f}")
    
    print("\nPoint Density Analysis:")
    for points, acc in density_results.items():
        print(f"{points} points: {acc:.4f}")