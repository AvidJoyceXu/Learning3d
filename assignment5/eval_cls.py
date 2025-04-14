import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn.functional as F
torch.backends.cudnn.enabled = False
from models import cls_model, PointNet2ClsSSG
from utils import create_dir, rotate_point_cloud
import os

def visualize_point_cloud(points, title, save_path=None):
    """Visualize a single point cloud."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='.')
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def find_failure_cases(predictions, labels, points, num_classes=3):
    """Find failure cases for each class."""
    failures = {i: [] for i in range(num_classes)}
    for i in range(len(predictions)):
        if predictions[i] != labels[i]:
            failures[labels[i].item()].append((points[i], predictions[i].item()))
    return failures

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')
    parser.add_argument('--model_type', type=str, default="pointnet", help='The model type: pointnet or pointnet2')
    parser.add_argument('--normal_channel', action='store_true', help='Use normal channel for PointNet++')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)
    class_names = ['chair', 'vase', 'lamp']

    # ------ TO DO: Initialize Model for Classification Task ------
    if args.model_type == "pointnet2":
        model = PointNet2ClsSSG(num_classes=args.num_cls_class, normal_channel=args.normal_channel).to(args.device)
    else:
        model = cls_model(num_classes=args.num_cls_class).to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print("successfully loaded checkpoint from {}".format(model_path))

    # Load and prepare data
    ind = np.random.choice(10000, args.num_points, replace=False)
    test_data = np.load(args.test_data)[:,ind,:]
    test_label = np.load(args.test_label)

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
            predictions = model(rotated_data)
            pred_label = torch.max(predictions, 1)[1]
        
        # Compute accuracy
        accuracy = pred_label.eq(test_label_tensor).cpu().sum().item() / test_label_tensor.size()[0]
        rotation_results[angle] = accuracy
        print(f"Accuracy with {angle}° rotation: {accuracy:.4f}")
        
        # Visualize a few samples
        if angle in rotation_angles:  # Visualize original and 90-degree rotation
            num_samples = 3
            indices = np.random.choice(len(rotated_data), num_samples, replace=False)
            for idx in indices:
                points = rotated_data[idx].cpu().numpy()
                true_label = int(test_label_tensor[idx].item())
                pred = int(pred_label[idx].item())
                title = f"Rotation {angle}° - True: {class_names[true_label]}, Predicted: {class_names[pred]}"
                save_path = os.path.join(args.output_dir, f"rotation_{angle}_{idx}.png")
                visualize_point_cloud(points, title, save_path)

    # Experiment 2: Point Density Analysis
    print("\n=== Experiment 2: Point Density Analysis ===")
    point_densities = [1000, 5000, 10000]  # number of points
    density_results = {}
    
    for num_points in point_densities:
        print(f"\nTesting with {num_points} points:")
        # Sample points
        ind = np.random.choice(10000, num_points, replace=False)
        sampled_data = torch.from_numpy(np.load(args.test_data))[:,ind,:].to(args.device)
        test_label_tensor = torch.from_numpy(np.load(args.test_label)).to(args.device)
        
        # Make predictions
        with torch.no_grad():
            predictions = model(sampled_data)
            pred_label = torch.max(predictions, 1)[1]
        
        # Compute accuracy
        accuracy = pred_label.eq(test_label_tensor).cpu().sum().item() / test_label_tensor.size()[0]
        density_results[num_points] = accuracy
        print(f"Accuracy with {num_points} points: {accuracy:.4f}")
        
        # Visualize a few samples
        if num_points in [1000, 10000]:  # Visualize lowest and highest density
            num_samples = 3
            indices = np.random.choice(len(sampled_data), num_samples, replace=False)
            for idx in indices:
                points = sampled_data[idx].cpu().numpy()
                true_label = int(test_label_tensor[idx].item())
                pred = int(pred_label[idx].item())
                title = f"Points {num_points} - True: {class_names[true_label]}, Predicted: {class_names[pred]}"
                save_path = os.path.join(args.output_dir, f"density_{num_points}_{idx}.png")
                visualize_point_cloud(points, title, save_path)

    # Print summary of results
    print("\n=== Summary of Results ===")
    print("\nRotation Analysis:")
    for angle, acc in rotation_results.items():
        print(f"{angle}° rotation: {acc:.4f}")
    
    print("\nPoint Density Analysis:")
    for points, acc in density_results.items():
        print(f"{points} points: {acc:.4f}")

    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy(np.load(args.test_label)).to(args.device)

    # ------ TO DO: Make Prediction ------
    with torch.no_grad():
        predictions = model(test_data.to(args.device))
        pred_label = torch.max(predictions, 1)[1]

    # Compute Accuracy
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    print("test accuracy: {}".format(test_accuracy))

    # Visualize random samples
    num_samples = 3
    indices = np.random.choice(len(test_data), num_samples, replace=False)
    
    print("\nVisualizing random samples:")
    for idx in indices:
        points = test_data[idx].cpu().numpy()
        true_label = int(test_label[idx].item())
        pred = int(pred_label[idx].item())
        title = f"True: {class_names[true_label]}, Predicted: {class_names[pred]}"
        visualize_point_cloud(points, title, save_path=os.path.join(args.output_dir, f"sample_{idx}.png"))

    # Find and visualize failure cases
    failures = find_failure_cases(pred_label, test_label, test_data)
    
    print("\nAnalyzing failure cases:")
    for true_class in range(len(class_names)):
        if failures[true_class]:
            points, pred_class = failures[true_class][0]  # Take first failure case
            points = points.cpu().numpy()
            title = f"Failure Case - True: {class_names[true_class]}, Predicted: {class_names[pred_class]}"
            visualize_point_cloud(points, title, save_path=os.path.join(args.output_dir, f"failure_case_{true_class}_{pred_class}.png"))
            print(f"Found failure case for {class_names[true_class]}: misclassified as {class_names[pred_class]}")
        else:
            print(f"No failure cases found for {class_names[true_class]}")

    # Remove the debugger
    # import ipdb; ipdb.set_trace()

