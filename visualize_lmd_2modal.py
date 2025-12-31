import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add local project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, 'project')
DEFAULT_DATA_ROOT = os.environ.get('LMD_DATA_ROOT', os.path.join(SCRIPT_DIR, 'data'))
sys.path.append(PROJECT_ROOT)

import saverloader
# from fire import Fire  # Not needed anymore
from nets.segnet_relu import Segnet_relu
from nets.segnet_relu_feats_bias import Segnet_relu_feats_bias
import copy
import pickle
import utils.misc
import utils.vox
import utils.geom
import utils.basic
import random
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

random.seed(125)
np.random.seed(125)

# Scene settings
scene_centroid_x = 0.0
scene_centroid_y = 1.0
scene_centroid_z = 0.0

scene_centroid_py = np.array([scene_centroid_x,
                              scene_centroid_y,
                              scene_centroid_z]).reshape([1, 3])
scene_centroid = torch.from_numpy(scene_centroid_py).float()

# BEV bounds
XMIN, XMAX = -50, 50
ZMIN, ZMAX = -50, 50
YMIN, YMAX = -5, 5
bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

Z, Y, X = 200, 8, 200

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag


def save_lmd_features_2modal(global_step, mask_model, fusion_model, rgb_camXs, pix_T_cams, cam0_T_camXs, 
                            vox_util, sensor_occ_mem0, device, output_dir, modality='radar'):
    """
    Save LMD features for 2-modality (Camera + Radar/LiDAR) visualization
    """
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    png_output_dir = os.path.join(os.path.dirname(output_dir), 'visualize_2')
    os.makedirs(png_output_dir, exist_ok=True)
    
    mask_model.eval()
    fusion_model.eval()
    
    # Generate mask from mask model
    with torch.no_grad():
        act_mask, norm_mask, real_seg_bev_e, center_bev_e, offset_bev_e = mask_model(
                rgb_camXs=rgb_camXs,
                pix_T_cams=pix_T_cams,
                cam0_T_camXs=cam0_T_camXs,
                vox_util=vox_util,
                rad_occ_mem0=sensor_occ_mem0)
    
    # Get features from fusion model
    with torch.no_grad():
        camera_act_feat, camera_norm_feat, seg_bev_e, activations_camera, out_dict = fusion_model(
                    rgb_camXs=rgb_camXs,
                    pix_T_cams=pix_T_cams,
                    cam0_T_camXs=cam0_T_camXs,
                    vox_util=vox_util,
                    rad_occ_mem0=sensor_occ_mem0, 
                    act_mask=act_mask, 
                    norm_mask=norm_mask)
    
    # Extract features from output_prediction
    # Note: Even when using LiDAR, the model uses 'segmentation_rad' key
    features_dict = {
        'camera_features': out_dict.get('segmentation_cam', None),
        f'{modality}_features': out_dict.get('segmentation_rad', None),  # Both radar and lidar use 'rad' key
        'bias_prediction': seg_bev_e,  # This is the bias prediction
        'mask_prediction': real_seg_bev_e,  # From mask model
    }
    
    # Save features as numpy arrays
    for key, value in features_dict.items():
        if value is not None:
            np_value = value.cpu().numpy()
            save_path = os.path.join(output_dir, f"step_{global_step:04d}_{key}.npy")
            np.save(save_path, np_value)
            print(f"Saved {key} with shape {np_value.shape} to {save_path}")
    
    # Save as pickle file (Shapley style naming)
    save_path_pkl = os.path.join(output_dir, f"lmd_2modal_bev_maps_{global_step}.pkl")
    
    # Create results dictionary in Shapley format
    results = {
        'sample_idx': global_step,
        'lmd_maps': {
            'camera': features_dict.get('camera_features'),
            'radar' if modality == 'radar' else 'lidar': features_dict.get(f'{modality}_features'),
            'bias': features_dict.get('bias_prediction')
        },
        'modality': modality,
        'timestamp': time.time()
    }
    
    # Convert tensors to CPU before saving
    results_cpu = {}
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            results_cpu[key] = value.cpu()
        elif isinstance(value, dict):
            results_cpu[key] = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                               for k, v in value.items()}
        else:
            results_cpu[key] = value
    
    with open(save_path_pkl, 'wb') as f:
        pickle.dump(results_cpu, f)
    print(f"Results saved to {save_path_pkl}")
    
    # Create visualizations
    create_visualizations_2modal(features_dict, global_step, png_output_dir, modality)
    
    return features_dict


def create_visualizations_2modal(features_dict, global_step, output_dir, modality='radar'):
    """
    Visualize 2-modality LMD feature maps using Shapley-style visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    # Features to plot based on modality (using title format from Shapley)
    if modality == 'radar':
        features = {
            'Camera': features_dict.get('camera_features'),
            'Radar': features_dict.get('radar_features'), 
            'Bias': features_dict.get('bias_prediction'),
            'Mask': features_dict.get('mask_prediction')
        }
    else:  # lidar
        features = {
            'Camera': features_dict.get('camera_features'),
            'Lidar': features_dict.get('lidar_features'), 
            'Bias': features_dict.get('bias_prediction'),
            'Mask': features_dict.get('mask_prediction')
        }
    
    title_map = {
        'Camera': 'Camera-based Prediction',
        'Radar': 'Radar-based Prediction',
        'Lidar': 'Lidar-based Prediction',
        'Bias': 'Bias-based Prediction',
        'Mask': 'Model Prediction',
    }

    # Plot each feature
    for idx, (name, feature) in enumerate(features.items()):
        if idx >= 4:
            break
            
        if feature is not None:
            # Convert to numpy and select first channel
            map_np = feature[0, 0].cpu().numpy()
            
            # Use RdBu_r colormap with symmetric vmin/vmax (Shapley style)
            im = axes[idx].imshow(map_np, cmap='RdBu_r', 
                                 vmin=-np.abs(map_np).max(), 
                                 vmax=np.abs(map_np).max())
            title = title_map.get(name, f'{name} Prediction')
            axes[idx].set_title(title)
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        else:
            axes[idx].text(0.5, 0.5, 'Not Available', ha='center', va='center', 
                          transform=axes[idx].transAxes, fontsize=12)
            title = title_map.get(name, f'{name} Prediction')
            axes[idx].set_title(title)
            axes[idx].axis('off')
    
    plt.suptitle(f'LMD 2 Modal Value BEV Maps - Sample {global_step}', fontsize=16)
    plt.tight_layout()
    
    # Save to visualization_2 directory (Shapley style)
    vis_output = os.path.join(output_dir, f'lmd_2_modal_bev_visualization_{global_step}.png')
    plt.savefig(vis_output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {vis_output}")


def create_individual_visualizations_2modal(features_dict, global_step, output_dir, modality='radar'):
    """
    Create individual high-quality visualizations for each feature
    """
    individual_dir = os.path.join(output_dir, f'individual_features_step_{global_step:04d}_{modality}')
    os.makedirs(individual_dir, exist_ok=True)
    
    # Define visualization settings for each feature type
    if modality == 'radar':
        viz_configs = {
            'camera_features': {'cmap': 'viridis', 'title': 'Camera Feature Map'},
            'radar_features': {'cmap': 'plasma', 'title': 'Radar Feature Map'},
            'bias_prediction': {'cmap': 'RdBu_r', 'title': 'Bias Prediction'},
            'mask_prediction': {'cmap': 'coolwarm', 'title': 'Model Prediction'}
        }
    else:  # lidar
        viz_configs = {
            'camera_features': {'cmap': 'viridis', 'title': 'Camera Feature Map'},
            'lidar_features': {'cmap': 'hot', 'title': 'LiDAR Feature Map'},
            'bias_prediction': {'cmap': 'RdBu_r', 'title': 'Bias Prediction'},
            'mask_prediction': {'cmap': 'coolwarm', 'title': 'Model Prediction'}
        }
    
    for feature_name, config in viz_configs.items():
        if features_dict.get(feature_name) is not None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            
            # Get feature data
            feature = features_dict[feature_name]
            if len(feature.shape) > 3:
                feature_vis = feature[0, 0].cpu().numpy()
            else:
                feature_vis = feature[0].cpu().numpy()
            
            # Apply sigmoid for predictions
            if 'prediction' in feature_name or 'features' in feature_name:
                feature_vis = 1 / (1 + np.exp(-feature_vis))
            
            # Plot with appropriate scaling
            im = ax.imshow(feature_vis, cmap=config['cmap'], aspect='equal')
            ax.set_title(config['title'], fontsize=16, fontweight='bold', pad=15)
            ax.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label('Value', fontsize=12)
            
            # Add statistics
            stats_text = (f'Mean: {feature_vis.mean():.4f}\n'
                         f'Std: {feature_vis.std():.4f}\n'
                         f'Min: {feature_vis.min():.4f}\n'
                         f'Max: {feature_vis.max():.4f}')
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                   fontsize=11, verticalalignment='top')
            
            # Save individual plot
            save_path = os.path.join(individual_dir, f"{feature_name}_step_{global_step:04d}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
    print(f"Saved individual visualizations to {individual_dir}")


def load_data_sample(global_step, device, modality='radar', data_root=None):
    """
    Load camera and radar/lidar data for a specific step
    """
    if data_root is None:
        data_root = DEFAULT_DATA_ROOT
    if modality == 'radar':
        # Load camera/radar data
        data_dir = os.path.join(data_root, 'batch_1_validation_data')
        data_path = os.path.join(data_dir, f"target_data_0{global_step:4d}.pt")
        
        if os.path.exists(data_path):
            data = torch.load(data_path, map_location=device, weights_only=False)
            rgb_camXs, pix_T_cams, cam0_T_camXs, _, sensor_occ_mem0, seg_bev_g, valid_bev_g = data
            return rgb_camXs, pix_T_cams, cam0_T_camXs, sensor_occ_mem0, seg_bev_g, valid_bev_g
        else:
            raise FileNotFoundError(f"Data not found at {data_path}")
    else:  # lidar
        # Load camera/lidar data
        data_dir = os.path.join(data_root, 'batch_1_validation_data_lidar')
        data_path = os.path.join(data_dir, f"target_data_0{global_step:4d}.pt")
        
        if os.path.exists(data_path):
            data = torch.load(data_path, map_location=device, weights_only=False)
            rgb_camXs, pix_T_cams, cam0_T_camXs, _, sensor_occ_mem0, seg_bev_g, valid_bev_g = data
            return rgb_camXs, pix_T_cams, cam0_T_camXs, sensor_occ_mem0, seg_bev_g, valid_bev_g
        else:
            raise FileNotFoundError(f"Data not found at {data_path}")


def main():
    """Main function to run LMD BEV map calculation for two-modality models."""
    parser = argparse.ArgumentParser(description='Calculate LMD values as BEV maps for two-modality models')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (rgb_mine for camera+lidar, rad25 for camera+radar)')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['camera_lidar', 'camera_radar'],
                       help='Type of two-modality model')
    parser.add_argument('--start_step', type=int, default=0,
                       help='Start sample index')
    parser.add_argument('--end_step', type=int, default=6019,
                       help='End sample index')
    parser.add_argument('--step_interval', type=int, default=10,
                       help='Step interval between samples')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Visualize LMD maps')
    parser.add_argument('--output_dir', type=str, default='lmd_results',
                       help='Directory to save results')
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT,
                       help='Root with batch_1_validation_data and batch_1_validation_data_lidar')
    parser.add_argument('--encoder_type', type=str, default='res101',
                       help='Encoder type')
    parser.add_argument('--use_metaradar', action='store_true', default=True,
                       help='Use metaradar')
    parser.add_argument('--do_rgbcompress', action='store_true', default=True,
                       help='Do RGB compression')
    
    args = parser.parse_args()
    
    # Extract device ID
    device = args.device
    device_ids = [int(device.split(':')[1])] if ':' in device else [0]
    
    # Determine modality from model_type
    modality = 'radar' if args.model_type == 'camera_radar' else 'lidar'
    use_radar = (args.model_type == 'camera_radar')
    use_lidar = (args.model_type == 'camera_lidar')
    
    print(f"Running in {args.model_type} mode...")
    print(f"Using checkpoint: {args.checkpoint}")
    
    # Create output directories based on model type
    output_dir = os.path.join(args.output_dir, args.model_type)
    data_dir = os.path.join(output_dir, 'data')
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Set up vox_util
    vox_util = utils.vox.Vox_util(
        Z, Y, X,
        scene_centroid=scene_centroid.to(device),
        bounds=bounds,
        assert_cube=False)
    
    # Load mask model (segnet_relu)
    print("Loading mask model...")
    mask_model = Segnet_relu(Z, Y, X, vox_util, use_radar=use_radar, use_lidar=use_lidar, 
                           use_metaradar=args.use_metaradar, do_rgbcompress=args.do_rgbcompress, 
                           encoder_type=args.encoder_type)
    mask_model = mask_model.to(device)
    mask_model = torch.nn.DataParallel(mask_model, device_ids=device_ids)
    _ = saverloader.load(args.checkpoint, mask_model.module, ignore_load=None)
    parameters = list(mask_model.parameters())
    requires_grad(parameters, False)
    mask_model.eval()
    
    # Load fusion model (segnet_relu_feats_bias)
    print("Loading fusion model...")
    fusion_model = Segnet_relu_feats_bias(Z, Y, X, use_radar=use_radar, use_lidar=use_lidar, 
                                        use_metaradar=args.use_metaradar, do_rgbcompress=args.do_rgbcompress, 
                                        encoder_type=args.encoder_type, rand_flip=False)
    fusion_model = fusion_model.to(device)
    fusion_model = torch.nn.DataParallel(fusion_model, device_ids=device_ids)
    _ = saverloader.load(args.checkpoint, fusion_model.module, ignore_load=None)
    parameters = list(fusion_model.parameters())
    requires_grad(parameters, False)
    fusion_model.eval()
    
    print(f"Processing steps from {args.start_step} to {args.end_step} with interval {args.step_interval}")
    
    # Process each step
    for global_step in range(args.start_step, args.end_step + 1, args.step_interval):
        if global_step > 6019:  # Max available data
            break
            
        print(f"\n{'='*60}")
        print(f"Processing sample {global_step}")
        print(f"{'='*60}")
        
        try:
            # Load data
            rgb_camXs, pix_T_cams, cam0_T_camXs, sensor_occ_mem0, seg_bev_g, valid_bev_g = load_data_sample(
                global_step, device, modality, data_root=args.data_root)
            
            # Save features and create visualizations
            features = save_lmd_features_2modal(global_step, mask_model, fusion_model, 
                                               rgb_camXs, pix_T_cams, cam0_T_camXs, 
                                               vox_util, sensor_occ_mem0, device, data_dir, modality)
            
            # Print summary statistics (Shapley style)
            print("\nLMD Value Map Statistics:")
            lmd_maps = {
                'camera': features.get('camera_features'),
                modality: features.get(f'{modality}_features'),
                'bias': features.get('bias_prediction')
            }
            
            for modality_name, lmd_map in lmd_maps.items():
                if lmd_map is not None:
                    print(f"\n{modality_name.capitalize()}:")
                    print(f"  Shape: {lmd_map.shape}")
                    print(f"  Mean: {lmd_map.mean().item():.6f}")
                    print(f"  Std: {lmd_map.std().item():.6f}")
                    print(f"  Min: {lmd_map.min().item():.6f}")
                    print(f"  Max: {lmd_map.max().item():.6f}")
            
        except Exception as e:
            print(f"Error processing sample {global_step}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nProcessing complete! Results saved to {output_dir}")


def visualize_lmd_maps(lmd_maps, sample_idx, save_path):
    """
    Visualize LMD value BEV maps (Shapley style).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    # Plot each modality's LMD map
    for idx, (name, lmd_map) in enumerate(lmd_maps.items()):
        if idx >= 4:
            break
            
        if lmd_map is not None:
            # Convert to numpy and select first channel
            map_np = lmd_map[0, 0].cpu().numpy()
            
            im = axes[idx].imshow(map_np, cmap='RdBu_r', vmin=-np.abs(map_np).max(), vmax=np.abs(map_np).max())
            axes[idx].set_title(f'{name.capitalize()} LMD Values')
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        else:
            axes[idx].text(0.5, 0.5, 'Not Available', ha='center', va='center')
            axes[idx].axis('off')
    
    plt.suptitle(f'LMD Value BEV Maps - Sample {sample_idx}', fontsize=16)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    main()
