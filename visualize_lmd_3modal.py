import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, 'project')
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

DEFAULT_DATA_ROOT = os.environ.get('LMD_DATA_ROOT', os.path.join(SCRIPT_DIR, 'data'))
DEFAULT_CKPT_DIR = os.environ.get(
    'LMD_CKPT_DIR',
    os.path.join(SCRIPT_DIR, 'checkpoints', '4x5_3e-4s_cam_rad_lid_18:06:48'),
)
DEFAULT_OUTPUT_DIR = os.environ.get(
    'LMD_OUTPUT_DIR',
    os.path.join(SCRIPT_DIR, 'lmd_results', '3modal'),
)

import saverloader
from nets.segnet_relu_feats import Segnet_relu_feats
from nets.segnet_relu_feats_bias_three import Segnet_relu_feats_bias as Segnet_relu_feats_bias_three
import copy
import pickle
import utils.misc
import utils.improc
import utils.vox
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


def _cam_radar_path(data_root, step):
    return os.path.join(data_root, "batch_1_validation_data", f"target_data_0{step:4d}.pt")


def _lidar_path(data_root, step):
    return os.path.join(data_root, "batch_1_validation_data_lidar", f"target_data_0{step:4d}.pt")


def save_lmd_features_3modal(global_step, mask_model, fusion_model, rgb_camXs, pix_T_cams, cam0_T_camXs, 
                            vox_util, in_occ_mem0, lid_occ_mem0, device, output_dir):
    """
    Save LMD features for 3-modality (Camera + Radar + LiDAR) with bias prediction
    """
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    vis_output_dir = os.path.join(output_dir, 'visualize_3')
    os.makedirs(vis_output_dir, exist_ok=True)
    
    mask_model.eval()
    fusion_model.eval()
    
    # Generate mask from mask model
    with torch.no_grad():
        act_mask, norm_mask, real_seg_bev_e, center_bev_e, offset_bev_e = mask_model(
                rgb_camXs=rgb_camXs,
                pix_T_cams=pix_T_cams,
                cam0_T_camXs=cam0_T_camXs,
                vox_util=vox_util,
                rad_occ_mem0=[lid_occ_mem0, in_occ_mem0])
    
    # Get features from fusion model
    with torch.no_grad():
        camera_act_feat, camera_norm_feat, seg_bev_e, activations_camera, out_dict = fusion_model(
                    rgb_camXs=rgb_camXs,
                    pix_T_cams=pix_T_cams,
                    cam0_T_camXs=cam0_T_camXs,
                    vox_util=vox_util,
                    rad_occ_mem0=[lid_occ_mem0, in_occ_mem0], 
                    act_mask=act_mask, 
                    norm_mask=norm_mask)
    
    # Extract features from output_prediction
    features_dict = {
        'camera_features': out_dict.get('segmentation_cam', None),
        'radar_features': out_dict.get('segmentation_rad', None),
        'lidar_features': out_dict.get('segmentation_lid', None),
        'bias_prediction': seg_bev_e,  # Final bias prediction
        'mask_prediction': real_seg_bev_e,  # From mask model
    }
    
    # Save features as numpy arrays
    for key, value in features_dict.items():
        if value is not None:
            np_value = value.cpu().numpy()
            save_path = os.path.join(output_dir, f"step_{global_step:04d}_{key}.npy")
            np.save(save_path, np_value)
            print(f"Saved {key} with shape {np_value.shape} to {save_path}")
    
    # Save as pickle file (LMD style naming)
    save_path_pkl = os.path.join(output_dir, f"lmd_3modal_bev_maps_{global_step}.pkl")
    
    # Create results dictionary in LMD format
    results = {
        'sample_idx': global_step,
        'lmd_maps': {
            'camera': features_dict.get('camera_features'),
            'radar': features_dict.get('radar_features'),
            'lidar': features_dict.get('lidar_features'),
            'bias': features_dict.get('bias_prediction')
        },
        'mask_prediction': features_dict.get('mask_prediction'),
        'modality': '3modal',
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
    create_visualizations_3modal(features_dict, global_step, vis_output_dir)
    
    return features_dict


def create_visualizations_3modal(features_dict, global_step, output_dir):
    """
    Visualize 3-modality LMD feature maps
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Features to plot
    features = {
        'Camera': features_dict.get('camera_features'),
        'Radar': features_dict.get('radar_features'),
        'Lidar': features_dict.get('lidar_features'),
        'Bias': features_dict.get('bias_prediction'),
        'Mask': features_dict.get('mask_prediction'),
        'Combined': None  # Placeholder for combined visualization
    }

    title_map = {
        'Camera': 'Camera-based Prediction',
        'Radar': 'Radar-based Prediction',
        'Lidar': 'Lidar-based Prediction',
        'Bias': 'Bias-based Prediction',
        'Mask': 'Model Prediction',
        'Combined': 'Combined Prediction',
    }
    
    # Calculate combined prediction
    if all(features[k] is not None for k in ['Camera', 'Radar', 'Lidar', 'Bias']):
        features['Combined'] = features['Camera'] + features['Radar'] + features['Lidar'] + features['Bias']
    
    # Plot each feature
    for idx, (name, feature) in enumerate(features.items()):
        if idx >= 6:
            break
            
        if feature is not None:
            # Convert to numpy and select first channel
            map_np = feature[0, 0].cpu().numpy()
            
            # Use RdBu_r colormap with symmetric vmin/vmax
            im = axes[idx].imshow(map_np, cmap='RdBu_r', 
                                 vmin=-np.abs(map_np).max(), 
                                 vmax=np.abs(map_np).max())
            axes[idx].set_title(title_map.get(name, f'{name} Prediction'))
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        else:
            axes[idx].text(0.5, 0.5, 'Not Available', ha='center', va='center', 
                          transform=axes[idx].transAxes, fontsize=12)
            axes[idx].set_title(title_map.get(name, f'{name} Prediction'))
            axes[idx].axis('off')
    
    plt.suptitle(f'LMD 3-Modal Value BEV Maps - Sample {global_step}', fontsize=16)
    plt.tight_layout()
    
    # Save visualization
    vis_output = os.path.join(output_dir, f'lmd_3modal_bev_visualization_{global_step}.png')
    plt.savefig(vis_output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {vis_output}")


def create_individual_visualizations_3modal(features_dict, global_step, output_dir):
    """
    Create individual high-quality visualizations for each feature
    """
    individual_dir = os.path.join(output_dir, f'individual_features_step_{global_step:04d}')
    os.makedirs(individual_dir, exist_ok=True)
    
    title_map = {
        'camera': 'Camera-based Prediction',
        'radar': 'Radar-based Prediction',
        'lidar': 'Lidar-based Prediction',
        'bias': 'Bias-based Prediction',
        'mask': 'Model Prediction',
    }

    # Features to save individually
    features = {
        'camera': features_dict.get('camera_features'),
        'radar': features_dict.get('radar_features'),
        'lidar': features_dict.get('lidar_features'),
        'bias': features_dict.get('bias_prediction'),
        'mask': features_dict.get('mask_prediction')
    }
    
    for name, feature in features.items():
        if feature is not None:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Convert to numpy and select first channel
            map_np = feature[0, 0].cpu().numpy()
            
            # Use RdBu_r colormap with symmetric vmin/vmax
            im = ax.imshow(map_np, cmap='RdBu_r', 
                          vmin=-np.abs(map_np).max(), 
                          vmax=np.abs(map_np).max())
            title = title_map.get(name, f'{name.capitalize()} Prediction')
            ax.set_title(f'{title} - Step {global_step}', fontsize=14)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            save_path = os.path.join(individual_dir, f'{name}_lmd_step_{global_step:04d}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()


def main(
        exp_name='visualize_lmd_3modal',
        init_dir=DEFAULT_CKPT_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
        data_root=DEFAULT_DATA_ROOT,
        ignore_load=None,
        # eval settings
        start_step=1,
        end_step=6019,
        step_interval=50,
        visualize=True,
        # model
        encoder_type='res101',
        use_radar=True,
        use_lidar=True,
        use_metaradar=True,
        do_rgbcompress=True,
        # cuda
        device_ids=[0],
):
    batch_size = 1
    device = 'cuda:%d' % device_ids[0]
    
    # Model name
    model_name = "%s" % init_dir.split('/')[-1]
    print(f'Model: {model_name}')
    print(f'Output directory: {output_dir}')
    
    # Create output directories
    data_output_dir = os.path.join(output_dir, 'data')
    vis_output_dir = os.path.join(data_output_dir, 'visualize_3')
    os.makedirs(data_output_dir, exist_ok=True)
    os.makedirs(vis_output_dir, exist_ok=True)
    
    # Set up vox_util
    vox_util = utils.vox.Vox_util(
        Z, Y, X,
        scene_centroid=scene_centroid.to(device),
        bounds=bounds,
        assert_cube=False)
    
    # Set up mask model
    model = Segnet_relu_feats(Z, Y, X, use_radar=use_radar, use_lidar=use_lidar, 
                             use_metaradar=use_metaradar, do_rgbcompress=do_rgbcompress, 
                             encoder_type=encoder_type)
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    _ = saverloader.load(init_dir, model.module, ignore_load=ignore_load)
    parameters = list(model.parameters())
    requires_grad(parameters, False)
    
    # Set up fusion model with bias
    fusion_model = Segnet_relu_feats_bias_three(Z, Y, X, use_radar=use_radar, use_lidar=use_lidar, 
                                               use_metaradar=use_metaradar, do_rgbcompress=do_rgbcompress, 
                                               encoder_type=encoder_type, rand_flip=False)
    fusion_model = fusion_model.to(device)
    fusion_model = torch.nn.DataParallel(fusion_model, device_ids=device_ids)
    _ = saverloader.load(init_dir, fusion_model.module, ignore_load=ignore_load)
    parameters = list(fusion_model.parameters())
    requires_grad(parameters, False)
    
    fusion_model.eval()
    model.eval()
    
    print(f"Processing steps from {start_step} to {end_step} with interval {step_interval}")
    
    # Process selected steps
    for global_step in range(start_step, min(end_step + 1, 6020), step_interval):
        print(f"\nProcessing step {global_step}...")
        
        try:
            # Load camera/radar data
            camera_radar_data = torch.load(
                _cam_radar_path(data_root, global_step),
                map_location=torch.device(device),
                weights_only=False,
            )
            rgb_camXs, pix_T_cams, cam0_T_camXs, _, in_occ_mem0, seg_bev_g, valid_bev_g = camera_radar_data
            
            # Load lidar data
            _, _, _, _, lid_occ_mem0, _, _ = torch.load(
                _lidar_path(data_root, global_step),
                map_location=torch.device(device),
                weights_only=False,
            )
            
            # Save LMD features
            features_dict = save_lmd_features_3modal(
                global_step, model, fusion_model, rgb_camXs, pix_T_cams, cam0_T_camXs,
                vox_util, in_occ_mem0, lid_occ_mem0, device, data_output_dir
            )
            
            # Tile visualization is created in save_lmd_features_3modal.
                
        except FileNotFoundError as e:
            print(f"Warning: Could not find data file for step {global_step}: {e}")
            continue
        except Exception as e:
            print(f"Error processing step {global_step}: {e}")
            continue
    
    print(f"\nProcessing complete! Results saved to {output_dir}")
    print(f"Data files: {data_output_dir}")
    if visualize:
        print(f"Visualizations: {vis_output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and save 3-modal LMD features')
    
    # Model and data paths
    parser.add_argument('--checkpoint', type=str,
                       default=DEFAULT_CKPT_DIR,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str,
                       default=DEFAULT_OUTPUT_DIR,
                       help='Output directory for results')
    parser.add_argument('--data_root', type=str,
                       default=DEFAULT_DATA_ROOT,
                       help='Root with batch_1_validation_data and batch_1_validation_data_lidar')
    
    # Processing settings
    parser.add_argument('--start_step', type=int, default=1, 
                       help='Starting step for processing')
    parser.add_argument('--end_step', type=int, default=6019, 
                       help='Ending step for processing')
    parser.add_argument('--step_interval', type=int, default=50, 
                       help='Interval between processed steps')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Create visualizations')
    
    # Model settings
    parser.add_argument('--device_ids', type=int, nargs='+', default=[0],
                       help='GPU device IDs to use')
    
    args = parser.parse_args()
    
    main(
        init_dir=args.checkpoint,
        output_dir=args.output_dir,
        data_root=args.data_root,
        start_step=args.start_step,
        end_step=args.end_step,
        step_interval=args.step_interval,
        visualize=args.visualize,
        device_ids=args.device_ids
    )
