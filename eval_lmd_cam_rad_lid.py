import os
import sys

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
    os.path.join(SCRIPT_DIR, 'output_eval_pkl_variants'),
)

import time
import pickle
import random

import numpy as np
import torch
from fire import Fire

import saverloader
from nets.segnet_relu_feats import Segnet_relu_feats
from nets.segnet_relu_feats_bias_three import Segnet_relu_feats_bias as Segnet_relu_feats_bias_three
import utils.vox

torch.multiprocessing.set_sharing_strategy('file_system')
random.seed(125)
np.random.seed(125)

scene_centroid_x = 0.0
scene_centroid_y = 1.0
scene_centroid_z = 0.0

scene_centroid_py = np.array([scene_centroid_x,
                              scene_centroid_y,
                              scene_centroid_z]).reshape([1, 3])
scene_centroid = torch.from_numpy(scene_centroid_py).float()

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


def calc_metric(global_step, mask_model, fusion_model, rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util, in_occ_mem0, lid_occ_mem0, act_mask, norm_mask, device, seg_bev_g, valid_bev_g, data_root):
    fusion_model.eval()
    
    with torch.no_grad():
        _, _, seg_bev_e, _, out_dict = fusion_model(
            rgb_camXs=rgb_camXs,
            pix_T_cams=pix_T_cams,
            cam0_T_camXs=cam0_T_camXs,
            vox_util=vox_util,
            rad_occ_mem0=[lid_occ_mem0, in_occ_mem0],
            act_mask=act_mask,
            norm_mask=norm_mask,
        )

    final_eval_metrics = {}
    metrics = {}
    seg_bev_e_round = torch.sigmoid(seg_bev_e).round()
    intersection = (seg_bev_e_round*seg_bev_g*valid_bev_g).sum()
    union = ((seg_bev_e_round+seg_bev_g)*valid_bev_g).clamp(0,1).sum()
    metrics['intersection'] = intersection.item()
    metrics['union'] = union.item()
    
    # Metrics for camera+radar perturbation -> lidar prediction
    final_eval_metrics['cam_rad_perturb_lid_correlation'] = []
    final_eval_metrics['cam_rad_perturb_lid_cosine_similarity'] = []
    final_eval_metrics['cam_rad_perturb_lid_mse'] = []
    
    # Metrics for camera+radar perturbation -> camera-to-camera
    final_eval_metrics['cam_rad_perturb_cam_correlation'] = []
    final_eval_metrics['cam_rad_perturb_cam_cosine_similarity'] = []
    final_eval_metrics['cam_rad_perturb_cam_mse'] = []
    
    # Metrics for camera+radar perturbation -> radar-to-radar
    final_eval_metrics['cam_rad_perturb_rad_correlation'] = []
    final_eval_metrics['cam_rad_perturb_rad_cosine_similarity'] = []
    final_eval_metrics['cam_rad_perturb_rad_mse'] = []
    
    # Metrics for camera+lidar perturbation -> radar prediction
    final_eval_metrics['cam_lid_perturb_rad_correlation'] = []
    final_eval_metrics['cam_lid_perturb_rad_cosine_similarity'] = []
    final_eval_metrics['cam_lid_perturb_rad_mse'] = []
    
    # Metrics for camera+lidar perturbation -> camera-to-camera
    final_eval_metrics['cam_lid_perturb_cam_correlation'] = []
    final_eval_metrics['cam_lid_perturb_cam_cosine_similarity'] = []
    final_eval_metrics['cam_lid_perturb_cam_mse'] = []
    
    # Metrics for camera+lidar perturbation -> lidar-to-lidar
    final_eval_metrics['cam_lid_perturb_lid_correlation'] = []
    final_eval_metrics['cam_lid_perturb_lid_cosine_similarity'] = []
    final_eval_metrics['cam_lid_perturb_lid_mse'] = []
    
    # Metrics for radar+lidar perturbation -> camera prediction
    final_eval_metrics['rad_lid_perturb_cam_correlation'] = []
    final_eval_metrics['rad_lid_perturb_cam_cosine_similarity'] = []
    final_eval_metrics['rad_lid_perturb_cam_mse'] = []
    
    # Metrics for radar+lidar perturbation -> radar-to-radar
    final_eval_metrics['rad_lid_perturb_rad_correlation'] = []
    final_eval_metrics['rad_lid_perturb_rad_cosine_similarity'] = []
    final_eval_metrics['rad_lid_perturb_rad_mse'] = []
    
    # Metrics for radar+lidar perturbation -> lidar-to-lidar
    final_eval_metrics['rad_lid_perturb_lid_correlation'] = []
    final_eval_metrics['rad_lid_perturb_lid_cosine_similarity'] = []
    final_eval_metrics['rad_lid_perturb_lid_mse'] = []
    
    iter_start_time = time.time()
    #print("here it is : ", global_step_inner)
    
    with torch.no_grad():
        # to store atomic evaluation values
        eval_metrics = {}
    
        for i in range(1, 3): 
            eval_check_left = global_step - 2000 * i
            eval_check_right = global_step + 2000 * i
            if (eval_check_left > 0) and (eval_check_left <= 6019):
                # Load camera/radar data from original path
                camera_radar_data = torch.load(
                    _cam_radar_path(data_root, eval_check_left),
                    map_location=torch.device(device),
                    weights_only=False,
                )
                ptb_rgb_camXs, ptb_pix_T_cams, ptb_cam0_T_camXs, vox_util, ptb_in_occ_mem0 = camera_radar_data[:5]
                # Load lidar data from separate path
                _, _, _, _, ptb_lid_occ_mem0, _, _ = torch.load(
                    _lidar_path(data_root, eval_check_left),
                    map_location=torch.device(device),
                    weights_only=False,
                )
                # Experiment 1: camera+radar perturbation -> measure lidar prediction
                act_mask_cam_rad, norm_mask_cam_rad, _, _, _ = mask_model(
                    rgb_camXs=ptb_rgb_camXs,
                    pix_T_cams=ptb_pix_T_cams,
                    cam0_T_camXs=ptb_cam0_T_camXs,
                    vox_util=vox_util,
                    rad_occ_mem0=[lid_occ_mem0, ptb_in_occ_mem0],
                )

                _, _, _, _, out_dict_cam_rad = fusion_model(
                    rgb_camXs=ptb_rgb_camXs,
                    pix_T_cams=ptb_pix_T_cams,
                    cam0_T_camXs=ptb_cam0_T_camXs,
                    vox_util=vox_util,
                    rad_occ_mem0=[lid_occ_mem0, ptb_in_occ_mem0],
                    act_mask=act_mask,
                    norm_mask=norm_mask,
                )
                                
                # Experiment 2: camera+lidar perturbation -> measure radar prediction
                act_mask_cam_lid, norm_mask_cam_lid, _, _, _ = mask_model(
                    rgb_camXs=ptb_rgb_camXs,
                    pix_T_cams=ptb_pix_T_cams,
                    cam0_T_camXs=ptb_cam0_T_camXs,
                    vox_util=vox_util,
                    rad_occ_mem0=in_occ_mem0,
                    lid_occ_mem0=ptb_lid_occ_mem0,
                )

                _, _, _, _, out_dict_cam_lid = fusion_model(
                    rgb_camXs=ptb_rgb_camXs,
                    pix_T_cams=ptb_pix_T_cams,
                    cam0_T_camXs=ptb_cam0_T_camXs,
                    vox_util=vox_util,
                    rad_occ_mem0=[ptb_lid_occ_mem0, in_occ_mem0],
                    act_mask=act_mask,
                    norm_mask=norm_mask,
                )
                                
                # Experiment 3: radar+lidar perturbation -> measure camera prediction
                act_mask_rad_lid, norm_mask_rad_lid, _, _, _ = mask_model(
                    rgb_camXs=rgb_camXs,
                    pix_T_cams=pix_T_cams,
                    cam0_T_camXs=cam0_T_camXs,
                    vox_util=vox_util,
                    rad_occ_mem0=[ptb_lid_occ_mem0, ptb_in_occ_mem0],
                )

                _, _, _, _, out_dict_rad_lid = fusion_model(
                    rgb_camXs=rgb_camXs,
                    pix_T_cams=pix_T_cams,
                    cam0_T_camXs=cam0_T_camXs,
                    vox_util=vox_util,
                    rad_occ_mem0=[ptb_lid_occ_mem0, ptb_in_occ_mem0],
                    act_mask=act_mask,
                    norm_mask=norm_mask,
                )
                    
                eval_metrics = collect_evaluation_three_modality(out_dict, out_dict_cam_rad, out_dict_cam_lid, out_dict_rad_lid, eval_metrics)
                final_eval_metrics['cam_rad_perturb_lid_correlation'].append(eval_metrics['cam_rad_perturb_lid_correlation'])
                final_eval_metrics['cam_rad_perturb_lid_cosine_similarity'].append(eval_metrics['cam_rad_perturb_lid_cosine_similarity'])
                final_eval_metrics['cam_rad_perturb_lid_mse'].append(eval_metrics['cam_rad_perturb_lid_mse'])
                
                final_eval_metrics['cam_rad_perturb_cam_correlation'].append(eval_metrics['cam_rad_perturb_cam_correlation'])
                final_eval_metrics['cam_rad_perturb_cam_cosine_similarity'].append(eval_metrics['cam_rad_perturb_cam_cosine_similarity'])
                final_eval_metrics['cam_rad_perturb_cam_mse'].append(eval_metrics['cam_rad_perturb_cam_mse'])
                
                final_eval_metrics['cam_rad_perturb_rad_correlation'].append(eval_metrics['cam_rad_perturb_rad_correlation'])
                final_eval_metrics['cam_rad_perturb_rad_cosine_similarity'].append(eval_metrics['cam_rad_perturb_rad_cosine_similarity'])
                final_eval_metrics['cam_rad_perturb_rad_mse'].append(eval_metrics['cam_rad_perturb_rad_mse'])
                
                final_eval_metrics['cam_lid_perturb_rad_correlation'].append(eval_metrics['cam_lid_perturb_rad_correlation'])
                final_eval_metrics['cam_lid_perturb_rad_cosine_similarity'].append(eval_metrics['cam_lid_perturb_rad_cosine_similarity'])
                final_eval_metrics['cam_lid_perturb_rad_mse'].append(eval_metrics['cam_lid_perturb_rad_mse'])
                
                final_eval_metrics['cam_lid_perturb_cam_correlation'].append(eval_metrics['cam_lid_perturb_cam_correlation'])
                final_eval_metrics['cam_lid_perturb_cam_cosine_similarity'].append(eval_metrics['cam_lid_perturb_cam_cosine_similarity'])
                final_eval_metrics['cam_lid_perturb_cam_mse'].append(eval_metrics['cam_lid_perturb_cam_mse'])
                
                final_eval_metrics['cam_lid_perturb_lid_correlation'].append(eval_metrics['cam_lid_perturb_lid_correlation'])
                final_eval_metrics['cam_lid_perturb_lid_cosine_similarity'].append(eval_metrics['cam_lid_perturb_lid_cosine_similarity'])
                final_eval_metrics['cam_lid_perturb_lid_mse'].append(eval_metrics['cam_lid_perturb_lid_mse'])
                
                final_eval_metrics['rad_lid_perturb_cam_correlation'].append(eval_metrics['rad_lid_perturb_cam_correlation'])
                final_eval_metrics['rad_lid_perturb_cam_cosine_similarity'].append(eval_metrics['rad_lid_perturb_cam_cosine_similarity'])
                final_eval_metrics['rad_lid_perturb_cam_mse'].append(eval_metrics['rad_lid_perturb_cam_mse'])
                
                final_eval_metrics['rad_lid_perturb_rad_correlation'].append(eval_metrics['rad_lid_perturb_rad_correlation'])
                final_eval_metrics['rad_lid_perturb_rad_cosine_similarity'].append(eval_metrics['rad_lid_perturb_rad_cosine_similarity'])
                final_eval_metrics['rad_lid_perturb_rad_mse'].append(eval_metrics['rad_lid_perturb_rad_mse'])
                
                final_eval_metrics['rad_lid_perturb_lid_correlation'].append(eval_metrics['rad_lid_perturb_lid_correlation'])
                final_eval_metrics['rad_lid_perturb_lid_cosine_similarity'].append(eval_metrics['rad_lid_perturb_lid_cosine_similarity'])
                final_eval_metrics['rad_lid_perturb_lid_mse'].append(eval_metrics['rad_lid_perturb_lid_mse'])

            
            elif (eval_check_right > 0) and (eval_check_right <= 6019) : 
                # Load camera/radar data from original path
                camera_radar_data = torch.load(
                    _cam_radar_path(data_root, eval_check_right),
                    map_location=torch.device(device),
                    weights_only=False,
                )
                ptb_rgb_camXs, ptb_pix_T_cams, ptb_cam0_T_camXs, vox_util, ptb_in_occ_mem0 = camera_radar_data[:5]
                # Load lidar data from separate path
                _, _, _, _, ptb_lid_occ_mem0, _, _ = torch.load(
                    _lidar_path(data_root, eval_check_right),
                    map_location=torch.device(device),
                    weights_only=False,
                )
                # Experiment 1: camera+radar perturbation -> measure lidar prediction
                act_mask_cam_rad, norm_mask_cam_rad, _, _, _ = mask_model(
                    rgb_camXs=ptb_rgb_camXs,
                    pix_T_cams=ptb_pix_T_cams,
                    cam0_T_camXs=ptb_cam0_T_camXs,
                    vox_util=vox_util,
                    rad_occ_mem0=[lid_occ_mem0, ptb_in_occ_mem0],
                )

                _, _, _, _, out_dict_cam_rad = fusion_model(
                    rgb_camXs=ptb_rgb_camXs,
                    pix_T_cams=ptb_pix_T_cams,
                    cam0_T_camXs=ptb_cam0_T_camXs,
                    vox_util=vox_util,
                    rad_occ_mem0=[lid_occ_mem0, ptb_in_occ_mem0],
                    act_mask=act_mask,
                    norm_mask=norm_mask,
                )
                                
                # Experiment 2: camera+lidar perturbation -> measure radar prediction
                act_mask_cam_lid, norm_mask_cam_lid, _, _, _ = mask_model(
                    rgb_camXs=ptb_rgb_camXs,
                    pix_T_cams=ptb_pix_T_cams,
                    cam0_T_camXs=ptb_cam0_T_camXs,
                    vox_util=vox_util,
                    rad_occ_mem0=[ptb_lid_occ_mem0, in_occ_mem0],
                )

                _, _, _, _, out_dict_cam_lid = fusion_model(
                    rgb_camXs=ptb_rgb_camXs,
                    pix_T_cams=ptb_pix_T_cams,
                    cam0_T_camXs=ptb_cam0_T_camXs,
                    vox_util=vox_util,
                    rad_occ_mem0=[ptb_lid_occ_mem0, in_occ_mem0],
                    act_mask=act_mask,
                    norm_mask=norm_mask,
                )
                                
                # Experiment 3: radar+lidar perturbation -> measure camera prediction
                act_mask_rad_lid, norm_mask_rad_lid, _, _, _ = mask_model(
                    rgb_camXs=rgb_camXs,
                    pix_T_cams=pix_T_cams,
                    cam0_T_camXs=cam0_T_camXs,
                    vox_util=vox_util,
                    rad_occ_mem0=[ptb_lid_occ_mem0, ptb_in_occ_mem0],
                )

                _, _, _, _, out_dict_rad_lid = fusion_model(
                    rgb_camXs=rgb_camXs,
                    pix_T_cams=pix_T_cams,
                    cam0_T_camXs=cam0_T_camXs,
                    vox_util=vox_util,
                    rad_occ_mem0=[ptb_lid_occ_mem0, ptb_in_occ_mem0],
                    act_mask=act_mask,
                    norm_mask=norm_mask,
                )

                eval_metrics = collect_evaluation_three_modality(out_dict, out_dict_cam_rad, out_dict_cam_lid, out_dict_rad_lid, eval_metrics)
                final_eval_metrics['cam_rad_perturb_lid_correlation'].append(eval_metrics['cam_rad_perturb_lid_correlation'])
                final_eval_metrics['cam_rad_perturb_lid_cosine_similarity'].append(eval_metrics['cam_rad_perturb_lid_cosine_similarity'])
                final_eval_metrics['cam_rad_perturb_lid_mse'].append(eval_metrics['cam_rad_perturb_lid_mse'])
                
                final_eval_metrics['cam_rad_perturb_cam_correlation'].append(eval_metrics['cam_rad_perturb_cam_correlation'])
                final_eval_metrics['cam_rad_perturb_cam_cosine_similarity'].append(eval_metrics['cam_rad_perturb_cam_cosine_similarity'])
                final_eval_metrics['cam_rad_perturb_cam_mse'].append(eval_metrics['cam_rad_perturb_cam_mse'])
                
                final_eval_metrics['cam_rad_perturb_rad_correlation'].append(eval_metrics['cam_rad_perturb_rad_correlation'])
                final_eval_metrics['cam_rad_perturb_rad_cosine_similarity'].append(eval_metrics['cam_rad_perturb_rad_cosine_similarity'])
                final_eval_metrics['cam_rad_perturb_rad_mse'].append(eval_metrics['cam_rad_perturb_rad_mse'])
                
                final_eval_metrics['cam_lid_perturb_rad_correlation'].append(eval_metrics['cam_lid_perturb_rad_correlation'])
                final_eval_metrics['cam_lid_perturb_rad_cosine_similarity'].append(eval_metrics['cam_lid_perturb_rad_cosine_similarity'])
                final_eval_metrics['cam_lid_perturb_rad_mse'].append(eval_metrics['cam_lid_perturb_rad_mse'])
                
                final_eval_metrics['cam_lid_perturb_cam_correlation'].append(eval_metrics['cam_lid_perturb_cam_correlation'])
                final_eval_metrics['cam_lid_perturb_cam_cosine_similarity'].append(eval_metrics['cam_lid_perturb_cam_cosine_similarity'])
                final_eval_metrics['cam_lid_perturb_cam_mse'].append(eval_metrics['cam_lid_perturb_cam_mse'])
                
                final_eval_metrics['cam_lid_perturb_lid_correlation'].append(eval_metrics['cam_lid_perturb_lid_correlation'])
                final_eval_metrics['cam_lid_perturb_lid_cosine_similarity'].append(eval_metrics['cam_lid_perturb_lid_cosine_similarity'])
                final_eval_metrics['cam_lid_perturb_lid_mse'].append(eval_metrics['cam_lid_perturb_lid_mse'])
                
                final_eval_metrics['rad_lid_perturb_cam_correlation'].append(eval_metrics['rad_lid_perturb_cam_correlation'])
                final_eval_metrics['rad_lid_perturb_cam_cosine_similarity'].append(eval_metrics['rad_lid_perturb_cam_cosine_similarity'])
                final_eval_metrics['rad_lid_perturb_cam_mse'].append(eval_metrics['rad_lid_perturb_cam_mse'])
                
                final_eval_metrics['rad_lid_perturb_rad_correlation'].append(eval_metrics['rad_lid_perturb_rad_correlation'])
                final_eval_metrics['rad_lid_perturb_rad_cosine_similarity'].append(eval_metrics['rad_lid_perturb_rad_cosine_similarity'])
                final_eval_metrics['rad_lid_perturb_rad_mse'].append(eval_metrics['rad_lid_perturb_rad_mse'])
                
                final_eval_metrics['rad_lid_perturb_lid_correlation'].append(eval_metrics['rad_lid_perturb_lid_correlation'])
                final_eval_metrics['rad_lid_perturb_lid_cosine_similarity'].append(eval_metrics['rad_lid_perturb_lid_cosine_similarity'])
                final_eval_metrics['rad_lid_perturb_lid_mse'].append(eval_metrics['rad_lid_perturb_lid_mse'])
                

    iter_time = time.time()-iter_start_time

    print("For data num %d, total eval time is %.2f"% (global_step, iter_time))
    # print('%s; step %06d/%d; rtime %.2f; itime %.2f (%.2f ms); loss %.5f; iou_ev %.1f' % (
    #     model_name, global_step, max_iters, read_time, iter_time, 1000*time_pool_ev.mean(),
    #     total_loss.item(), 100*intersection/union))
  
    return metrics, final_eval_metrics

    
def main(
        exp_name='eval',
        batch_size=1,
        init_dir=DEFAULT_CKPT_DIR,
        data_root=DEFAULT_DATA_ROOT,
        output_dir=DEFAULT_OUTPUT_DIR,
        ignore_load=None,
        # model
        encoder_type='res101',
        use_radar=True,
        use_lidar=True,
        use_metaradar=True,
        do_rgbcompress=True,
        # cuda
        device_ids=[1],
):
    B = batch_size
    max_iters = 6019
    global_step = 0
    # file_name = "evaluation_data_without_sigmoid_uniform_ratio_l2_test%d.pkl" % global_step  # 파일 이름을 원하는 대로 지정
    # ratio rule - normalization rule
    os.makedirs(output_dir, exist_ok=True)
    file_name = os.path.join(
        output_dir,
        "three_modality_perturbation_uniform_identity%d.pkl" % global_step,
    )

    #pdb.set_trace()
    assert(B % len(device_ids) == 0) # batch size must be divisible by number of gpus

    device = 'cuda:%d' % device_ids[0]
    
    ## autogen a name
    model_name = "%s" % init_dir.split('/')[-1]
    model_name += "_%d" % B
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    #print('model_name', model_name)

    vox_util = utils.vox.Vox_util(
        Z, Y, X,
        scene_centroid=scene_centroid.to(device),
        bounds=bounds,
        assert_cube=False)
    # max_iters = 4500
    #len(val_dataloader) # determine iters by length of dataset

    # pdb.set_trace()
    # set up model & seg loss
    model = Segnet_relu_feats(Z, Y, X, use_radar=use_radar, use_lidar=use_lidar, use_metaradar=use_metaradar, do_rgbcompress=do_rgbcompress, encoder_type=encoder_type)
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    parameters = list(model.parameters())
    # load checkpoint
    _ = saverloader.load(init_dir, model.module, ignore_load=ignore_load)
    requires_grad(parameters, False)
    
    fusion_model = Segnet_relu_feats_bias_three(Z, Y, X, use_radar=use_radar, use_lidar=use_lidar, use_metaradar=use_metaradar, do_rgbcompress=do_rgbcompress, encoder_type=encoder_type, rand_flip=False)
    fusion_model = fusion_model.to(device)
    fusion_model = torch.nn.DataParallel(fusion_model, device_ids=device_ids)
    _ = saverloader.load(init_dir, fusion_model.module, ignore_load=ignore_load)
    parameters = list(fusion_model.parameters())
    requires_grad(parameters, False)
    
    fusion_model.eval()
    model.eval()

    final_eval_bins = {}
    # Metrics for camera+radar perturbation -> lidar prediction
    final_eval_bins['cam_rad_perturb_lid_correlation'] = []
    final_eval_bins['cam_rad_perturb_lid_cosine_similarity'] = []
    final_eval_bins['cam_rad_perturb_lid_mse'] = []
    
    # Metrics for camera+radar perturbation -> camera-to-camera
    final_eval_bins['cam_rad_perturb_cam_correlation'] = []
    final_eval_bins['cam_rad_perturb_cam_cosine_similarity'] = []
    final_eval_bins['cam_rad_perturb_cam_mse'] = []
    
    # Metrics for camera+radar perturbation -> radar-to-radar
    final_eval_bins['cam_rad_perturb_rad_correlation'] = []
    final_eval_bins['cam_rad_perturb_rad_cosine_similarity'] = []
    final_eval_bins['cam_rad_perturb_rad_mse'] = []
    
    # Metrics for camera+lidar perturbation -> radar prediction
    final_eval_bins['cam_lid_perturb_rad_correlation'] = []
    final_eval_bins['cam_lid_perturb_rad_cosine_similarity'] = []
    final_eval_bins['cam_lid_perturb_rad_mse'] = []
    
    # Metrics for camera+lidar perturbation -> camera-to-camera
    final_eval_bins['cam_lid_perturb_cam_correlation'] = []
    final_eval_bins['cam_lid_perturb_cam_cosine_similarity'] = []
    final_eval_bins['cam_lid_perturb_cam_mse'] = []
    
    # Metrics for camera+lidar perturbation -> lidar-to-lidar
    final_eval_bins['cam_lid_perturb_lid_correlation'] = []
    final_eval_bins['cam_lid_perturb_lid_cosine_similarity'] = []
    final_eval_bins['cam_lid_perturb_lid_mse'] = []
    
    # Metrics for radar+lidar perturbation -> camera prediction
    final_eval_bins['rad_lid_perturb_cam_correlation'] = []
    final_eval_bins['rad_lid_perturb_cam_cosine_similarity'] = []
    final_eval_bins['rad_lid_perturb_cam_mse'] = []
    
    # Metrics for radar+lidar perturbation -> radar-to-radar
    final_eval_bins['rad_lid_perturb_rad_correlation'] = []
    final_eval_bins['rad_lid_perturb_rad_cosine_similarity'] = []
    final_eval_bins['rad_lid_perturb_rad_mse'] = []
    
    # Metrics for radar+lidar perturbation -> lidar-to-lidar
    final_eval_bins['rad_lid_perturb_lid_correlation'] = []
    final_eval_bins['rad_lid_perturb_lid_cosine_similarity'] = []
    final_eval_bins['rad_lid_perturb_lid_mse'] = []
    
    intersection = 0
    union = 0
    
    while global_step < max_iters:
        global_step += 1
        iter_start_time = time.time()
        read_start_time = time.time()
                           
        read_time = time.time()-read_start_time
        # Load camera/radar data from original path
        camera_radar_data = torch.load(
            _cam_radar_path(data_root, global_step),
            map_location=torch.device(device),
            weights_only=False,
        )
        rgb_camXs, pix_T_cams, cam0_T_camXs, _, in_occ_mem0, seg_bev_g, valid_bev_g = camera_radar_data
        # Load lidar data from separate path
        _, _, _, _, lid_occ_mem0, _, _ = torch.load(
            _lidar_path(data_root, global_step),
            map_location=torch.device(device),
            weights_only=False,
        )
        
        with torch.no_grad():
            act_feat, norm_feat, _, _, _ = model(
                    rgb_camXs=rgb_camXs,
                    pix_T_cams=pix_T_cams,
                    cam0_T_camXs=cam0_T_camXs,
                    vox_util=vox_util,
                    rad_occ_mem0=[lid_occ_mem0, in_occ_mem0])

        # pdb.set_trace()
        metrics, final_eval_metrics = calc_metric(
            global_step,
            model,
            fusion_model,
            rgb_camXs,
            pix_T_cams,
            cam0_T_camXs,
            vox_util,
            in_occ_mem0,
            lid_occ_mem0,
            act_feat,
            norm_feat,
            device,
            seg_bev_g,
            valid_bev_g,
            data_root,
        )

        intersection += metrics['intersection']
        union += metrics['union']

        #seg_bev_e_round = torch.sigmoid(seg_bev_e).round()
        final_eval_bins['cam_rad_perturb_lid_correlation'].append(sum(final_eval_metrics["cam_rad_perturb_lid_correlation"]) / len(final_eval_metrics["cam_rad_perturb_lid_correlation"]))
        final_eval_bins["cam_rad_perturb_lid_cosine_similarity"].append(sum(final_eval_metrics["cam_rad_perturb_lid_cosine_similarity"]) / len(final_eval_metrics["cam_rad_perturb_lid_cosine_similarity"]))
        final_eval_bins["cam_rad_perturb_lid_mse"].append(sum(final_eval_metrics["cam_rad_perturb_lid_mse"]) / len(final_eval_metrics["cam_rad_perturb_lid_mse"]))
        
        final_eval_bins["cam_rad_perturb_cam_correlation"].append(sum(final_eval_metrics["cam_rad_perturb_cam_correlation"]) / len(final_eval_metrics["cam_rad_perturb_cam_correlation"]))
        final_eval_bins["cam_rad_perturb_cam_cosine_similarity"].append(sum(final_eval_metrics["cam_rad_perturb_cam_cosine_similarity"]) / len(final_eval_metrics["cam_rad_perturb_cam_cosine_similarity"]))
        final_eval_bins["cam_rad_perturb_cam_mse"].append(sum(final_eval_metrics["cam_rad_perturb_cam_mse"]) / len(final_eval_metrics["cam_rad_perturb_cam_mse"]))
        
        final_eval_bins["cam_rad_perturb_rad_correlation"].append(sum(final_eval_metrics["cam_rad_perturb_rad_correlation"]) / len(final_eval_metrics["cam_rad_perturb_rad_correlation"]))
        final_eval_bins["cam_rad_perturb_rad_cosine_similarity"].append(sum(final_eval_metrics["cam_rad_perturb_rad_cosine_similarity"]) / len(final_eval_metrics["cam_rad_perturb_rad_cosine_similarity"]))
        final_eval_bins["cam_rad_perturb_rad_mse"].append(sum(final_eval_metrics["cam_rad_perturb_rad_mse"]) / len(final_eval_metrics["cam_rad_perturb_rad_mse"]))

        final_eval_bins["cam_lid_perturb_rad_correlation"].append(sum(final_eval_metrics["cam_lid_perturb_rad_correlation"]) / len(final_eval_metrics["cam_lid_perturb_rad_correlation"]))
        final_eval_bins["cam_lid_perturb_rad_cosine_similarity"].append(sum(final_eval_metrics["cam_lid_perturb_rad_cosine_similarity"]) / len(final_eval_metrics["cam_lid_perturb_rad_cosine_similarity"]))
        final_eval_bins["cam_lid_perturb_rad_mse"].append(sum(final_eval_metrics["cam_lid_perturb_rad_mse"]) / len(final_eval_metrics["cam_lid_perturb_rad_mse"]))
        
        final_eval_bins["cam_lid_perturb_cam_correlation"].append(sum(final_eval_metrics["cam_lid_perturb_cam_correlation"]) / len(final_eval_metrics["cam_lid_perturb_cam_correlation"]))
        final_eval_bins["cam_lid_perturb_cam_cosine_similarity"].append(sum(final_eval_metrics["cam_lid_perturb_cam_cosine_similarity"]) / len(final_eval_metrics["cam_lid_perturb_cam_cosine_similarity"]))
        final_eval_bins["cam_lid_perturb_cam_mse"].append(sum(final_eval_metrics["cam_lid_perturb_cam_mse"]) / len(final_eval_metrics["cam_lid_perturb_cam_mse"]))
        
        final_eval_bins["cam_lid_perturb_lid_correlation"].append(sum(final_eval_metrics["cam_lid_perturb_lid_correlation"]) / len(final_eval_metrics["cam_lid_perturb_lid_correlation"]))
        final_eval_bins["cam_lid_perturb_lid_cosine_similarity"].append(sum(final_eval_metrics["cam_lid_perturb_lid_cosine_similarity"]) / len(final_eval_metrics["cam_lid_perturb_lid_cosine_similarity"]))
        final_eval_bins["cam_lid_perturb_lid_mse"].append(sum(final_eval_metrics["cam_lid_perturb_lid_mse"]) / len(final_eval_metrics["cam_lid_perturb_lid_mse"]))

        final_eval_bins["rad_lid_perturb_cam_correlation"].append(sum(final_eval_metrics["rad_lid_perturb_cam_correlation"]) / len(final_eval_metrics["rad_lid_perturb_cam_correlation"]))
        final_eval_bins["rad_lid_perturb_cam_cosine_similarity"].append(sum(final_eval_metrics["rad_lid_perturb_cam_cosine_similarity"]) / len(final_eval_metrics["rad_lid_perturb_cam_cosine_similarity"]))
        final_eval_bins["rad_lid_perturb_cam_mse"].append(sum(final_eval_metrics["rad_lid_perturb_cam_mse"]) / len(final_eval_metrics["rad_lid_perturb_cam_mse"]))
        
        final_eval_bins["rad_lid_perturb_rad_correlation"].append(sum(final_eval_metrics["rad_lid_perturb_rad_correlation"]) / len(final_eval_metrics["rad_lid_perturb_rad_correlation"]))
        final_eval_bins["rad_lid_perturb_rad_cosine_similarity"].append(sum(final_eval_metrics["rad_lid_perturb_rad_cosine_similarity"]) / len(final_eval_metrics["rad_lid_perturb_rad_cosine_similarity"]))
        final_eval_bins["rad_lid_perturb_rad_mse"].append(sum(final_eval_metrics["rad_lid_perturb_rad_mse"]) / len(final_eval_metrics["rad_lid_perturb_rad_mse"]))
        
        final_eval_bins["rad_lid_perturb_lid_correlation"].append(sum(final_eval_metrics["rad_lid_perturb_lid_correlation"]) / len(final_eval_metrics["rad_lid_perturb_lid_correlation"]))
        final_eval_bins["rad_lid_perturb_lid_cosine_similarity"].append(sum(final_eval_metrics["rad_lid_perturb_lid_cosine_similarity"]) / len(final_eval_metrics["rad_lid_perturb_lid_cosine_similarity"]))
        final_eval_bins["rad_lid_perturb_lid_mse"].append(sum(final_eval_metrics["rad_lid_perturb_lid_mse"]) / len(final_eval_metrics["rad_lid_perturb_lid_mse"]))
    
    
        iter_time = time.time() - iter_start_time
        print('\n%s; step %06d/%d; rtime %.2f; itime %.2f; iou : %.4f' % (
            model_name, global_step, max_iters, read_time, iter_time, 100*intersection/union))
        print('cam+rad->lid: corr %.4f; cos %.4f; mse %.4f' %(
            final_eval_bins['cam_rad_perturb_lid_correlation'][-1], 
            final_eval_bins['cam_rad_perturb_lid_cosine_similarity'][-1],
            final_eval_bins['cam_rad_perturb_lid_mse'][-1]
        ))
        print('cam+rad->cam: corr %.4f; cos %.4f; mse %.4f' %(
            final_eval_bins['cam_rad_perturb_cam_correlation'][-1], 
            final_eval_bins['cam_rad_perturb_cam_cosine_similarity'][-1],
            final_eval_bins['cam_rad_perturb_cam_mse'][-1]
        ))
        print('cam+rad->rad: corr %.4f; cos %.4f; mse %.4f' %(
            final_eval_bins['cam_rad_perturb_rad_correlation'][-1], 
            final_eval_bins['cam_rad_perturb_rad_cosine_similarity'][-1],
            final_eval_bins['cam_rad_perturb_rad_mse'][-1]
        ))
        print('cam+lid->rad: corr %.4f; cos %.4f; mse %.4f' %(
            final_eval_bins['cam_lid_perturb_rad_correlation'][-1], 
            final_eval_bins['cam_lid_perturb_rad_cosine_similarity'][-1],
            final_eval_bins['cam_lid_perturb_rad_mse'][-1]
        ))
        print('cam+lid->cam: corr %.4f; cos %.4f; mse %.4f' %(
            final_eval_bins['cam_lid_perturb_cam_correlation'][-1], 
            final_eval_bins['cam_lid_perturb_cam_cosine_similarity'][-1],
            final_eval_bins['cam_lid_perturb_cam_mse'][-1]
        ))
        print('cam+lid->lid: corr %.4f; cos %.4f; mse %.4f' %(
            final_eval_bins['cam_lid_perturb_lid_correlation'][-1], 
            final_eval_bins['cam_lid_perturb_lid_cosine_similarity'][-1],
            final_eval_bins['cam_lid_perturb_lid_mse'][-1]
        ))
        print('rad+lid->cam: corr %.4f; cos %.4f; mse %.4f' %(
            final_eval_bins['rad_lid_perturb_cam_correlation'][-1], 
            final_eval_bins['rad_lid_perturb_cam_cosine_similarity'][-1],
            final_eval_bins['rad_lid_perturb_cam_mse'][-1]
        ))
        print('rad+lid->rad: corr %.4f; cos %.4f; mse %.4f' %(
            final_eval_bins['rad_lid_perturb_rad_correlation'][-1], 
            final_eval_bins['rad_lid_perturb_rad_cosine_similarity'][-1],
            final_eval_bins['rad_lid_perturb_rad_mse'][-1]
        ))
        print('rad+lid->lid: corr %.4f; cos %.4f; mse %.4f' %(
            final_eval_bins['rad_lid_perturb_lid_correlation'][-1], 
            final_eval_bins['rad_lid_perturb_lid_cosine_similarity'][-1],
            final_eval_bins['rad_lid_perturb_lid_mse'][-1]
        ))
        
        if global_step % 300 == 0 :
            # file_name = "evaluation_data_without_sigmoid_uniform_ratio_l2_test%d.pkl" % global_step  # 파일 이름을 원하는 대로 지정
            print("saving checkpoints...")
            with open(file_name, 'wb') as file:
                pickle.dump(final_eval_bins, file)


    # file_name = "evaluation_data_without_sigmoid_uniform_ratio_l2_test%d.pkl" % global_step  # 파일 이름을 원하는 대로 지정
    with open(file_name, 'wb') as file:
        pickle.dump(final_eval_bins, file)


def collect_evaluation_three_modality(out_dict, out_dict_cam_rad, out_dict_cam_lid, out_dict_rad_lid, eval_metrics):
    
    # Camera+Radar perturbation -> Lidar prediction metrics
    eval_metrics["cam_rad_perturb_lid_correlation"] = calculating_metrics((out_dict["segmentation_lid"]), (out_dict_cam_rad["segmentation_lid"]), "correlation")
    eval_metrics["cam_rad_perturb_lid_cosine_similarity"] = calculating_metrics((out_dict["segmentation_lid"]), (out_dict_cam_rad["segmentation_lid"]), "cosine_similarity")
    eval_metrics["cam_rad_perturb_lid_mse"] = calculating_metrics((out_dict["segmentation_lid"]), (out_dict_cam_rad["segmentation_lid"]), "MSE")
    
    # Camera+Radar perturbation -> Camera-to-Camera metrics
    eval_metrics["cam_rad_perturb_cam_correlation"] = calculating_metrics((out_dict["segmentation_cam"]), (out_dict_cam_rad["segmentation_cam"]), "correlation")
    eval_metrics["cam_rad_perturb_cam_cosine_similarity"] = calculating_metrics((out_dict["segmentation_cam"]), (out_dict_cam_rad["segmentation_cam"]), "cosine_similarity")
    eval_metrics["cam_rad_perturb_cam_mse"] = calculating_metrics((out_dict["segmentation_cam"]), (out_dict_cam_rad["segmentation_cam"]), "MSE")
    
    # Camera+Radar perturbation -> Radar-to-Radar metrics
    eval_metrics["cam_rad_perturb_rad_correlation"] = calculating_metrics((out_dict["segmentation_rad"]), (out_dict_cam_rad["segmentation_rad"]), "correlation")
    eval_metrics["cam_rad_perturb_rad_cosine_similarity"] = calculating_metrics((out_dict["segmentation_rad"]), (out_dict_cam_rad["segmentation_rad"]), "cosine_similarity")
    eval_metrics["cam_rad_perturb_rad_mse"] = calculating_metrics((out_dict["segmentation_rad"]), (out_dict_cam_rad["segmentation_rad"]), "MSE")
    
    # Camera+Lidar perturbation -> Radar prediction metrics
    eval_metrics["cam_lid_perturb_rad_correlation"] = calculating_metrics((out_dict["segmentation_rad"]), (out_dict_cam_lid["segmentation_rad"]), "correlation")
    eval_metrics["cam_lid_perturb_rad_cosine_similarity"] = calculating_metrics((out_dict["segmentation_rad"]), (out_dict_cam_lid["segmentation_rad"]), "cosine_similarity")
    eval_metrics["cam_lid_perturb_rad_mse"] = calculating_metrics((out_dict["segmentation_rad"]), (out_dict_cam_lid["segmentation_rad"]), "MSE")
    
    # Camera+Lidar perturbation -> Camera-to-Camera metrics
    eval_metrics["cam_lid_perturb_cam_correlation"] = calculating_metrics((out_dict["segmentation_cam"]), (out_dict_cam_lid["segmentation_cam"]), "correlation")
    eval_metrics["cam_lid_perturb_cam_cosine_similarity"] = calculating_metrics((out_dict["segmentation_cam"]), (out_dict_cam_lid["segmentation_cam"]), "cosine_similarity")
    eval_metrics["cam_lid_perturb_cam_mse"] = calculating_metrics((out_dict["segmentation_cam"]), (out_dict_cam_lid["segmentation_cam"]), "MSE")
    
    # Camera+Lidar perturbation -> Lidar-to-Lidar metrics
    eval_metrics["cam_lid_perturb_lid_correlation"] = calculating_metrics((out_dict["segmentation_lid"]), (out_dict_cam_lid["segmentation_lid"]), "correlation")
    eval_metrics["cam_lid_perturb_lid_cosine_similarity"] = calculating_metrics((out_dict["segmentation_lid"]), (out_dict_cam_lid["segmentation_lid"]), "cosine_similarity")
    eval_metrics["cam_lid_perturb_lid_mse"] = calculating_metrics((out_dict["segmentation_lid"]), (out_dict_cam_lid["segmentation_lid"]), "MSE")
    
    # Radar+Lidar perturbation -> Camera prediction metrics
    eval_metrics["rad_lid_perturb_cam_correlation"] = calculating_metrics((out_dict["segmentation_cam"]), (out_dict_rad_lid["segmentation_cam"]), "correlation")
    eval_metrics["rad_lid_perturb_cam_cosine_similarity"] = calculating_metrics((out_dict["segmentation_cam"]), (out_dict_rad_lid["segmentation_cam"]), "cosine_similarity")
    eval_metrics["rad_lid_perturb_cam_mse"] = calculating_metrics((out_dict["segmentation_cam"]), (out_dict_rad_lid["segmentation_cam"]), "MSE")
    
    # Radar+Lidar perturbation -> Radar-to-Radar metrics
    eval_metrics["rad_lid_perturb_rad_correlation"] = calculating_metrics((out_dict["segmentation_rad"]), (out_dict_rad_lid["segmentation_rad"]), "correlation")
    eval_metrics["rad_lid_perturb_rad_cosine_similarity"] = calculating_metrics((out_dict["segmentation_rad"]), (out_dict_rad_lid["segmentation_rad"]), "cosine_similarity")
    eval_metrics["rad_lid_perturb_rad_mse"] = calculating_metrics((out_dict["segmentation_rad"]), (out_dict_rad_lid["segmentation_rad"]), "MSE")
    
    # Radar+Lidar perturbation -> Lidar-to-Lidar metrics
    eval_metrics["rad_lid_perturb_lid_correlation"] = calculating_metrics((out_dict["segmentation_lid"]), (out_dict_rad_lid["segmentation_lid"]), "correlation")
    eval_metrics["rad_lid_perturb_lid_cosine_similarity"] = calculating_metrics((out_dict["segmentation_lid"]), (out_dict_rad_lid["segmentation_lid"]), "cosine_similarity")
    eval_metrics["rad_lid_perturb_lid_mse"] = calculating_metrics((out_dict["segmentation_lid"]), (out_dict_rad_lid["segmentation_lid"]), "MSE")
    
    return eval_metrics
    

def calculating_metrics(data_1, data_2, name="correlation"):
    
    if name == "correlation" : 
        mean_1 = data_1.mean()
        # variance_1 = data_1.var(unbiased=False)
        mean_2 = data_2.mean()
        # variance_2 = data_2.var(unbiased=False) 
        numerator = torch.sum((data_1 - mean_1) * (data_2 - mean_2))
        denominator = torch.sqrt(torch.sum((data_1 - mean_1) ** 2)) * torch.sqrt(torch.sum((data_2 - mean_2) ** 2))
        correlation = numerator / denominator
        return correlation
    
    elif name == "cosine_similarity" : 
        dot_product = torch.dot((data_1).flatten(), (data_2).flatten())
        norm1 = torch.norm(data_1)
        norm2 = torch.norm(data_2)
        cosine_similarity = dot_product / (norm1 * norm2)
        return cosine_similarity
    
    elif name == "MSE" : 
        mse = torch.mean((data_1 - data_2) ** 2)
        return mse
        
if __name__ == '__main__':
    Fire(main)
