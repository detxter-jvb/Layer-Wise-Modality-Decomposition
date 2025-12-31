import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, 'project')
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

DEFAULT_DATA_ROOT = os.environ.get('LMD_DATA_ROOT', os.path.join(SCRIPT_DIR, 'data'))
DEFAULT_CKPT_DIR = os.environ.get(
    'LMD_CKPT_DIR',
    os.path.join(SCRIPT_DIR, 'checkpoints', '8x5_5e-4_rad25_18_55_34'),
)

import time
import numpy as np
import saverloader
from fire import Fire
from nets.segnet_relu import Segnet_relu
from nets.segnet_relu_feats_bias import Segnet_relu_feats_bias
import pickle
import utils.vox
import random
import torch
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


def _data_path(data_root, step):
    return os.path.join(data_root, "batch_1_validation_data", f"target_data_0{step:4d}.pt")


def calc_metric(global_step, mask_model, fusion_model, rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util, in_occ_mem0, act_mask, norm_mask, device, seg_bev_g, valid_bev_g, data_root):
    fusion_model.eval()
    with torch.no_grad():
        _, _, seg_bev_e, _, out_dict = fusion_model(
                    rgb_camXs=rgb_camXs,
                    pix_T_cams=pix_T_cams,
                    cam0_T_camXs=cam0_T_camXs,
                    vox_util=vox_util,
                    rad_occ_mem0=in_occ_mem0, 
                    act_mask=act_mask, 
                    norm_mask=norm_mask)

    final_eval_metrics = {}
    metrics = {}
    
    seg_bev_e_round = torch.sigmoid(seg_bev_e).round()
    intersection = (seg_bev_e_round*seg_bev_g*valid_bev_g).sum()
    union = ((seg_bev_e_round+seg_bev_g)*valid_bev_g).clamp(0,1).sum()
    metrics['intersection'] = intersection.item()
    metrics['union'] = union.item()
    
    final_eval_metrics['rad_perturb_rad_correlation'] = []
    final_eval_metrics['rad_perturb_cam_correlation'] = []
    final_eval_metrics['cam_perturb_rad_correlation'] = []
    final_eval_metrics['cam_perturb_cam_correlation'] = []
    final_eval_metrics['rad_perturb_rad_cosine_similarity'] = []
    final_eval_metrics['rad_perturb_cam_cosine_similarity'] = []
    final_eval_metrics['cam_perturb_rad_cosine_similarity'] = []
    final_eval_metrics['cam_perturb_cam_cosine_similarity'] = []
    final_eval_metrics['rad_perturb_rad_l2_norm'] = []
    final_eval_metrics['rad_perturb_cam_l2_norm'] = []
    final_eval_metrics['cam_perturb_rad_l2_norm'] = []
    final_eval_metrics['cam_perturb_cam_l2_norm'] = []
    final_eval_metrics['rad_perturb_rad_IOU'] = []
    final_eval_metrics['rad_perturb_cam_IOU'] = []
    final_eval_metrics['cam_perturb_rad_IOU'] = []
    final_eval_metrics['cam_perturb_cam_IOU'] = []
    final_eval_metrics['rad_perturb_rad_MSE'] = []
    final_eval_metrics['rad_perturb_cam_MSE'] = []
    final_eval_metrics['cam_perturb_rad_MSE'] = []
    final_eval_metrics['cam_perturb_cam_MSE'] = []
    
    iter_start_time = time.time()
    with torch.no_grad():
        # to store atomic evaluation values
        eval_metrics = {}
    
        for i in range(1, 11): 
            eval_check_left = global_step - 500 * i
            eval_check_right = global_step + 500 * i
            if (eval_check_left > 0) and (eval_check_left <= 6019):
                ptb_rgb_camXs, ptb_pix_T_cams, ptb_cam0_T_camXs, vox_util, ptb_in_occ_mem0, _, _ = torch.load(
                    _data_path(data_root, eval_check_left),
                    map_location=torch.device(device),
                    weights_only=False,
                )
                with torch.no_grad():
                    # radar's perturbation
                    _, _, _, _, out_dict_2 = fusion_model(
                                rgb_camXs=rgb_camXs,
                                pix_T_cams=pix_T_cams,
                                cam0_T_camXs=cam0_T_camXs,
                                vox_util=vox_util,
                                rad_occ_mem0= ptb_in_occ_mem0,
                                act_mask=act_mask, 
                                norm_mask=norm_mask)
                                # radar must be inconsistent / camera must be consistent
                with torch.no_grad():
                    # camera's perturbation
                    _, _, _, _, out_dict_3 = fusion_model(
                                rgb_camXs=ptb_rgb_camXs,
                                pix_T_cams=ptb_pix_T_cams,
                                cam0_T_camXs=ptb_cam0_T_camXs,
                                vox_util=vox_util,
                                rad_occ_mem0=in_occ_mem0, 
                                act_mask=act_mask, 
                                norm_mask=norm_mask)
                eval_metrics = collect_evaluation(out_dict, out_dict_2, out_dict_3, eval_metrics)
                final_eval_metrics['rad_perturb_rad_correlation'].append(eval_metrics['rad_perturb_rad_correlation'])
                final_eval_metrics['rad_perturb_cam_correlation'].append(eval_metrics['rad_perturb_cam_correlation'])
                final_eval_metrics['rad_perturb_rad_cosine_similarity'].append(eval_metrics['rad_perturb_rad_cosine_similarity'])
                final_eval_metrics['rad_perturb_cam_cosine_similarity'].append(eval_metrics['rad_perturb_cam_cosine_similarity'])
                final_eval_metrics['cam_perturb_rad_correlation'].append(eval_metrics['cam_perturb_rad_correlation'])
                final_eval_metrics['cam_perturb_cam_correlation'].append(eval_metrics['cam_perturb_cam_correlation'])
                final_eval_metrics['cam_perturb_rad_cosine_similarity'].append(eval_metrics['cam_perturb_rad_cosine_similarity'])
                final_eval_metrics['cam_perturb_cam_cosine_similarity'].append(eval_metrics['cam_perturb_cam_cosine_similarity'])
                final_eval_metrics['rad_perturb_rad_l2_norm'].append(eval_metrics['rad_perturb_rad_l2_norm'])
                final_eval_metrics['rad_perturb_cam_l2_norm'].append(eval_metrics['rad_perturb_cam_l2_norm'])
                final_eval_metrics['cam_perturb_rad_l2_norm'].append(eval_metrics['cam_perturb_rad_l2_norm'])
                final_eval_metrics['cam_perturb_cam_l2_norm'].append(eval_metrics['cam_perturb_cam_l2_norm'])
                final_eval_metrics['rad_perturb_rad_IOU'].append(eval_metrics['rad_perturb_rad_IOU'])
                final_eval_metrics['rad_perturb_cam_IOU'].append(eval_metrics['rad_perturb_cam_IOU'])
                final_eval_metrics['cam_perturb_rad_IOU'].append(eval_metrics['cam_perturb_rad_IOU'])
                final_eval_metrics['cam_perturb_cam_IOU'].append(eval_metrics['cam_perturb_cam_IOU'])
                final_eval_metrics['rad_perturb_rad_MSE'].append(eval_metrics['rad_perturb_rad_MSE'])
                final_eval_metrics['rad_perturb_cam_MSE'].append(eval_metrics['rad_perturb_cam_MSE'])
                final_eval_metrics['cam_perturb_rad_MSE'].append(eval_metrics['cam_perturb_rad_MSE'])
                final_eval_metrics['cam_perturb_cam_MSE'].append(eval_metrics['cam_perturb_cam_MSE'])

            # radar must be inconsistent (hugely assisted by camera) / camera must be consistent    
            elif (eval_check_right > 0) and (eval_check_right <= 6019) : 
                ptb_rgb_camXs, ptb_pix_T_cams, ptb_cam0_T_camXs, vox_util, ptb_in_occ_mem0, _, _ = torch.load(
                    _data_path(data_root, eval_check_right),
                    map_location=torch.device(device),
                    weights_only=False,
                )
                with torch.no_grad():
                    # radar's perturbation
                    _, _, _, _, out_dict_2 = fusion_model(
                                rgb_camXs=rgb_camXs,
                                pix_T_cams=pix_T_cams,
                                cam0_T_camXs=cam0_T_camXs,
                                vox_util=vox_util,
                                rad_occ_mem0= ptb_in_occ_mem0, 
                                act_mask=act_mask,
                                norm_mask=norm_mask)
                                # radar must be inconsistent / camera must be consistent
                with torch.no_grad():
                    # camera's perturbation
                    _, _, _, _, out_dict_3 = fusion_model(
                                rgb_camXs=ptb_rgb_camXs,
                                pix_T_cams=ptb_pix_T_cams,
                                cam0_T_camXs=ptb_cam0_T_camXs,
                                vox_util=vox_util,
                                rad_occ_mem0=in_occ_mem0, 
                                act_mask=act_mask, 
                                norm_mask=norm_mask)

                eval_metrics = collect_evaluation(out_dict, out_dict_2, out_dict_3, eval_metrics)
                final_eval_metrics['rad_perturb_rad_correlation'].append(eval_metrics['rad_perturb_rad_correlation'])
                final_eval_metrics['rad_perturb_cam_correlation'].append(eval_metrics['rad_perturb_cam_correlation'])
                final_eval_metrics['rad_perturb_rad_cosine_similarity'].append(eval_metrics['rad_perturb_rad_cosine_similarity'])
                final_eval_metrics['rad_perturb_cam_cosine_similarity'].append(eval_metrics['rad_perturb_cam_cosine_similarity'])
                final_eval_metrics['cam_perturb_rad_correlation'].append(eval_metrics['cam_perturb_rad_correlation'])
                final_eval_metrics['cam_perturb_cam_correlation'].append(eval_metrics['cam_perturb_cam_correlation'])
                final_eval_metrics['cam_perturb_rad_cosine_similarity'].append(eval_metrics['cam_perturb_rad_cosine_similarity'])
                final_eval_metrics['cam_perturb_cam_cosine_similarity'].append(eval_metrics['cam_perturb_cam_cosine_similarity'])           
                final_eval_metrics['rad_perturb_rad_l2_norm'].append(eval_metrics['rad_perturb_rad_l2_norm'])
                final_eval_metrics['rad_perturb_cam_l2_norm'].append(eval_metrics['rad_perturb_cam_l2_norm'])
                final_eval_metrics['cam_perturb_rad_l2_norm'].append(eval_metrics['cam_perturb_rad_l2_norm'])
                final_eval_metrics['cam_perturb_cam_l2_norm'].append(eval_metrics['cam_perturb_cam_l2_norm'])
                final_eval_metrics['rad_perturb_rad_IOU'].append(eval_metrics['rad_perturb_rad_IOU'])
                final_eval_metrics['rad_perturb_cam_IOU'].append(eval_metrics['rad_perturb_cam_IOU'])
                final_eval_metrics['cam_perturb_rad_IOU'].append(eval_metrics['cam_perturb_rad_IOU'])
                final_eval_metrics['cam_perturb_cam_IOU'].append(eval_metrics['cam_perturb_cam_IOU'])
                final_eval_metrics['rad_perturb_rad_MSE'].append(eval_metrics['rad_perturb_rad_MSE'])
                final_eval_metrics['rad_perturb_cam_MSE'].append(eval_metrics['rad_perturb_cam_MSE'])
                final_eval_metrics['cam_perturb_rad_MSE'].append(eval_metrics['cam_perturb_rad_MSE'])
                final_eval_metrics['cam_perturb_cam_MSE'].append(eval_metrics['cam_perturb_cam_MSE'])                

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
        ignore_load=None,
        encoder_type='res101',
        use_radar=True,
        use_lidar=False,
        use_metaradar=True,
        do_rgbcompress=True,
        data_root=DEFAULT_DATA_ROOT,
        device_ids=[3],
):
    B = batch_size
    max_iters = 6019
    global_step = 4500
    # file_name = "evaluation_data_without_sigmoid_uniform_ratio_l2_test%d.pkl" % global_step  # 파일 이름을 원하는 대로 지정
    # ratio rule - normalization rule
    file_name = "activation_ratio_bn_identity_in_ratio_2_radar_final_ver_dh2%d.pkl" % global_step

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
    
    # set up model & seg loss
    model = Segnet_relu(Z, Y, X, use_radar=use_radar, use_lidar=use_lidar, use_metaradar=use_metaradar, do_rgbcompress=do_rgbcompress, encoder_type=encoder_type)
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    parameters = list(model.parameters())
    # load checkpoint
    _ = saverloader.load(init_dir, model.module, ignore_load=ignore_load)
    requires_grad(parameters, False)
    
    fusion_model = Segnet_relu_feats_bias(Z, Y, X, use_radar=use_radar, use_lidar=use_lidar, use_metaradar=use_metaradar, do_rgbcompress=do_rgbcompress, encoder_type=encoder_type, rand_flip=False)
    #fusion_model = Segnet_relu_feats(Z, Y, X, use_radar=use_radar, use_lidar=use_lidar, use_metaradar=use_metaradar, do_rgbcompress=do_rgbcompress, encoder_type=encoder_type, rand_flip=False, act_mask=act_mask, norm_mask=norm_mask)
    #model = Segnet_relu_feats(Z, Y, X, use_radar=use_radar, use_lidar=use_lidar, use_metaradar=use_metaradar, do_rgbcompress=do_rgbcompress, encoder_type=encoder_type, rand_flip=False, act_mask=act_feat, norm_mask=norm_feat)
    fusion_model = fusion_model.to(device)
    fusion_model = torch.nn.DataParallel(fusion_model, device_ids=device_ids)
    _ = saverloader.load(init_dir, fusion_model.module, ignore_load=ignore_load)
    parameters = list(fusion_model.parameters())
    requires_grad(parameters, False)
    
    fusion_model.eval()
    model.eval()

    metrics = {}
    final_eval_bins = {}
    final_eval_bins['rad_perturb_rad_correlation'] = []
    final_eval_bins["rad_perturb_cam_correlation"] = []
    final_eval_bins["rad_perturb_rad_cosine_similarity"] = []
    final_eval_bins["rad_perturb_cam_cosine_similarity"] = []

    final_eval_bins["cam_perturb_rad_correlation"] = []
    final_eval_bins["cam_perturb_cam_correlation"] = []
    final_eval_bins["cam_perturb_rad_cosine_similarity"] = []
    final_eval_bins["cam_perturb_cam_cosine_similarity"] = []

    final_eval_bins["rad_perturb_rad_l2_norm"] = []
    final_eval_bins["rad_perturb_cam_l2_norm"] = []
    final_eval_bins["cam_perturb_rad_l2_norm"] = []
    final_eval_bins["cam_perturb_cam_l2_norm"] = []

    final_eval_bins["rad_perturb_rad_IOU"] = []
    final_eval_bins["rad_perturb_cam_IOU"] = []
    final_eval_bins["cam_perturb_rad_IOU"] = []
    final_eval_bins["cam_perturb_cam_IOU"] = []
    
    final_eval_bins["rad_perturb_rad_MSE"] = []
    final_eval_bins["rad_perturb_cam_MSE"] = []
    final_eval_bins["cam_perturb_rad_MSE"] = []
    final_eval_bins["cam_perturb_cam_MSE"] = []
    
    intersection = 0
    union = 0
    
    while global_step < max_iters:
        global_step += 1
        iter_start_time = time.time()
        read_start_time = time.time()
                           
        read_time = time.time()-read_start_time
        rgb_camXs, pix_T_cams, cam0_T_camXs, _, in_occ_mem0, seg_bev_g, valid_bev_g = torch.load(
            _data_path(data_root, global_step),
            map_location=torch.device(device),
            weights_only=False,
        )
        
        with torch.no_grad():
            act_feat, norm_feat, _, _, _ = model(
                    rgb_camXs=rgb_camXs,
                    pix_T_cams=pix_T_cams,
                    cam0_T_camXs=cam0_T_camXs,
                    vox_util=vox_util,
                    rad_occ_mem0=in_occ_mem0)

        metrics, final_eval_metrics = calc_metric(
            global_step, model, fusion_model, rgb_camXs, pix_T_cams, cam0_T_camXs,
            vox_util, in_occ_mem0, act_feat, norm_feat, device, seg_bev_g, valid_bev_g, data_root
        )
        intersection += metrics['intersection']
        union += metrics['union']

        #seg_bev_e_round = torch.sigmoid(seg_bev_e).round()
        final_eval_bins['rad_perturb_rad_correlation'].append(sum(final_eval_metrics["rad_perturb_rad_correlation"]) / len(final_eval_metrics["rad_perturb_rad_correlation"]))
        final_eval_bins["rad_perturb_cam_correlation"].append(sum(final_eval_metrics["rad_perturb_cam_correlation"]) / len(final_eval_metrics["rad_perturb_cam_correlation"]))
        final_eval_bins["rad_perturb_rad_cosine_similarity"].append(sum(final_eval_metrics["rad_perturb_rad_cosine_similarity"]) / len(final_eval_metrics["rad_perturb_rad_cosine_similarity"]))
        final_eval_bins["rad_perturb_cam_cosine_similarity"].append(sum(final_eval_metrics["rad_perturb_cam_cosine_similarity"]) / len(final_eval_metrics["rad_perturb_cam_cosine_similarity"]))

        final_eval_bins["cam_perturb_rad_correlation"].append(sum(final_eval_metrics["cam_perturb_rad_correlation"]) / len(final_eval_metrics["cam_perturb_rad_correlation"]))
        final_eval_bins["cam_perturb_cam_correlation"].append(sum(final_eval_metrics["cam_perturb_cam_correlation"]) / len(final_eval_metrics["cam_perturb_cam_correlation"]))
        final_eval_bins["cam_perturb_rad_cosine_similarity"].append(sum(final_eval_metrics["cam_perturb_rad_cosine_similarity"]) / len(final_eval_metrics["cam_perturb_rad_cosine_similarity"]))
        final_eval_bins["cam_perturb_cam_cosine_similarity"].append(sum(final_eval_metrics["cam_perturb_cam_cosine_similarity"]) / len(final_eval_metrics["cam_perturb_cam_cosine_similarity"]))

        final_eval_bins["rad_perturb_rad_l2_norm"].append(sum(final_eval_metrics["rad_perturb_rad_l2_norm"]) / len(final_eval_metrics["rad_perturb_rad_l2_norm"]))
        final_eval_bins["rad_perturb_cam_l2_norm"].append(sum(final_eval_metrics["rad_perturb_cam_l2_norm"]) / len(final_eval_metrics["rad_perturb_cam_l2_norm"]))
        final_eval_bins["cam_perturb_rad_l2_norm"].append(sum(final_eval_metrics["cam_perturb_rad_l2_norm"]) / len(final_eval_metrics["cam_perturb_rad_l2_norm"]))
        final_eval_bins["cam_perturb_cam_l2_norm"].append(sum(final_eval_metrics["cam_perturb_cam_l2_norm"]) / len(final_eval_metrics["cam_perturb_cam_l2_norm"]))
    
        final_eval_bins["rad_perturb_rad_IOU"].append(sum(final_eval_metrics["rad_perturb_rad_IOU"]) / len(final_eval_metrics["rad_perturb_rad_IOU"]))
        final_eval_bins["rad_perturb_cam_IOU"].append(sum(final_eval_metrics["rad_perturb_cam_IOU"]) / len(final_eval_metrics["rad_perturb_cam_IOU"]))
        final_eval_bins["cam_perturb_rad_IOU"].append(sum(final_eval_metrics["cam_perturb_rad_IOU"]) / len(final_eval_metrics["cam_perturb_rad_IOU"]))
        final_eval_bins["cam_perturb_cam_IOU"].append(sum(final_eval_metrics["cam_perturb_cam_IOU"]) / len(final_eval_metrics["cam_perturb_cam_IOU"]))
        
        final_eval_bins["rad_perturb_rad_MSE"].append(sum(final_eval_metrics["rad_perturb_rad_MSE"]) / len(final_eval_metrics["rad_perturb_rad_MSE"]))
        final_eval_bins["rad_perturb_cam_MSE"].append(sum(final_eval_metrics["rad_perturb_cam_MSE"]) / len(final_eval_metrics["rad_perturb_cam_MSE"]))
        final_eval_bins["cam_perturb_rad_MSE"].append(sum(final_eval_metrics["cam_perturb_rad_MSE"]) / len(final_eval_metrics["cam_perturb_rad_MSE"]))
        final_eval_bins["cam_perturb_cam_MSE"].append(sum(final_eval_metrics["cam_perturb_cam_MSE"]) / len(final_eval_metrics["cam_perturb_cam_MSE"]))


        iter_time = time.time() - iter_start_time
        print('\n%s; step %06d/%d; rtime %.2f; itime %.2f; iou : %.4f' % (
            model_name, global_step, max_iters, read_time, iter_time, 100*intersection/union))
        print('rad_rad_corr %.6f; rad_cam_corr %.6f; rad_rad_cos %.6f; rad_cam_cos %.6f; cam_rad_corr %.6f; cam_cam_corr %.6f; cam_rad_cos %.6f; cam_cam_cos %.6f' %(
            final_eval_bins['rad_perturb_rad_correlation'][-1], final_eval_bins['rad_perturb_cam_correlation'][-1], final_eval_bins['rad_perturb_rad_cosine_similarity'][-1], final_eval_bins['rad_perturb_cam_cosine_similarity'][-1],
            final_eval_bins['cam_perturb_rad_correlation'][-1], final_eval_bins['cam_perturb_cam_correlation'][-1], final_eval_bins['cam_perturb_rad_cosine_similarity'][-1], final_eval_bins['cam_perturb_cam_cosine_similarity'][-1]
        ))
        print('rad_rad_l2 %.6f; rad_cam_l2 %.6f; cam_rad_l2 %.6f; cam_cam_l2 %.6f' %(
            final_eval_bins['rad_perturb_rad_l2_norm'][-1], final_eval_bins['rad_perturb_cam_l2_norm'][-1], final_eval_bins['cam_perturb_rad_l2_norm'][-1], final_eval_bins['cam_perturb_cam_l2_norm'][-1]
        ))
        print('rad_rad_IOU %.6f; rad_cam_IOU %.6f; cam_rad_IOU %.6f; cam_cam_IOU %.6f' %(
            final_eval_bins['rad_perturb_rad_IOU'][-1], final_eval_bins['rad_perturb_cam_IOU'][-1], final_eval_bins['cam_perturb_rad_IOU'][-1], final_eval_bins['cam_perturb_cam_IOU'][-1]
        ))
        print('rad_rad_MSE %.6f; rad_cam_MSE %.6f; cam_rad_MSE %.6f; cam_cam_MSE %.6f' %(
            final_eval_bins['rad_perturb_rad_MSE'][-1], final_eval_bins['rad_perturb_cam_MSE'][-1], final_eval_bins['cam_perturb_rad_MSE'][-1], final_eval_bins['cam_perturb_cam_MSE'][-1]
        ))
        if global_step % 300 == 0 :
            # file_name = "evaluation_data_without_sigmoid_uniform_ratio_l2_test%d.pkl" % global_step  # 파일 이름을 원하는 대로 지정
            print("saving checkpoints...")
            with open(file_name, 'wb') as file:
                pickle.dump(final_eval_bins, file)


    # file_name = "evaluation_data_without_sigmoid_uniform_ratio_l2_test%d.pkl" % global_step  # 파일 이름을 원하는 대로 지정
    with open(file_name, 'wb') as file:
        pickle.dump(final_eval_bins, file)


def collect_evaluation(out_dict, out_dict_2, out_dict_3, eval_metrics):
    
    eval_metrics["rad_perturb_rad_correlation"] = calculating_metrics((out_dict["segmentation_rad"]), (out_dict_2["segmentation_rad"]), "correlation")
    eval_metrics["rad_perturb_cam_correlation"] = calculating_metrics((out_dict["segmentation_cam"]), (out_dict_2["segmentation_cam"]), "correlation")
    eval_metrics["rad_perturb_rad_cosine_similarity"] = calculating_metrics((out_dict["segmentation_rad"]), (out_dict_2["segmentation_rad"]), "cosine_similarity")
    eval_metrics["rad_perturb_cam_cosine_similarity"] = calculating_metrics((out_dict["segmentation_cam"]), (out_dict_2["segmentation_cam"]), "cosine_similarity")
    
    eval_metrics["cam_perturb_rad_correlation"] = calculating_metrics((out_dict["segmentation_rad"]), (out_dict_3["segmentation_rad"]), "correlation")
    eval_metrics["cam_perturb_cam_correlation"] = calculating_metrics((out_dict["segmentation_cam"]), (out_dict_3["segmentation_cam"]), "correlation")
    eval_metrics["cam_perturb_rad_cosine_similarity"] = calculating_metrics((out_dict["segmentation_rad"]), (out_dict_3["segmentation_rad"]), "cosine_similarity")
    eval_metrics["cam_perturb_cam_cosine_similarity"] = calculating_metrics((out_dict["segmentation_cam"]), (out_dict_3["segmentation_cam"]), "cosine_similarity")
    
    eval_metrics["rad_perturb_rad_l2_norm"] = calculating_metrics((out_dict["segmentation_rad"]), (out_dict_2["segmentation_rad"]), "l2_norm")
    eval_metrics["rad_perturb_cam_l2_norm"] = calculating_metrics((out_dict["segmentation_cam"]), (out_dict_2["segmentation_cam"]), "l2_norm")
    eval_metrics["cam_perturb_rad_l2_norm"] = calculating_metrics((out_dict["segmentation_rad"]), (out_dict_3["segmentation_rad"]), "l2_norm")
    eval_metrics["cam_perturb_cam_l2_norm"] = calculating_metrics((out_dict["segmentation_cam"]), (out_dict_3["segmentation_cam"]), "l2_norm")

    eval_metrics["rad_perturb_rad_IOU"] = calculating_metrics((out_dict["segmentation_rad"]), (out_dict_2["segmentation_rad"]), "IOU")
    eval_metrics["rad_perturb_cam_IOU"] = calculating_metrics((out_dict["segmentation_cam"]), (out_dict_2["segmentation_cam"]), "IOU")
    eval_metrics["cam_perturb_rad_IOU"] = calculating_metrics((out_dict["segmentation_rad"]), (out_dict_3["segmentation_rad"]), "IOU")
    eval_metrics["cam_perturb_cam_IOU"] = calculating_metrics((out_dict["segmentation_cam"]), (out_dict_3["segmentation_cam"]), "IOU")

    eval_metrics["rad_perturb_rad_MSE"] = calculating_metrics((out_dict["segmentation_rad"]), (out_dict_2["segmentation_rad"]), "MSE")
    eval_metrics["rad_perturb_cam_MSE"] = calculating_metrics((out_dict["segmentation_cam"]), (out_dict_2["segmentation_cam"]), "MSE")
    eval_metrics["cam_perturb_rad_MSE"] = calculating_metrics((out_dict["segmentation_rad"]), (out_dict_3["segmentation_rad"]), "MSE")
    eval_metrics["cam_perturb_cam_MSE"] = calculating_metrics((out_dict["segmentation_cam"]), (out_dict_3["segmentation_cam"]), "MSE")

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
    
    elif name == "l2_norm" : 
        l2_norm = torch.norm(data_1 - data_2, p=2)
        return l2_norm

    elif name == "IOU" : 
        data_1_seg = torch.sigmoid(data_1).round()
        data_2_seg = torch.sigmoid(data_2).round()
        intersection_seg = (data_1_seg * data_2_seg).sum()
        union_seg = ((data_1_seg + data_2_seg)).clamp(0,1).sum()
        iou_seg = intersection_seg / union_seg
        return iou_seg
    
    elif name == "MSE" :
        mse = torch.mean((data_1 - data_2) ** 2)
        return mse

    elif name == "NRMSE_stdev":
        rmse = torch.sqrt(torch.mean((data_1 - data_2) ** 2))
        y_std = torch.std(data_2)
        nrmse = rmse / (y_std + 1e-8)  # 0으로 나누기 방지
        return nrmse

    elif name == "NRMSE_range":
        rmse = torch.sqrt(torch.mean((data_1 - data_2) ** 2))
        y_range = data_1.max() - data_1.min()
        nrmse = rmse / (y_range + 1e-8)  # 0으로 나누기 방지
        return nrmse
        
        
if __name__ == '__main__':
    Fire(main)
