import os
import sys
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, 'project')
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

DEFAULT_NUSCENES_DIR = os.environ.get('NUSCENES_DATA_DIR', os.path.join(SCRIPT_DIR, 'nuscenes'))

import numpy as np
from fire import Fire
import torch

import nuscenesdataset
import utils.basic
import utils.geom
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


def save_batch(d, device, global_step, save_radar, save_lidar,
               radar_out_dir, lidar_out_dir, use_metaradar):
    (
        imgs, rots, trans, intrins,
        _lidar0_data, _lidar0_extra, lidar_data, _lidar_extra,
        _lrtlist, _vislist, _tidlist, _scorelist,
        seg_bev_g, valid_bev_g, _center_bev_g, _offset_bev_g,
        radar_data, _egopose
    ) = d

    _, T, _, _, _, _ = imgs.shape
    assert T == 1

    imgs = imgs[:, 0]
    rots = rots[:, 0]
    trans = trans[:, 0]
    intrins = intrins[:, 0]
    seg_bev_g = seg_bev_g[:, 0]
    valid_bev_g = valid_bev_g[:, 0]

    rgb_camXs = imgs.float().to(device)
    rgb_camXs = rgb_camXs - 0.5

    seg_bev_g = seg_bev_g.to(device)
    valid_bev_g = valid_bev_g.to(device)

    B = rgb_camXs.shape[0]
    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)

    intrins_ = __p(intrins)
    pix_T_cams_ = utils.geom.merge_intrinsics(*utils.geom.split_intrinsics(intrins_)).to(device)
    pix_T_cams = __u(pix_T_cams_)

    velo_T_cams = utils.geom.merge_rtlist(rots, trans).to(device)
    cams_T_velo = __u(utils.geom.safe_inverse(__p(velo_T_cams)))
    cam0_T_camXs = utils.geom.get_camM_T_camXs(velo_T_cams, ind=0)

    vox_util = utils.vox.Vox_util(
        Z, Y, X,
        scene_centroid=scene_centroid.to(device),
        bounds=bounds,
        assert_cube=False)

    if save_lidar:
        lidar_data = lidar_data[:, 0]
        xyz_velo0 = lidar_data.to(device).permute(0, 2, 1)
        mag = torch.norm(xyz_velo0, dim=2)
        xyz_velo0 = xyz_velo0[:, mag[0] > 1]
        xyz_cam0 = utils.geom.apply_4x4(cams_T_velo[:, 0], xyz_velo0)
        occ_mem0 = vox_util.voxelize_xyz(xyz_cam0, Z, Y, X, assert_cube=False)

        os.makedirs(lidar_out_dir, exist_ok=True)
        data_name = os.path.join(lidar_out_dir, "target_data_0%4d.pt" % global_step)
        data_tuple = (rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util, occ_mem0, seg_bev_g, valid_bev_g)
        torch.save(data_tuple, data_name)

    if save_radar:
        radar_data = radar_data[:, 0]
        rad_data = radar_data.to(device).permute(0, 2, 1)
        xyz_rad = rad_data[:, :, :3]
        meta_rad = rad_data[:, :, 3:]

        rad_xyz_cam0 = utils.geom.apply_4x4(cams_T_velo[:, 0], xyz_rad)
        rad_occ_mem0 = vox_util.voxelize_xyz(rad_xyz_cam0, Z, Y, X, assert_cube=False)
        if use_metaradar:
            radar_in_occ_mem0 = vox_util.voxelize_xyz_and_feats(
                rad_xyz_cam0, meta_rad, Z, Y, X, assert_cube=False)
        else:
            radar_in_occ_mem0 = rad_occ_mem0

        os.makedirs(radar_out_dir, exist_ok=True)
        data_name = os.path.join(radar_out_dir, "target_data_0%4d.pt" % global_step)
        data_tuple = (rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util, radar_in_occ_mem0, seg_bev_g, valid_bev_g)
        torch.save(data_tuple, data_name)

    if global_step % 50 == 0:
        print("this is step number : ", global_step)


def main(
        dset='trainval',  # we will just use val
        shuffle=False,
        batch_size=8,
        nworkers=12,
        data_dir=DEFAULT_NUSCENES_DIR,
        res_scale=2,
        ncams=6,
        nsweeps=5,
        use_radar_filters=False,
        use_metaradar=True,
        save_mode='radar',  # radar, lidar, both
        radar_out_dir='batch_1_validation_data_radar',
        lidar_out_dir='batch_1_validation_data_lidar',
        device_ids=[4, 5, 6, 7],
):
    if isinstance(device_ids, int):
        device_ids = [device_ids]
    device = 'cpu' if not device_ids else 'cuda:%d' % device_ids[0]

    final_dim = (int(224 * res_scale), int(400 * res_scale))
    data_aug_conf = {
        'final_dim': final_dim,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'ncams': ncams,
    }
    _, val_dataloader = nuscenesdataset.compile_data(
        dset,
        data_dir,
        data_aug_conf=data_aug_conf,
        centroid=scene_centroid_py,
        bounds=bounds,
        res_3d=(Z, Y, X),
        bsz=batch_size,
        nworkers=1,
        nworkers_val=nworkers,
        shuffle=shuffle,
        use_radar_filters=use_radar_filters,
        seqlen=1,
        nsweeps=nsweeps,
        do_shuffle_cams=False,
        get_tids=True,
    )

    save_mode = save_mode.lower()
    if save_mode not in ['radar', 'lidar', 'both']:
        raise ValueError("save_mode must be one of: 'radar', 'lidar', 'both'")
    save_radar = save_mode in ['radar', 'both']
    save_lidar = save_mode in ['lidar', 'both']

    for global_step, sample in enumerate(val_dataloader, start=1):
        with torch.no_grad():
            save_batch(
                sample,
                device,
                global_step,
                save_radar,
                save_lidar,
                radar_out_dir,
                lidar_out_dir,
                use_metaradar,
            )


if __name__ == '__main__':
    Fire(main)
