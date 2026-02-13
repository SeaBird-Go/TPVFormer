'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-08-15 19:21:45
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''



import argparse, torch, os, json
import shutil
import numpy as np
import os.path as osp
import mmcv
from mmcv import Config
from collections import OrderedDict



def revise_ckpt(state_dict):
    tmp_k = list(state_dict.keys())[0]
    if tmp_k.startswith('module.'):
        state_dict = OrderedDict(
            {k[7:]: v for k, v in state_dict.items()})
    return state_dict


if __name__ == "__main__":
    import sys; sys.path.insert(0, os.path.abspath('.'))

    device = torch.device('cuda:0')
    # device = torch.device('cpu')
    ## prepare config
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv04_occupancy.py')
    parser.add_argument('--work-dir', type=str, default='out/tpv_occupancy')
    parser.add_argument('--ckpt-path', type=str, default='out/tpv_occupancy/latest.pth')
    parser.add_argument('--vis-train', action='store_true', default=False)
    parser.add_argument('--save-path', type=str, default='out/tpv_occupancy/frames')
    parser.add_argument('--frame-idx', type=int, default=0, nargs='+', 
                        help='idx of frame to visualize, the idx corresponds to the order in pkl file.')
    parser.add_argument('--mode', type=int, default=0, help='0: occupancy, 1: predicted point cloud, 2: gt point cloud')

    args = parser.parse_args()
    print(args)

    cfg = Config.fromfile(args.py_config)
    dataset_config = cfg.dataset_params

    # prepare model
    logger = mmcv.utils.get_logger('mmcv')
    logger.setLevel("WARNING")
    if cfg.get('occupancy', False):
        from builder import tpv_occupancy_builder as model_builder
    else:
        from builder import tpv_lidarseg_builder as model_builder
    my_model = model_builder.build(cfg.model).to(device)
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        print(my_model.load_state_dict(revise_ckpt(ckpt)))
    my_model.eval()

    # prepare data
    from nuscenes import NuScenes
    from visualization.dataset import ImagePoint_NuScenes_vis, DatasetWrapper_NuScenes_vis

    if args.vis_train:
        pkl_path = 'data/nuscenes_lidarseg_infos_train.pkl'
    else:
        pkl_path = 'data/nuscenes_lidarseg_infos_val.pkl'
    
    data_path = 'data/nuscenes'
    label_mapping = dataset_config['label_mapping']

    pt_dataset = ImagePoint_NuScenes_vis(
        data_path, imageset=pkl_path,
        label_mapping=label_mapping)

    dataset = DatasetWrapper_NuScenes_vis(
        pt_dataset,
        grid_size=cfg.grid_size,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["fill_label"],
        phase='val'
    )
    print(len(dataset))

    for index in args.frame_idx:
        print(f'processing frame {index}')
        batch_data, filelist, scene_meta, timestamp = dataset[index]
        imgs, img_metas, vox_label, grid, pt_label = batch_data
        imgs = torch.from_numpy(np.stack([imgs]).astype(np.float32)).to(device)
        grid = torch.from_numpy(np.stack([grid]).astype(np.float32)).to(device)
        with torch.no_grad():
            outputs_vox, outputs_pts = my_model(img=imgs, 
                                                img_metas=[img_metas], 
                                                points=grid.clone())
        
            predict_vox = torch.argmax(outputs_vox, dim=1) # bs, w, h, z
            predict_vox = predict_vox.squeeze(0).cpu().numpy() # w, h, z

            predict_pts = torch.argmax(outputs_pts, dim=1) # bs, n, 1, 1
            predict_pts = predict_pts.squeeze().cpu().numpy() # n

        voxel_origin = dataset_config['min_volume_space']
        voxel_max = dataset_config['max_volume_space']
        grid_size = cfg.grid_size
        resolution = [(e - s) / l for e, s, l in zip(voxel_max, voxel_origin, grid_size)]

        print(predict_pts.shape, predict_pts.dtype)
        print(predict_pts.min(), predict_pts.max(), np.unique(predict_pts))
        print(predict_vox.shape)
        print(grid_size, resolution)
        print(grid.shape, pt_label.shape)

        np.savez('2683-tpvformer.npz', 
                 predict_pts=predict_pts, 
                 predict_vox=predict_vox,
                 voxel_origin=voxel_origin,
                 resolution=resolution,
                 grid=grid.squeeze(0).cpu().numpy(),
                 pt_label=pt_label.squeeze(-1))