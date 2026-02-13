import pickle
import os
import os.path as osp
from nuscenes import NuScenes



def add_lidarseg_info(data_infos, nusc, data_path="data/nuscenes/"):
    for info in data_infos:
        lidar_sd_token = nusc.get('sample', info['token'])['data']['LIDAR_TOP']
        lidarseg_labels_filename = os.path.join(data_path, nusc.get('lidarseg', lidar_sd_token)['filename'])
        info['lidarseg'] = lidarseg_labels_filename
    return data_infos


def main():
    version = "v1.0-trainval"
    data_path="data/nuscenes/"
    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)

    for split in ['train', 'val']:
        pickle_path = f"data/nuscenes_infos_{split}.pkl"
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        data_infos = data['infos']
        print(len(data_infos))

        data_infos = add_lidarseg_info(data_infos, nusc, data_path=data_path)
        print(len(data_infos))

        pickle_path = f"data/nuscenes_lidarseg_infos_{split}.pkl"

        data_new = {'metadata': data['metadata'], 
                    'infos': data_infos}

        with open(pickle_path, 'wb') as f:
            pickle.dump(data_new, f)


def main2():
    split = "val"
    pickle_path = f"data/nuscenes_infos_{split}.pkl"
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    data_infos = data['infos']
    data_infos = list(sorted(data_infos, key=lambda e: e['timestamp']))

    print(len(data_infos))

    info = data_infos[916]
    print(info['token'])
    
    

if __name__ == "__main__":
    main2()