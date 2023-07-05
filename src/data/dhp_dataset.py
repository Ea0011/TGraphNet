
"""
Adapted from https://raw.githubusercontent.com/paTRICK-swk/P-STMO/be41b53bf8bb0a1391f8b71fd42615f820e5d83c/common/load_data_3dhp_mae.py
"""

import torch
import numpy as np
from data.generators import ChunkedGenerator_Seq2Seq, ChunkedGeneratorDHP


class DHPDataset:
    def __init__(self, data_dir, train=True):
        self.train = train
        self.data_dir = data_dir
        self.pos3d, self.pos2d, self.valid_frame = self.prepare_data(data_dir, train=train)

    def normalize_2d(self, pose, w, h):
        assert pose.shape[-1] == 2
        return pose / w * 2 - [1, h / w]

    def map_h36m_to_3dhp(self, target):
        mapping = {
          0: 14,
          1: 8,
          2: 9,
          3: 10,
          4: 11,
          5: 12,
          6: 13,
          7: 15,
          10: 0,
          11: 5,
          12: 6,
          13: 7,
          14: 2,
          15: 3,
          16: 4,
          8: 1,
          9: 16,
        }

        target_h36m = np.zeros_like(target)
        for index in range(target.shape[1]):
            target_h36m[:, index] = target[:, mapping[index]]

        return target_h36m

    def prepare_data(self, path, train=True):
        out_poses_3d = {}
        out_poses_2d = {}
        out_poses_valid = {}

        if train is True:
            data = np.load(path + "data_train_3dhp.npz",  allow_pickle=True)['data'].item()
            for seq in data.keys():
                for cam in data[seq][0].keys():
                    anim = data[seq][0][cam]

                    subject_name, seq_name = seq.split(" ")

                    data_3d = anim['data_3d']
                    data_h36m = self.map_h36m_to_3dhp(data_3d)
                    data_h36m[:, 1:] -= data_h36m[:, :1]
                    out_poses_3d[seq] = data_h36m

                    data_2d = anim['data_2d']

                    data_2d[..., :2] = self.normalize_2d(data_2d[..., :2], w=2048, h=2048)
                    out_poses_2d.append(self.map_h36m_to_3dhp(data_2d))

            return out_poses_3d, out_poses_2d
        else:
            data = np.load(path + "data_test_3dhp.npz", allow_pickle=True)['data'].item()
            for seq in data.keys():
                anim = data[seq]
                out_poses_valid[seq] = anim['valid']
                data_3d = anim['data_3d']
                data_h36m = self.map_h36m_to_3dhp(data_3d)
                data_h36m[:, 1:] -= data_h36m[:, :1]
                out_poses_3d[seq] = data_h36m

                data_2d = anim['data_2d']

                if seq == "TS5" or seq == "TS6":
                    width = 1920
                    height = 1080
                else:
                    width = 2048
                    height = 2048
                data_2d[..., :2] = self.normalize_2d(data_2d[..., :2], w=width, h=height)
                out_poses_2d[seq] = self.map_h36m_to_3dhp(data_2d)

            return out_poses_3d, out_poses_2d, out_poses_valid

    def __len__(self):
        return len(self.gt2d)


if __name__ == "__main__":
    dataset = DHPDataset(data_dir="../data/", train=False)

    gen = ChunkedGeneratorDHP(64, cameras=None, poses_2d=dataset.pos2d, poses_3d=dataset.pos3d,
                              valid_frame=dataset.valid_frame, train=False,
                                                   chunk_length=1, pad=0, out_all=True, shuffle=False,
                                                   augment=False, reverse_aug=False,)

    print(f"N Frames: {gen.num_frames()}, N Batches {gen.num_batches}")
    idx = 0
    for seq_name, start_3d, end_3d, flip, reverse in gen.pairs:
        cam, batch_3d, batch_2d, seq, subject, cam_ind= gen.get_batch(seq_name, start_3d, end_3d, flip, reverse)
        batch_3d = torch.FloatTensor(batch_3d)
        traj = batch_3d[:, :, :1].clone()
        idx += 1

        print(batch_3d)

        break
