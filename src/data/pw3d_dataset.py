import numpy as np
import os
import glob
import pickle
from common.h36m_skeleton import *
from common.camera_params import normalize_screen_coordinates, project_to_2d_linear, project_to_2d, normalize_screen_coordinates_torch
from data.generators import ChunkedGenerator_Seq, ChunkedGenerator_Frame, ChunkedGenerator_Seq2Seq


class PW3D:
    def __init__(self, data_file, actions="all"):
        self.file = data_file
        self.actions = actions
        self.cam, self.pos2d, self.pos3d = [], [], []

        self.data = self.load_data(self.file)
        self.data = {k: v for k, v in self.data.items() if k.split("_")[1] in self.actions} if self.actions != "all" else self.data

        self.preprocess_data(self.data)

    def load_data(self, fp):
        file = open(fp, 'rb')
        data = pickle.load(file)

        file.close()

        return data

    def preprocess_data(self, data):
        for vid, d in data.items():
            self.pos2d.append(np.array(d['pos2d']))
            self.pos3d.append(np.array(d['pos3d']))
            self.cam.append(np.array(d['cam']))


if __name__ == "__main__":
    test_data = PW3D("../data/pw3d_test.pkl", actions="all")
    pos2d, pos3d, cameras = test_data.pos2d, test_data.pos3d, test_data.cam

    kps_left, kps_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]

    gen = ChunkedGenerator_Seq2Seq(128, cameras, pos3d, pos2d, chunk_length=11, pad=35, out_all=True, shuffle=False,
                                                    augment=False, reverse_aug=False,
                                                    kps_left=kps_left, kps_right=kps_right,
                                                    joints_left=kps_left,
                                                    joints_right=kps_right)
    print(f"N Frames: {gen.num_frames()}, N Batches {gen.num_batches}")
    idx = 0
    for cam, batch_3d, batch_6d, batch_2d, batch_edge in gen.next_epoch():
        cam = torch.from_numpy(cam.astype('float32')).clone()
        batch_3d = torch.FloatTensor(batch_3d)
        traj = batch_3d[:, :, :1].clone()
        batch_3d[:, :, 0] = 0
        prj = project_to_2d((batch_3d + traj), cam[:, :9])
        prj_norm = normalize_screen_coordinates_torch(prj, cam[:, -2][0], cam[:, -1][0])
        print(torch.mean(torch.norm(torch.from_numpy(batch_2d) - prj_norm, dim=-1), dim=1))
        print(torch.mean(torch.norm(prj_norm - torch.from_numpy(batch_2d), dim=3)))
        idx += 1
