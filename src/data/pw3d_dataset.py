import numpy as np
import os
import glob
import pickle5 as pickle
from common.h36m_skeleton import *
from common.camera_params import normalize_screen_coordinates, project_to_2d_linear, project_to_2d, normalize_screen_coordinates_torch
from data.generators import ChunkedGenerator_Seq, ChunkedGenerator_Frame, ChunkedGenerator_Seq2Seq


class PW3D:
    def __init__(self, data_file, actions="all"):
        self.file = data_file
        self.actions = actions
        self.cam, self.pos2d, self.pos3d = [], [], []

        self.data = self.load_data(self.file)
        self.data = {k: v for k, v in self.data.items() if k in self.actions} if self.actions != "all" else self.data

        self.preprocess_data(self.data)

    def load_data(self, fp):
        file = open(fp, 'rb')
        data = pickle.load(file)

        file.close()

        return data

    def preprocess_data(self, data):
        for vid, d in data.items():
            print(vid)
            self.pos2d.append(np.array(d['pos2d']))
            self.pos3d.append(np.array(d['pos3d']))
            self.cam.append(np.array(d['cam']))


if __name__ == "__main__":
    test_data = PW3D("../data/pw3d_test.pkl", actions=['downtown_walkBridge_01_0'])
    pos2d, pos3d, cameras = test_data.pos2d, test_data.pos3d, test_data.cam

    kps_left, kps_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]

    gen = ChunkedGenerator_Seq2Seq(128, cameras, pos3d, pos2d, chunk_length=11, pad=35, out_all=True, shuffle=False,
                                                    augment=False, reverse_aug=False,
                                                    kps_left=kps_left, kps_right=kps_right,
                                                    joints_left=kps_left,
                                                    joints_right=kps_right)
