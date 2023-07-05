import argparse
from time import strftime, gmtime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.benchmark as benchmark
from common.model import print_layers, count_parameters, get_model_size

from features.networks import TGraphNet, TGraphNetSeq
from angles import *
from evaluation import *

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Benchmark Parameters')
parser.add_argument('--batch_size', metavar='B', type=int,
                    help='Batch size. Number of sequences in a singe input',
                    default=64)
parser.add_argument('--mode', help='which passes to benchmark', type=str, choices=['forward', 'both'], default="forward")
parser.add_argument('--n_batches', help='number of batches', type=int, default=1)
parser.add_argument('--n_measurements', help='number times to benchmark before report', type=int, default=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def forward_backward_bench(model, optimizer, batch_size, n_frames, n_batches=1):
    """
    Training loop of the model
    """
    model.train()
    optimizer.zero_grad()

    # synthetic data
    for i in tqdm(range(n_batches)):
        batch_pose_2d = torch.randn((batch_size, n_frames, 17, 2)).to(device)
        batch_edge_feat = torch.randn((batch_size, n_frames, 16, 4)).to(device)
        batch_pose_3d = torch.randn((batch_size, 17, 3)).to(device)
        batch_angles_6d = torch.randn((batch_size, 16, 6)).to(device)

        # print("1. Before forward pass: {}".format(torch.cuda.memory_allocated(device)))
        predicted_pos3d, pred_cam = model(batch_pose_2d, batch_edge_feat)
        print(predicted_pos3d.shape)
        # print("2. After forward pass: {}".format(torch.cuda.memory_allocated(device)))

        # concat static hip orientation (zero) for ploss
        # batch_size = batch_pose_2d.shape[0]
        # hip_ori = torch.tensor([[[1., 0., 0., 1., 0., 0.]]]).repeat(batch_size,1,1).to(device)
        # predicted_angle_6d = torch.cat((hip_ori, predicted_angle_6d), dim=1)
        # batch_angles_6d = torch.cat((hip_ori, batch_angles_6d), dim=1)

        # predicted_angle_mat = rot6d_to_rotmat(predicted_angle_6d)
        # batch_angles_mat = rot6d_to_rotmat(batch_angles_6d)

        loss_train = F.mse_loss(predicted_pos3d, predicted_pos3d)

        # update model
        optimizer.zero_grad()
        loss_train.backward()
        # print("3. After backward pass: {}".format(torch.cuda.memory_allocated(device)))
        optimizer.step()
        # print("4. After optim step: {}".format(torch.cuda.memory_allocated(device)))

    return "Done"


def forward_bench(model, batch_size, n_frames):
    model.train()

    # synthetic data
    batch_pose_2d = torch.randn((batch_size, n_frames, 17, 2)).to(device)
    batch_edge_feat = torch.randn((batch_size, n_frames, 16, 4)).to(device)
    batch_pose_3d = torch.randn((batch_size, 17, 3)).to(device)
    batch_angles_6d = torch.randn((batch_size, 16, 6)).to(device)

    predicted_pos3d, predicted_angle_6d = model(batch_pose_2d, batch_edge_feat)

    return predicted_pos3d, predicted_angle_6d


if __name__ == "__main__":
    # gcn = TGraphNetSeq(infeat_v=2,
    #                 infeat_e=4,
    #                 nhid_v=[[256, 256], [256, 256], [256, 256], [256, 256]],
    #                 nhid_e=[[256, 256], [256, 256], [256, 256], [256, 256]],
    #                 n_oute=6,
    #                 n_outv=3,
    #                 gcn_window=[3, 3, 3, 3,],
    #                 tcn_window=[3, 3, 3, 3,],
    #                 num_groups=4,
    #                 aggregate=[True] * 4,
    #                 in_frames=81,
    #                 gconv_stages=[1, 1, 1, 1],
    #                 dropout=0.1,
    #                 use_residual_connections=True,
    #                 use_non_parametric=False,
    #                 use_edge_conv=False,
    #                 learn_adj=False).to(device)

    # [[16, 32], [32, 64], [64, 128], [128, 256]]
    gcn = TGraphNetSeq(infeat_v=2,
                    infeat_e=4,
                    nhid_v=[[2, 128], [128, 256]],
                    nhid_e=[[2, 32], [32, 64], [64, 128]],
                    n_oute=6,
                    n_outv=3,
                    gcn_window=[3, 3],
                    tcn_window=[3, 3],
                    num_groups=4,
                    aggregate=[True, False],
                    in_frames=9,
                    gconv_stages=[3, 4],
                    dropout=0.1,
                    use_residual_connections=True,
                    use_non_parametric=False,
                    use_edge_conv=False,
                    learn_adj=False).to(device)

    print(gcn)
    print_layers(gcn)
    print(count_parameters(gcn), get_model_size(gcn))

    args = parser.parse_args()

    if args.mode == "both":
        opt = optim.AdamW(gcn.parameters(), lr=0.001)
        t0 = benchmark.Timer(
            stmt="forward_backward_bench(gcn, opt, {}, 9, {})".format(str(args.batch_size), str(args.n_batches)),
            label="forward and backward pass benchmark on {} device".format(str(device)),
            globals={
                'forward_backward_bench': forward_backward_bench,
                'opt': opt,
                'gcn': gcn,
                'device': device,
            }
        )
        print(t0.timeit(args.n_measurements))
    else:
        t1 = benchmark.Timer(
            stmt="forward_bench(gcn, {}, 81)".format(str(args.batch_size)),
            label="forward pass benchmark on {} device".format(str(device)),
            num_threads=torch.get_num_threads(),
            globals={
                'forward_backward_bench': forward_backward_bench,
                'forward_bench': forward_bench,
                'gcn': gcn,
                'device': device,
            }
        )
        print(t1.timeit(args.n_measurements))
