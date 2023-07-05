import numpy as np
import argparse
import time
from time import strftime, gmtime
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import json

from features.networks import TGraphNet, TGraphNetSeq
from graph import Graph
from angles import *
from evaluation import *
from common.utils import Params, set_logger, copy_weight, load_checkpoint, save_checkpoint_pos_ori, write_log, get_lr, write_train_summary_scalars, write_val_summary_joint, log_gradients_in_model
from common.model import print_layers, weight_init, count_parameters
from data.h36m_dataset import Human36M
from data.pw3d_dataset import PW3D
from data.dhp_dataset import DHPDataset
from common.h36m_skeleton import joint_id_to_names
from data.generators import ChunkedGenerator_Seq, UnchunkedGenerator_Seq, ChunkedGenerator_Frame, ChunkedGenerator_Seq2Seq, eval_data_prepare
import loss
from common.utils import change_momentum
from common.camera_params import project_to_2d_linear, project_to_2d, normalize_screen_coordinates_torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default="../../datasets/human3.6m/orig/pose",
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='../models',
                    help="Directory containing params.json")
parser.add_argument('--init_weight', default=None,
                    help="Path of the init weight file")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
# parser.add_argument('--exp_log', default='train-log',
#                     help="Optional, Log file for experiment details")
parser.add_argument('--tf_logdir', default='../runs/',
                    help="Optional, Directory for saving tensorboard logs.")

parser.add_argument('--exp_suffix', default="", help="experiment name")
parser.add_argument('--run_suffix', default="", help="experiment suffix")
parser.add_argument('--exp_desc', default="", help="experiment description")
parser.add_argument('--seed_value', default="", help="seed value")
parser.add_argument('--mode', default="train", help="train or test")


def evaluate(model, loss_fn, val_gen, metrics, params, epoch, writer, log_dict, exp, detailed=False, viz=True, joint_dict=None):
    t = time.time()
    model.eval()
    summary = []
    t_errs = torch.zeros(81)
    cnt = 0

    total_mpjpe = 0
    N = 0

    with torch.no_grad():
        for cameras_val, batch_3d, batch_6d, batch_2d, batch_edge in val_gen.next_epoch():
            input_2d = torch.FloatTensor(batch_2d).to(device)
            target_pose_3d = torch.FloatTensor(batch_3d).to(device)

            middle_index = int((target_pose_3d.shape[1] - 1) / 2)
            pad = 15
            start_index = middle_index - pad
            end_index = middle_index + pad + 1
            B, T, J, D = target_pose_3d.shape

            predicted_pos3d = model(input_2d)
            predicted_pos3d_center = predicted_pos3d[:, start_index:end_index].reshape(B, (2 * pad + 1), J, D)
            target_pose_3d = target_pose_3d.view_as(predicted_pos3d)
            target_pose_3d_center = target_pose_3d[:, start_index:end_index].reshape(B, (2 * pad + 1), J, D)


            inputs_traj = target_pose_3d[:, :, :1].clone()
            pred_traj = predicted_pos3d[:, :, :1].clone()

            inputs_traj_center = inputs_traj[:, start_index:end_index]
            pred_traj_center = pred_traj[:, start_index:end_index]


            # Train for root relative only this time around
            target_pose_3d[:, :, 0] = 0
            predicted_pos3d[:, :, 0] = 0

            cam_loss = torch.tensor(0)
            loss_pos = loss_fn[0](predicted_pos3d * 0.001, target_pose_3d * 0.001, torch.ones(17).to(predicted_pos3d.device))
            loss_dif = loss_fn[1](predicted_pos3d * 0.001, target_pose_3d * 0.001).to(predicted_pos3d.device)
            loss_velocity = loss_fn[2](predicted_pos3d * 0.001, target_pose_3d * 0.001, axis=1)
            loss_trajectory = loss_fn[0](pred_traj, inputs_traj, 1 / inputs_traj[:, :, :, 2])

            # loss_angle = loss_fn[1](predicted_angle_mat, batch_angles_mat)
            loss_val = loss_pos + loss_dif # + 2.0 * loss_velocity # + loss_proj # + cam_loss

            err_pos = metrics[0](
                predicted_pos3d_center.cpu().data,
                target_pose_3d_center.cpu().data
            )[0]

            total_mpjpe += err_pos * predicted_pos3d.shape[0] * predicted_pos3d.shape[1]
            N += predicted_pos3d.shape[0] * predicted_pos3d.shape[1]

            err_vel = metrics[1](
                predicted_pos3d_center.cpu().data,
                target_pose_3d_center.cpu().data,
                1
            )

            err_vel_traj = metrics[1](
                inputs_traj_center.cpu().data,
                pred_traj_center.cpu().data,
                1
            )

            err_traj = metrics[0](
                inputs_traj_center.cpu().data,
                pred_traj_center.cpu().data
            )[0]

            t_err = metrics[2](
                predicted_pos3d.cpu().data,
                target_pose_3d.cpu().data,
            )[-1]

            t_errs += t_err

            errors = torch.norm(predicted_pos3d_center - target_pose_3d_center, dim=len(target_pose_3d_center.shape)-1).reshape(-1, 17, 1).cpu().data.numpy()


            summary_batch = {
                'val_loss': loss_val.item(),
                'val_err_pos': err_pos.item(),
                'val_err_velocity': err_vel.item(),
                'val_loss_diff': loss_dif.item(),
                'val_loss_velocity': loss_velocity.item(),
                'val_loss_traj': loss_trajectory.item(),
                'val_err_traj': err_traj.item(),
                'val_cam_loss': cam_loss.item(),
                'val_err_traj_vel': err_vel_traj.item(),
                'val_errors': errors
                # 't_err': t_err[-1],
                # 'val_err_geodesic': mean_geodesic_distance.item(),
                # 'val_err_geodesic_wo_hip': mean_geodesic_error_without_hip.item(),
                # 'per_joint_geod_err': err_geodesic
            }
            summary.append(summary_batch)
            cnt += 1

            if cnt % params.save_summary_steps == 0:
                print(summary_batch)

        print(t_errs / cnt)
        print(total_mpjpe / N)

    # mean metrics
    metrics_loss_mean = np.mean([x['val_loss'] for x in summary])
    metrics_err_pos_mean = np.mean([x['val_err_pos'] for x in summary], axis=0)
    metrics_err_velocity_mean = np.mean([x['val_err_velocity'] for x in summary], axis=0)
    metrics_err_traj_mean = np.mean([x['val_err_traj'] for x in summary], axis=0)
    metrics_err_traj_vel_mean = np.mean([x['val_err_traj_vel'] for x in summary])
    metrics_err_pck_auc = np.concatenate([x['val_errors'] for x in summary], axis=0).astype(np.float64)


    pck_joint_ids = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]

    pck_all = metrics[3](
        metrics_err_pck_auc,
        150
    )
    pck_14j = metrics[3](
        metrics_err_pck_auc[:, pck_joint_ids],
        150
    )

    auc_min = 0.0
    auc_max = 155.0

    auc_range = np.arange(auc_min, auc_max, 5)
    pck_aucs_14j = []
    pck_aucs = []
    for pck_thresh_ in auc_range:
        err_pck_tem = pck(metrics_err_pck_auc, pck_thresh_)
        err_pck_14j_tem = pck(metrics_err_pck_auc[:, pck_joint_ids], pck_thresh_)
        pck_aucs.append(err_pck_tem)
        pck_aucs_14j.append(err_pck_14j_tem)

    auc_final = metrics[4](auc_range / auc_range.max(), pck_aucs)
    auc_final_14j = metrics[4](auc_range / auc_range.max(), pck_aucs_14j)

    # Log entries
    # for log file
    logging.info("- Val metrics -\t" +
          "Epoch: " + str(epoch) + "\t" +
          "loss_mean: {0:5.7f} ".format(metrics_loss_mean) + "\t" +
          "val_err_traj_vel: {0:5.7f} ".format(metrics_err_traj_vel_mean) + "\t" +
          "avg_err_pos: {0:5.3f} ".format(metrics_err_pos_mean) + "\t" +
          "avg_err_velocity: {0:5.3f} ".format(metrics_err_velocity_mean) + "\t" +
          "avg_err_traj: {0:5.3f} ".format(metrics_err_traj_mean) + "\t" +
          "avg_pck: {0:5.3f}".format(pck_all) + "\t" +
          "avg_pck_14j: {0:5.3f}".format(pck_14j) + "\t" +
          "auc: {0:5.3f}".format(auc_final) + "\t" +
          "auc_14j: {0:5.3f}".format(auc_final_14j) + "\t"
          )

    # for tensorboard
    summary_epoch = {
        'loss': {
            'val_loss': metrics_loss_mean,
            'val_loss_cam': metrics_err_traj_vel_mean,
        },
        'error': {
            'val_err_pos': metrics_err_pos_mean,
            'val_err_velocity': metrics_err_velocity_mean,
            'val_err_traj': metrics_err_traj_mean,
        }
    }
    summary_epoch["joint"] = {}
    summary_epoch["joint"]["val_pos"] = {}
    summary_epoch["joint"]["val_geod"] = {}

    # for log dict
    if log_dict:
        log_dict['val_losses'].append(metrics_loss_mean)
        # log_dict['val_geod_errors'].append(metrics_geod_err_with_hip_mean)
        log_dict['val_pos_errors'].append(metrics_err_pos_mean)

    # Update tensorboard writer
    if writer:
        write_val_summary_joint(writer=writer,
                                epoch=epoch,
                                summary_epoch=summary_epoch,
                                detailed=True)

    return {'val_err': metrics_err_pos_mean, 'val_geod_err': 0}

def main():
    args = parser.parse_args()
    exp = "stgcn"
    exp_suffix = args.exp_suffix
    run_suffix = args.run_suffix
    seed_value = args.seed_value

    exp_desc = args.exp_desc
    tb_logdir = args.tf_logdir + exp + '/' + exp_suffix + "_" + run_suffix
    model_dir = args.model_dir + "/" + exp + '/' + exp_suffix
    train_test = args.mode

    if not os.path.exists(tb_logdir):
        os.makedirs(tb_logdir)

    # Load parameters
    json_path = os.path.join('../models/', exp, exp_suffix, 'params.json')
    assert os.path.isfile(json_path), "No json file found at {}".format(json_path)
    params = Params(json_path)

    # Set the random seed for reproducible experiments
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
    else:
        torch.manual_seed(seed_value)

    set_logger(os.path.join(model_dir, exp + '_' + exp_suffix + ".log"))
    logging.info("##################################################################")
    logging.info("Experiment: " + exp + '_' + exp_suffix + '_' + run_suffix)
    logging.info("Train/test: " + train_test)
    logging.info("Description: " + exp_desc)
    logging.info(
        "Parameters:\tlearning rate: " + str(params.learning_rate) +
        "\tbatch_size: " + str(params.batch_size) +
        "\tepochs: " + str(params.start_epoch) + " - " + str(params.start_epoch + params.num_epochs) +
        "\tdrop out: " + str(params.dropout)
        )

    global log_dict_fname, log_dict_runname, log_dict_run_fname, log_dict
    log_dict_fname = os.path.join(model_dir, f"{exp + '_' + exp_suffix}.json")
    log_dict_runname = exp + '_' + exp_suffix + '_' + run_suffix
    log_dict_run_fname = os.path.join(model_dir, f"{log_dict_runname}.json")

    log_dict = {}
    log_dict['exp_name'] = exp + '_' + exp_suffix + '_' + run_suffix
    log_dict['exp_desc'] = exp_desc
    log_dict['tensorboard_logdir'] = tb_logdir
    log_dict['hyperparameters'] = {}
    for k in params.dict.keys():
        log_dict['hyperparameters'][k] = params.dict.get(k, "")

    logging.info("Loading test dataset....")
    test_dataset = DHPDataset(data_dir="../data/", train=False)
    pos2d, pos3d = test_dataset.pos2d, test_dataset.pos3d
    val_generator = ChunkedGenerator_Seq2Seq(params.batch_size, cameras=None, poses_2d=pos2d, poses_3d=pos3d,
                                                   chunk_length=31, pad=25, out_all=True, shuffle=False,
                                                   augment=False, reverse_aug=False,)

    # logging.info(f"N Val Frames: {val_generator.num_frames()}, N Val Batches {val_generator.num_batches}")
    logging.info("- done.")

    logging.info("Loading model:")
    model = TGraphNetSeq(infeat_v=params.input_node_feat,
                      infeat_e=params.input_edge_feat,
                      nhid_v=params.num_hidden_nodes,
                      nhid_e=params.num_hidden_edges,
                      n_oute=params.output_edge_feat,
                      n_outv=params.output_node_feat,
                      gcn_window=params.gcn_window,
                      tcn_window=params.tcn_window,
                      in_frames=params.in_frames,
                      gconv_stages=params.gconv_stages,
                      num_groups=params.num_groups,
                      aggregate=params.aggregate,
                      dropout=params.dropout,
                      use_residual_connections=params.use_residual_connections,
                      use_non_parametric=params.use_non_parametric,
                      use_edge_conv=params.use_edge_conv,
                      learn_adj=params.learn_adj,).to(device)

    logging.info("Num of parameters: " + str(count_parameters(model)))
    print_layers(model)
    print(model)
    logging.info("- done.")

    if params.init_weights:
        model.apply(weight_init)
        copy_weight(params.init_weights, model)
        log_dict['init_weights'] = params.init_weights
        logging.info("Model initialized from " + params.init_weights)
    else:
        model.apply(weight_init)
        logging.info("Model initialized")

    logging.info("Creating Optimizer....")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=params.learning_rate,
        amsgrad=True,
        weight_decay=params.weight_decay
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.lr_step_size, gamma=params.lr_gamma)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 90, 100], gamma=params.lr_gamma)

    logging.info("- done.")
    logging.info("Learning rate: {}".format(params.learning_rate))

    loss_fn = [loss.weighted_mpjpe, loss.motion_loss, loss.mean_velocity_error_train]
    metrics = [mpjpe, mean_velocity_error, t_mpjpe, pck, auc]

    ##################################################################
    # Validation
    ##################################################################
    logging.info("Evaluating {}".format(exp))
    logging.info("Restoring from {}".format(params.restore_file))
    load_checkpoint(params.restore_file, model, optimizer=None)
    logging.info("- done.")
    val_metrics = evaluate(model, loss_fn, val_generator, metrics, params, epoch=0, writer=None, log_dict=None, exp=exp, detailed=True, viz=False, joint_dict=joint_id_to_names)


if __name__ == "__main__":
    main()
