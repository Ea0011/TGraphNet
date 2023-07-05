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


def train(model, optimizer, loss_fn, train_gen, metrics, params, epoch, writer, log_dict, exp):
    """
    Training loop of the model
    """
    t = time.time()
    summary = []

    model.train()
    optimizer.zero_grad()

    n_batches = 0
    num_batches = train_gen.num_batches

    for cameras_train, batch_3d, batch_6d, batch_2d, batch_edge in train_gen.next_epoch():
        input_2d = torch.from_numpy(batch_2d).float().to(device)
        target_pose_3d = torch.from_numpy(batch_3d).float().to(device)
        cameras_train = torch.from_numpy(cameras_train.astype('float32')).to(device)
        predicted_pos3d = model(input_2d)

        # target_angle_6d = target_angle_6d.view_as(predicted_angle_6d)
        target_pose_3d = target_pose_3d.view_as(predicted_pos3d)

        # batch_size = input_2d.shape[0]
        # hip_ori = torch.tensor([[[1., 0., 0., 1., 0., 0.]]]).repeat(batch_size, 1, 1).to(device)
        # predicted_angle_6d = torch.cat((hip_ori, predicted_angle_6d), dim=1)
        # batch_angles_6d = torch.cat((hip_ori, target_angle_6d), dim=1)

        # predicted_angle_mat = rot6d_to_rotmat(predicted_angle_6d)
        # batch_angles_mat = rot6d_to_rotmat(batch_angles_6d)

        inputs_traj = target_pose_3d[:, :, :1].clone()
        pred_traj = predicted_pos3d[:, :, :1].clone()

        # Train for root relative only this time around
        target_pose_3d[:, :, 0] = 0
        predicted_pos3d[:, :, 0] = 0

        loss_pos = loss_fn[0](predicted_pos3d * 0.001, target_pose_3d * 0.001, torch.ones(17).to(predicted_pos3d.device))
        loss_dif = loss_fn[1](predicted_pos3d * 0.001, target_pose_3d * 0.001,).to(predicted_pos3d.device)
        # loss_dif = loss_fn[-1](predicted_pos3d * 0.001, torch.ones(17).to(predicted_pos3d.device))
        loss_velocity = loss_fn[2](predicted_pos3d * 0.001, target_pose_3d * 0.001, axis=1)
        loss_trajectory = loss_fn[0](pred_traj, inputs_traj, 1 / inputs_traj[:, :, :, 2])
        
        pred_joints_cam = predicted_pos3d + pred_traj
        pred_joints_img = normalize_screen_coordinates_torch(project_to_2d(pred_joints_cam, cameras_train), 1000, 1000)

        traj_smoothness = loss_fn[1](pred_traj * 0.001, inputs_traj * 0.001)

        loss_proj = loss_fn[0](pred_joints_img, input_2d, torch.ones(17).to(pred_joints_img.device))
        # loss_angle = loss_fn[1](predicted_angle_mat, batch_angles_mat)

        loss_train = loss_pos + loss_dif # + loss_trajectory + loss_proj + 0.001 * traj_smoothness # + loss_trajectory  # + loss_velocity # + loss_proj + 0.001 * traj_smoothness + loss_trajectory # + 2.0 * loss_velocity # + loss_proj # + 0.1 * traj_smoothnes # + cam_loss

        # update model
        optimizer.zero_grad()
        loss_train.backward()

        # if n_batches % params.save_summary_steps == 0:
        #     log_gradients_in_model(model, writer, int(f"{epoch+1}{n_batches}"))

        # nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        optimizer.step()

        n_batches += 1

        err_pos = metrics[0](
            predicted_pos3d.cpu().data,
            target_pose_3d.cpu().data
        )[0]

        err_vel = metrics[1](
            predicted_pos3d.cpu().data,
            target_pose_3d.cpu().data,
            1
        )

        err_vel_traj = metrics[1](
            pred_traj.cpu().data,
            inputs_traj.cpu().data,
            1
        )

        err_traj = metrics[0](
            inputs_traj.cpu().data,
            pred_traj.cpu().data
        )[0]

        # err_geodesic = metrics[1](
        #     predicted_angle_mat.cpu().data.numpy(),
        #     batch_angles_mat.cpu().data.numpy()
        # )
        # mean_geodesic_distance = np.mean(err_geodesic)
        # mean_geodesic_error_without_hip = np.mean(err_geodesic[1:])

        summary_batch = {
            'train_loss': loss_train.item(),
            'train_loss_pos': loss_pos.item(),
            'train_loss_diff': loss_dif.item(),
            'train_loss_velocity': loss_velocity.item(),
            'loss_traj_motion': err_vel_traj.item(),
            'train_err_pos': err_pos.item(),
            'train_err_velocity': err_vel.item(),
            'train_err_traj': err_traj.item(),
            'train_loss_proj': loss_proj.item(),
            'train_loss_traj': loss_trajectory.item(),
            'train_loss_motion_traj': traj_smoothness.item(),
            # 'train_err_geodesic': mean_geodesic_distance.item(),
            # 'train_err_geodesic_wo_hip': mean_geodesic_error_without_hip.item()
            }

        summary.append(summary_batch)

        if n_batches % params.save_summary_steps == 0:
            print('Epoch: {:04d}'.format(epoch),
                  'Batch: {}/{}'.format(n_batches, num_batches),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'loss_pos: {:.4f}'.format(loss_pos.item()),
                  'loss_diff: {:.4f}'.format(loss_dif.item()),
                  'loss_velocity: {:.4f}'.format(loss_velocity.item()),
                  'loss_traj_motion: {:.4f}'.format(err_vel_traj.item()),
                  'train_err_pos: {:.4f}'.format(err_pos.item()),
                  'train_err_vel: {:.4f}'.format(err_vel.item()),
                  'train_err_traj: {:.4f}'.format(err_traj.item()),
                  'train_loss_proj: {:.4f}'.format(loss_proj.item()),
                  'train_loss_traj: {:.4f}'.format(loss_trajectory.item()),
                  'train_loss_motion_traj: {:.4f}'.format(traj_smoothness.item()),
                  'time: {:.4f}s'.format(time.time() - t)
                  )

    # Decay BatchNorm momentum
    # momentum = 0.1 * np.exp(-(epoch + 1) / params.num_epochs * np.log(0.1 / 0.001))
    # changer = change_momentum(momentum)
    # model.apply(changer)

    # mean metrics
    metrics_loss_mean = np.mean([x['train_loss'] for x in summary])
    metrics_loss_pos_mean = np.mean([x['train_loss_pos'] for x in summary])
    metrics_loss_diff_mean = np.mean([x['train_loss_diff'] for x in summary])
    metrics_loss_velocity_mean = np.mean([x['train_loss_velocity'] for x in summary])
    metrics_loss_proj_mean = np.mean([x['train_loss_proj'] for x in summary])
    metrics_loss_traj_mean = np.mean([x['train_loss_traj'] for x in summary])
    metrics_loss_traj_motion_mean = np.mean([x['train_loss_motion_traj'] for x in summary])

    metrics_err_traj_velocity = np.mean([x['loss_traj_motion'] for x in summary])
    metrics_err_pos_mean = np.mean([x['train_err_pos'] for x in summary], axis=0)
    metrics_err_traj_mean = np.mean([x['train_err_traj'] for x in summary], axis=0)
    metrics_err_velocity_mean = np.mean([x['train_err_velocity'] for x in summary], axis=0)

    # metrics_geod_err_mean = np.mean([x['train_err_geodesic_wo_hip'] for x in summary], axis=0)
    # metrics_geod_err_with_hip_mean = np.mean([x['train_err_geodesic'] for x in summary], axis=0)

    # Logging
    # for log file
    logging.info(
        "- Train metrics -\t" +
        "Epoch: " + str(epoch) + "\t" +
        "loss: {0:5.7f} ".format(metrics_loss_mean) + "\t" +
        "loss_pos: {0:5.7f} ".format(metrics_loss_pos_mean) + "\t" +
        "avg_err_pos: {0:5.3f} ".format(metrics_err_pos_mean) + "\t" +
        "avg_err_velocity: {0:5.3f} ".format(metrics_err_velocity_mean) + "\t"
        "avg_loss_proj: {0:5.3f} ".format(metrics_loss_proj_mean) + "\t"
        "avg_loss_traj_velocity: {0:5.3f}".format(metrics_err_traj_velocity) + "\t"
    )

    # for tensorboard
    summary_epoch = {
        'loss': {
            'train_loss': metrics_loss_mean,
            'train_loss_pos': metrics_loss_pos_mean,
            'train_loss_motion': metrics_loss_diff_mean,
            'train_loss_velocity': metrics_loss_velocity_mean,
            'train_loss_traj': metrics_loss_traj_mean,
            'train_traj_motion_loss': metrics_loss_traj_motion_mean,
        },
        'error': {
            'train_err_pos': metrics_err_pos_mean,
            'train_err_velocity': metrics_err_velocity_mean,
            'train_err_traj': metrics_err_traj_mean,
            'traj_err_traj_velocity': metrics_err_traj_velocity,
            # 'train_err_geod_w/o_hip': metrics_geod_err_mean,
            # 'train_err_geod_with_hip': metrics_geod_err_with_hip_mean
        }
    }

    write_train_summary_scalars(writer=writer,
                                experiment=exp,
                                epoch=epoch,
                                batch_idx=n_batches,
                                num_batches=num_batches,
                                summary=summary_epoch)

    # for log dict
    log_dict['train_losses'].append(metrics_loss_mean)
    # log_dict['train_geod_errors'].append(metrics_geod_err_mean)
    log_dict['train_pos_errors'].append(metrics_err_pos_mean)

    return metrics_loss_mean


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
            cameras_val = torch.from_numpy(cameras_val.astype('float32')).to(device)

            # out_3d, input_2d = eval_data_prepare(params.in_frames, input_2d, target_pose_3d,)
            # target_pose_3d = out_3d.to(device)
            # input_2d = input_2d.to(device)
            # cameras_val = cameras_val.repeat(input_2d.shape[0], 1)


            middle_index = int((target_pose_3d.shape[1] - 1) / 2)
            pad = 15
            start_index = middle_index - pad
            end_index = middle_index + pad + 1
            B, T, J, D = target_pose_3d.shape

            predicted_pos3d = model(input_2d)
            predicted_pos3d_center = predicted_pos3d[:, start_index:end_index].reshape(B, (2 * pad + 1), J, D)

            # target_angle_6d = target_angle_6d[:, middle_index].view_as(predicted_angle_6d)
            target_pose_3d = target_pose_3d.view_as(predicted_pos3d)
            target_pose_3d_center = target_pose_3d[:, start_index:end_index].reshape(B, (2 * pad + 1), J, D)


            inputs_traj = target_pose_3d[:, :, :1].clone()
            pred_traj = predicted_pos3d[:, :, :1].clone()

            inputs_traj_center = inputs_traj[:, start_index:end_index]
            pred_traj_center = pred_traj[:, start_index:end_index]


            # Train for root relative only this time around
            target_pose_3d[:, :, 0] = 0
            predicted_pos3d[:, :, 0] = 0

            # cameras_val[:, 4:] = 0
            # pred_cam[:, 4:] = 0

            cam_loss = torch.tensor(0)
            loss_pos = loss_fn[0](predicted_pos3d * 0.001, target_pose_3d * 0.001, torch.ones(17).to(predicted_pos3d.device))
            loss_dif = loss_fn[1](predicted_pos3d * 0.001, target_pose_3d * 0.001).to(predicted_pos3d.device)
            loss_velocity = loss_fn[2](predicted_pos3d * 0.001, target_pose_3d * 0.001, axis=1)
            loss_trajectory = loss_fn[0](pred_traj, inputs_traj, 1 / inputs_traj[:, :, :, 2])

            pred_joints_cam = predicted_pos3d + pred_traj
            pred_joints_img = normalize_screen_coordinates_torch(project_to_2d_linear(pred_joints_cam, cameras_val[:, :9]), 1000, 1000)

            loss_proj = loss_fn[0](pred_joints_img, input_2d, torch.ones(17).to(pred_joints_img.device))

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

            # err_geodesic = metrics[1](
            #     predicted_angle_mat.cpu().data.numpy(),
            #     batch_angles_mat.cpu().data.numpy()
            # )
            # mean_geodesic_distance = np.mean(err_geodesic)
            # mean_geodesic_error_without_hip = np.mean(err_geodesic[1:])

            summary_batch = {
                'val_loss': loss_val.item(),
                'val_err_pos': err_pos.item(),
                'val_err_velocity': err_vel.item(),
                'val_loss_diff': loss_dif.item(),
                'val_loss_velocity': loss_velocity.item(),
                'val_loss_traj': loss_trajectory.item(),
                'val_err_traj': err_traj.item(),
                'val_cam_loss': cam_loss.item(),
                'val_loss_proj': loss_proj.item(),
                'val_err_traj_vel': err_vel_traj.item(),
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
    metrics_loss_proj_mean = np.mean([x['val_loss_proj'] for x in summary])


    # metrics_geod_err_mean = np.mean([x['val_err_geodesic_wo_hip'] for x in summary], axis=0)
    # metrics_geod_err_with_hip_mean = np.mean([x['val_err_geodesic'] for x in summary], axis=0)
    # metrics_geod_err_per_joint = np.mean([x['per_joint_geod_err'] for x in summary], axis=0)

    # Log entries
    # for log file
    logging.info("- Val metrics -\t" +
          "Epoch: " + str(epoch) + "\t" +
          "loss_mean: {0:5.7f} ".format(metrics_loss_mean) + "\t" +
          "val_err_traj_vel: {0:5.7f} ".format(metrics_err_traj_vel_mean) + "\t" +
          "loss_proj_mean: {0:5.7f} ".format(metrics_loss_proj_mean) + "\t" +
          "avg_err_pos: {0:5.3f} ".format(metrics_err_pos_mean) + "\t" +
          "avg_err_velocity: {0:5.3f} ".format(metrics_err_velocity_mean) + "\t"
          "avg_err_traj: {0:5.3f} ".format(metrics_err_traj_mean) + "\t"
        #   "avg_err_geodesic: {0:5.3f} ".format(metrics_geod_err_mean)
          )
    # for joint_id in range(len(metrics_err_joint)):
    #     logging.info("{0}:\t pos_err: {1:5.3f}\t geod_err: {2:5.3f}".format(joint_dict[joint_id], metrics_err_joint[joint_id], metrics_geod_err_per_joint[joint_id]))

    # for tensorboard
    summary_epoch = {
        'loss': {
            'val_loss': metrics_loss_mean,
            'val_loss_cam': metrics_err_traj_vel_mean,
            'val_loss_proj': metrics_loss_proj_mean,
        },
        'error': {
            'val_err_pos': metrics_err_pos_mean,
            'val_err_velocity': metrics_err_velocity_mean,
            'val_err_traj': metrics_err_traj_mean,
            # 'val_err_geod_w/o_hip': metrics_geod_err_mean,
            # 'val_err_geod_with_hip': metrics_geod_err_with_hip_mean
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


def train_and_evaluate(model, train_generator, val_generator, optimizer, scheduler, loss_fn, metrics, params,
                       model_dir, tf_logdir, log_dict, exp, viz, restore_file=None, joint_dict=None):
    """
    Train the model and evaluate every epoch
    :return:
    """
    best_val_err = params.last_best_err
    log_dict['best_val_err'] = {"epoch": "-", "err": best_val_err}

    # reload weights from restore_file if specified
    if restore_file:
        logging.info("Restoring parameters from {}".format(restore_file))
        _, best_val_err, best_val_geod_err = load_checkpoint(restore_file, model, optimizer, scheduler, last_best=True)
        log_dict['best_val_err'] = {"epoch": "-", "err": best_val_err , "geod_err": best_val_geod_err}
        logging.info("- done.")
        logging.info("last best val_err: " + str(best_val_err) + "\t last best geod_err: " + str(best_val_geod_err))

    # forced change of lr
    if params.force_lr_change:
        for param_group in optimizer.param_groups:
            param_group['lr'] = params.learning_rate
            logging.info("Forced lr change - start lr: " + str(param_group['lr']))

    writer = SummaryWriter(tf_logdir)
    end_epoch = params.start_epoch + params.num_epochs

    log_dict['train_losses'] = []
    log_dict['train_geod_errors'] = []
    log_dict['train_pos_errors'] = []
    log_dict['val_losses'] = []
    # log_dict['val_geod_errors'] = []
    log_dict['val_pos_errors'] = []
    # log_dict['val_geod_errors_joints'] = {}
    log_dict['val_pos_errors_joints'] = {}

    # for joint_id in range(params.num_joints):
    #     log_dict['val_geod_errors_joints'][joint_dict[joint_id]] = []

    for epoch in range(params.start_epoch, end_epoch):
        logging.info("Epoch {}/{}".format(epoch, end_epoch-1))
        t0 = time.time()
        # train for one epoch
        train_loss = train(model, optimizer, loss_fn, train_generator, metrics, params, epoch, writer, log_dict, exp)
        logging.info("Epoch {} training time: {}".format(epoch, strftime("%H:%M:%S", gmtime(time.time() - t0))))
        lr_now = get_lr(optimizer)
        logging.info("Epoch {} learning_rate: {:.10f}".format(epoch, lr_now))

        # update lr
        scheduler.step()

        is_best = False
        is_best_pos = False
        is_best_ori = False

        # evaluate every epoch, on the validation set
        if epoch % 1 == 0:
            t1 = time.time()
            val_metrics = evaluate(model, loss_fn, val_generator, metrics, params, epoch, writer, log_dict, exp, detailed=False, viz=viz, joint_dict=joint_dict)
            logging.info("Epoch {} validation time: {}".format(epoch, strftime("%H:%M:%S", gmtime(time.time() - t1))))
            val_err = val_metrics['val_err']
            val_geod_err = val_metrics['val_geod_err']

            if epoch > params.start_epoch:
                is_best = (val_err <= best_val_err) and (val_geod_err <= best_val_geod_err)
                is_best_pos = val_err <= best_val_err
                is_best_ori = val_geod_err <= best_val_geod_err
            elif epoch == 0:
                best_val_err = val_err
                best_val_geod_err = val_geod_err

            # If best_eval, best_save_path
            if is_best:
                logging.info("- Found new min validation error")
                best_val_err = val_err
                best_val_geod_err = val_geod_err
                log_dict['best_val_err'] = {"epoch": epoch, "err": best_val_err, "geod_err": best_val_geod_err}

            if is_best_pos:
                best_val_err = val_err
                log_dict['best_val_err_pos'] = {"epoch": epoch, "err": best_val_err}
                logging.info("- Found new min validation pos error")

            if is_best_ori:
                best_val_geod_err = val_geod_err
                log_dict['best_val_err_ori'] = {"epoch": epoch, "geod_err": best_val_geod_err}
                logging.info("- Found new min validation ori error")

        # # Save weights
        save_checkpoint_pos_ori({'epoch': epoch,
                                 'state_dict': model.state_dict(),
                                 'optim_dict': optimizer.state_dict(),
                                 'lr_dict': scheduler.state_dict(),
                                 'last_best_err': best_val_err,
                                 'last_best_geod_err': best_val_geod_err,
                                 },
                                is_best=is_best,
                                is_best_pos=is_best_pos,
                                is_best_ori=is_best_ori,
                                checkpoint=model_dir)

        # save temporary run log
        write_log(log_dict_run_fname, log_dict_runname, log_dict, "w")

    writer.close()


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

    if train_test == "train":
        logging.info("Loading training dataset....")
        train_dataset = Human36M(data_dir=args.data_dir, train=True, ds_category=params.ds_category)
        pos2d, pos3d, angles_6d, edge_features, cameras = train_dataset.pos2d, train_dataset.pos3d_centered, train_dataset.gt_angles_6d, train_dataset.edge_features, train_dataset.cam

        kps_left, kps_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]

        train_generator = ChunkedGenerator_Seq2Seq(params.batch_size, cameras=cameras, poses_2d=pos2d, poses_3d=pos3d,
                                                   chunk_length=5, pad=38, out_all=True, shuffle=True,
                                                   augment=False, reverse_aug=False,
                                                   kps_left=kps_left, kps_right=kps_right,
                                                   joints_left=kps_left,
                                                   joints_right=kps_right)
        # train_generator = ChunkedGenerator_Seq(params.batch_size//params.stride, cameras=None, poses_2d=pos2d, poses_3d=pos3d, rot_6d=angles_6d,
        #                                        edge_feat=edge_features, chunk_length=params.in_frames, pad=0, shuffle=True,)
        # train_generator = ChunkedGenerator_Frame(params.batch_size // params.stride, cameras=None, poses_2d=pos2d, poses_3d=pos3d, rot_6d=angles_6d,
        #                                          edge_feat=edge_features, chunk_length=11, pad=35, shuffle=True,)

        logging.info(f"N Frames: {train_generator.num_frames()}, N Batches {train_generator.num_batches}")
        logging.info("- done.")

    logging.info("Loading test dataset....")
    test_dataset = Human36M(data_dir=args.data_dir, train=False, ds_category=params.ds_category)
    # test_dataset = PW3D(data_file="../data/pw3d_test.pkl")
    cam, pos2d, pos3d, angles_6d, edge_features = test_dataset.cam, test_dataset.pos2d, test_dataset.pos3d_centered, test_dataset.gt_angles_6d, test_dataset.edge_features
    # cam, pos2d, pos3d, angles_6d, edge_features = test_dataset.cam, test_dataset.pos2d, test_dataset.pos3d, [], []
    # val_generator = UnchunkedGenerator_Seq(cameras=cam, poses_2d=pos2d, poses_3d=pos3d, rot_6d=angles_6d, edge_feat=edge_features,)
    val_generator = ChunkedGenerator_Seq2Seq(params.batch_size, cameras=cam, poses_2d=pos2d, poses_3d=pos3d,
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

    loss_fn = [loss.weighted_mpjpe, loss.motion_loss, loss.mean_velocity_error_train, loss.mean_diff_loss]
    metrics = [mpjpe, mean_velocity_error, t_mpjpe]

    if train_test == "train":
        logging.info("Starting training.. ")
        train_and_evaluate(model, train_generator, val_generator, optimizer, scheduler, loss_fn, metrics, params, model_dir, tb_logdir, log_dict,
                           exp, False, params.restore_file, joint_id_to_names)

        # save log file
        write_log(log_dict_fname, log_dict_runname, log_dict, "a")
        logging.info("Training finished.")
        logging.info("Log saved to " + log_dict_fname)
        try:
            os.remove(log_dict_run_fname)
            logging.info("run log file deleted: " + log_dict_run_fname)
        except OSError as error:
            logging.info("run log file not deleted " + log_dict_run_fname)

    ##################################################################
    # Validation
    ##################################################################
    if train_test == "test":
        logging.info("Evaluating {}".format(exp))
        logging.info("Restoring from {}".format(params.restore_file))
        load_checkpoint(params.restore_file, model, optimizer=None)
        logging.info("- done.")
        val_metrics = evaluate(model, loss_fn, val_generator, metrics, params, epoch=0, writer=None, log_dict=None, exp=exp, detailed=True, viz=False, joint_dict=joint_id_to_names)


if __name__ == "__main__":
    main()
