# refined from https://github.com/vanoracai/Exploiting-Spatial-temporal-Relationships-for-3D-Pose-Estimation-via-Graph-Convolutional-Networks

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy import integrate


def dist_acc(dists, thr=0.5):

    """ Return percentage below threshold while ignoring values with a -1 """

    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def calc_dists(preds, target, normalize):

    """

    calculate Eucledian distance per joint

    :param preds:

    :param target:

    :param normalize:

    :return: num_of_joints x num_batches

    """

    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for _batch in range(preds.shape[0]):
        for _joint in range(preds.shape[1]):
            if True: # all(target[_batch, _joint, :] > 1): #condition for heatmap, not suitable for regression
                normed_preds = preds[_batch, _joint, :] / normalize[_batch]
                normed_targets = target[_batch, _joint, :] / normalize[_batch]
                dists[_joint, _batch] = np.linalg.norm(normed_preds - normed_targets)

            else:
                dists[_joint, _batch] = -1

    return dists


def accuracy_pck_2d(output, target, out_height=224, out_width=224, thr=0.5):

    """

    Calculate accuracy according to PCK,

    First value to be returned is average accuracy across 'joint_inidces',

    followed by individual accuracies

    """

    batch_size = output.shape[0]
    num_joints = output.shape[1]
    dim = output.shape[2]

    joint_inidces = list(range(num_joints))
    norm = np.ones((batch_size, dim)) * np.array([out_height, out_width]) / 10
    dists = calc_dists(output, target, norm)
    acc = np.zeros((len(joint_inidces) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(joint_inidces)):
        acc[i + 1] = dist_acc(dists[joint_inidces[i]], thr)
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc

    return acc


def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    returns mean error across all data points
    and mean per joint error 17 x 1
    """
    assert predicted.shape == target.shape
    err = torch.norm(predicted - target, dim=len(target.shape)-1) # num_batch x num_joint
    return torch.mean(err), torch.mean(err, dim=0)


def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    # optimum rotation matrix of Y
    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    traceR = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = traceR * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    p_dist = np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1))

    # Return MPJPE
    return p_dist


def pck(errors, thresh=150): # OKPS reports pck @ 150mm
    """"
    Computes Percentage-Correct Keypoints
    :param thresh: Threshold value used for PCK
    :return: the final PCK value
    """
    errors_pck = errors <= thresh
    errors_pck = np.mean(errors_pck, axis=1)
    return np.mean(errors_pck)


def auc(xpts, ypts):
    """
    Calculates the AUC.
    :param xpts: Points on the X axis - the threshold values
    :param ypts: Points on the Y axis - the pck value for that threshold
    :return: The AUC value computed by integrating over pck values for all thresholds
    """
    a = np.min(xpts)
    b = np.max(xpts)
    interpolate = lambda x: np.interp(x, xpts, ypts)
    auc = integrate.quad(interpolate, a, b)[0]
    return auc


def geodesic_distance_mat(mat_pred, mat_gt):
    # Corresponds to 6th metric in our doc
    '''
      https://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf

      input: quat_pred -> predicted quaternion: B x 17 X 4
             quat_gt -> ground truth orientation quaternion: B x 17 X 4
      returns: The geodesic distance in radians between the GT
               and predicted rotations per joint.
    '''
    B, J, dim, _ = mat_pred.shape
    gt_mat_tr = np.swapaxes(mat_gt, 2, 3)
    prod = np.matmul(mat_pred, gt_mat_tr)

    expmap = R.from_matrix(prod.reshape(-1, dim, dim)).as_rotvec().reshape(B, J, 3)

    return np.linalg.norm(expmap, axis=-1, ord=2).mean(axis=0)


def per_axis_mpjae(mat_pred, mat_gt):
    B, J, dim, _ = mat_pred.shape
    eul_pred = R.from_matrix(mat_pred.reshape(-1, 3, 3)).as_euler("ZXY", degrees=True)  # N x 3
    eul_gt = R.from_matrix(mat_gt.reshape(-1, 3, 3)).as_euler("ZXY", degrees=True)

    Rz_pred = R.from_euler("Z", eul_pred[:, 0], degrees=True).as_matrix().reshape(B, J, 3, 3)
    Rx_pred = R.from_euler("X", eul_pred[:, 1], degrees=True).as_matrix().reshape(B, J, 3, 3)
    Ry_pred = R.from_euler("Y", eul_pred[:, 2], degrees=True).as_matrix().reshape(B, J, 3, 3)

    Rz_gt = R.from_euler("Z", eul_gt[:, 0], degrees=True).as_matrix().reshape(B, J, 3, 3)
    Rx_gt = R.from_euler("X", eul_gt[:, 1], degrees=True).as_matrix().reshape(B, J, 3, 3)
    Ry_gt = R.from_euler("Y", eul_gt[:, 2], degrees=True).as_matrix().reshape(B, J, 3, 3)

    x_err = geodesic_distance_mat(Rx_gt, Rx_pred)
    y_err = geodesic_distance_mat(Ry_gt, Ry_pred)
    z_err = geodesic_distance_mat(Rz_gt, Rz_pred)

    return x_err, y_err, z_err


def mean_velocity_error(predicted, target, axis=0):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape

    velocity_predicted = torch.diff(predicted, dim=axis)
    velocity_target = torch.diff(target, dim=axis)

    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=len(target.shape)-1))
