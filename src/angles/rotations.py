# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# https://github.com/kamisoel/kinematic_pose_estimation/tree/fd0fa7ce87b8b690e86572b2689604763c283d73

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import torch.nn.functional as F


# PyTorch-backed implementations

def qinverse(q, inplace=False):
    # We assume the quaternion to be normalized
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape)-1)


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def qeuler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)
    x, y, z, w = q[:,0], q[:,1], q[:,2], q[:,3]

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1+epsilon, 1-epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1+epsilon, 1-epsilon))
    elif order == 'zxy':
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1+epsilon, 1-epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1+epsilon, 1-epsilon))
    elif order == 'yxz':
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1+epsilon, 1-epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2*(q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1+epsilon, 1-epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    else:
        raise

    return torch.stack((x, y, z), dim=1).view(original_shape)

# Numpy-backed implementations


def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous()
    r = torch.from_numpy(r).contiguous()
    return qmul(q, r).numpy()


def qrot_np(q, v):
    q = torch.from_numpy(q).contiguous()
    v = torch.from_numpy(v).contiguous()
    return qrot(q, v).numpy()


def qeuler_np(q, order, epsilon=0, use_gpu=False):
    if use_gpu:
        q = torch.from_numpy(q).cuda()
        return qeuler(q, order, epsilon).cpu().numpy()
    else:
        q = torch.from_numpy(q).contiguous()
        return qeuler(q, order, epsilon).numpy()


def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4

    result = q.copy()
    dot_products = np.sum(q[1:]*q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0)%2).astype(bool)
    result[1:][mask] *= -1
    return result


def expmap_to_quaternion(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)

    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5*theta).reshape(-1, 1)
    xyz = 0.5*np.sinc(0.5*theta/np.pi)*e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)


def euler_to_quaternion_zxy(e):
    """
    Convert Euler angles (zxy) to quaternions (x,y,z,w)
    adapted from https://github.com/bunnybunbun37204/euler-to-quaternion-cpp/blob/main/quaternion.cpp
    verified from https://quaternions.online/
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)

    # yaw (Z), roll (X), pitch (Y)
    yaw = e[:, 0]
    roll = e[:, 1]
    pitch = e[:, 2]

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    q = np.hstack((qx.reshape(-1,1), qy.reshape(-1,1), qz.reshape(-1,1), qw.reshape(-1,1)))
    q = q.reshape(original_shape)
    return q


def eul2quat(e):
    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.reshape(-1, 3)

    r = R.from_euler(seq="ZXY", angles=e, degrees=True)
    q = r.as_quat()

    return q.reshape(original_shape)


def euler_to_quaternion(e, order):
    """
    Convert Euler angles to quaternions (w,x,y,z)
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.reshape(-1, 3)

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = np.stack((np.cos(x/2), np.sin(x/2), np.zeros_like(x), np.zeros_like(x)), axis=1)
    ry = np.stack((np.cos(y/2), np.zeros_like(y), np.sin(y/2), np.zeros_like(y)), axis=1)
    rz = np.stack((np.cos(z/2), np.zeros_like(z), np.zeros_like(z), np.sin(z/2)), axis=1)

    result = None
    for coord in order:
        if coord == 'x':
            r = rx
        elif coord == 'y':
            r = ry
        elif coord == 'z':
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul_np(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ['xyz', 'yzx', 'zxy']:
        result *= -1

    return result.reshape(original_shape)


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    assert quat.shape[-1] == 4
    B = quat.shape[0]
    J = quat.shape[1]
    quat = quat.reshape(-1, 4)
    norm_quat = quat
    # norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    x, y, z, w = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, J, 3, 3)

    return rotMat


def quat2mat_np(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    assert quat.shape[-1] == 4
    B = quat.shape[0]
    F = quat.shape[1]

    quat = quat.reshape(-1, 4)
    r = R.from_quat(quat)

    rotMat = r.as_matrix().reshape(B, F, 3, 3)

    return rotMat


def expmap2mat_np(expmap):
    assert expmap.shape[-1] == 3
    B, F, _ = expmap.shape
    r = R.from_rotvec(expmap.reshape(-1, 3))
    mat = r.as_matrix().reshape(B, F, 3, 3)

    return mat


def eul2mat_np(eul):
    """Convert euler to rotation matrix.
    Args:
    eul B x j x 3
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    assert eul.shape[-1] == 3
    F = eul.shape[0]
    J = eul.shape[1]

    eul = eul.reshape(-1, 3)
    r = R.from_euler(seq="ZXY", angles=eul, degrees=True)

    rotMat = r.as_matrix().reshape(F, J, 3, 3)

    return rotMat


def mat2eul_np(rmat):
    """Convert euler to rotation matrix.
    Args:
    eul B x j x 3 x 3
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    assert len(rmat.shape) == 4
    F = rmat.shape[0]
    J = rmat.shape[1]

    rmat = rmat.reshape(-1, 3, 3)
    r = R.from_matrix(rmat)

    eul = r.as_euler('ZXY', degrees=True).reshape(F, J, 3)

    return eul


def normalize_quat(v, tolerance=0.00001):
    # TODO: handle when mag2 is zero
    mag2 = torch.sum(torch.square(v), dim=-1)
    if torch.all(mag2>0.0):
        mag = torch.sqrt(mag2)
        mag = mag.unsqueeze(2).repeat(1,1,4)
        q_nm = v / mag
    return q_nm


def euler_from_quaternion(quat):
        """
        q(x,y,z,w) to e(z,x,y)
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        assert quat.shape[-1] == 4
        B = quat.shape[0]
        J = quat.shape[1]
        quat = quat.reshape(-1, 4)
        x, y, z, w = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
        x_sq = torch.square(x)
        y_sq = torch.square(y)
        z_sq = torch.square(z)

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * ( x_sq + y_sq)
        roll_x = torch.atan2(t0, t1).view(-1, 1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if torch.all(t2 > +1.0) else t2
        t2 = -1.0 if torch.all(t2 < -1.0) else t2
        pitch_y = torch.asin(t2).view(-1, 1)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y_sq + z_sq)
        yaw_z = torch.atan2(t3, t4).view(-1, 1)

        return torch.hstack((yaw_z, roll_x, pitch_y)).view(B, J, 3)  # in radians


def euler_from_quaternion_np(quat):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        assert quat.shape[-1] == 4
        original_shape = list(quat.shape)
        original_shape[-1] = 3
        quat = quat.reshape(-1, 4)
        x, y, z, w = quat[:,0], quat[:,1], quat[:,2], quat[:,3]

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1).reshape(-1, 1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if np.all(t2 > +1.0) else t2
        t2 = -1.0 if np.all(t2 < -1.0) else t2
        pitch_y = np.arcsin(t2).reshape(-1, 1)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4).reshape(-1, 1)

        return np.hstack((yaw_z, roll_x, pitch_y)).reshape(original_shape)  # in radians


def rotmat_to_rot6d(x):
    '''
    from branch gr_edvard
    x: B x J x 3 x 3
    ret: B x J x 6
    '''
    B, J, _, _ = x.shape
    rotmat = x.reshape(-1, 3, 3)
    rot6d = rotmat[:, :, :2].reshape(x.shape[0], -1)
    return rot6d.reshape(B, J, 6)


def rotmat_to_rot6d_np(x):
    '''
    x: J x 3 x 3
    ret: J x 6
    '''
    J, _, _ = x.shape
    rot6d = x[:, :, :2].reshape(x.shape[0], -1)
    return rot6d.reshape(J, 6)


# Borrowed from https://github.com/mkocabas/PARE/blob/5278450e08189dbc25487a28d93c13942182ed6a/pare/utils/geometry.py#L113
def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    B, J, _ = x.shape
    x = x.reshape(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1).reshape(B, J, 3, 3)


def unit_vector1(vector):
    """ Returns the unit vector of the vector.  """
    vector_u = vector / np.linalg.norm(vector, axis=1).reshape(-1,1)
    vector_u[np.isnan(vector_u)] = 0
    return vector_u


def angle_between1(v1, v2):
    '''
    v1: F x d
    V2: F x d
    '''
    v1_u = unit_vector1(v1)
    v2_u = unit_vector1(v2)
    cosine_angle = np.einsum('ij, ij -> i', v1_u, v2_u)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return angle


def rotmat(angle):
    """
    angle: B x 1
    R = [ cos theta -sin theta
          sin theta cos theta ]
    """
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    rotmat = np.stack((cos_theta,-1*sin_theta, sin_theta, cos_theta), axis=1)
    return rotmat
