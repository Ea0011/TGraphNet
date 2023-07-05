import torch
from angles import quat2mat, rot6d_to_rotmat

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def loss_deviation_identity(predicted_angle, target_angle):
    '''
        Angles can be quaternion or 6d or rotmat (default)
    '''
    assert predicted_angle.shape == target_angle.shape

    dim = predicted_angle.shape[-1]

    if dim == 4:
        predicted_angle = quat2mat(predicted_angle)
        target_angle = quat2mat(target_angle)
    elif dim == 6:
        predicted_angle = rot6d_to_rotmat(predicted_angle)
        target_angle = rot6d_to_rotmat(target_angle)

    predicted_angle = predicted_angle.reshape(-1, 3, 3)
    target_angle = target_angle.reshape(-1, 3, 3)

    num = target_angle.size(0)

    I = torch.eye(3, 3).unsqueeze(0).repeat(num, 1, 1).to(device)
    err = torch.norm(
        I - torch.bmm(predicted_angle, torch.transpose(target_angle, 1, 2)),
        dim=(1, 2)
    )
    return torch.mean(err)


def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    # assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape) - 1))


def mean_velocity_error_train(predicted, target, axis=0):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape

    velocity_predicted = torch.diff(predicted, dim=axis)
    velocity_target = torch.diff(target, dim=axis)

    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=len(target.shape)-1))


def mean_diff_loss(predicted, weight):
    """
    Temporal Consistency Loss (Loss imposed on the 1st derivative of the output sequence)
    """
    dif_seq = predicted[:, 1:, :, :] - predicted[:, :-1, :, :]
    weights_joints = torch.ones_like(dif_seq).to(weight.device)
    weights_mul = weight
    assert weights_mul.shape[0] == weights_joints.shape[-2]

    weights_joints = torch.mul(weights_joints.permute(0, 1, 3, 2),weights_mul).permute(0, 1, 3, 2)
    dif_seq = torch.mean(torch.multiply(weights_joints, torch.square(dif_seq)))

    return dif_seq


def motion_loss(predicted, target, intervals=[8, 12, 16, 24], operator=torch.cross):
    assert predicted.shape == target.shape
    loss = 0
    for itv in intervals:
        pred_encode = operator(predicted[:, :-itv], predicted[:, itv:], dim=3)
        target_encode = operator(target[:, :-itv], target[:, itv:], dim=3)
        loss += torch.mean(torch.abs(pred_encode - target_encode)) / len(intervals)

    return loss