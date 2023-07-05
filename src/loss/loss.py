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
