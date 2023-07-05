import logging
import torch.nn as nn


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return 'model size: {:.3f}MB'.format(size_all_mb)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_layers(model):
    for name, p in model.named_parameters():
        if p.requires_grad:
            psize_list = list(p.size())
            psize_str = [str(x) for x in psize_list]
            psize_str = ",".join(psize_str)
            logging.info(name + "\t"+psize_str)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

    if isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)