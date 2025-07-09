import torch
import torch.nn as nn


# Use CPU if CUDA is not available, otherwise use CUDA
DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
