import torch.nn as nn

def group_weight(weight_group, module,lr):
    group_decay = []
    weight_group.append(dict(params=module.parameters(), lr=lr))
    return weight_group