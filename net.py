import torch.nn as nn
import torch.nn.functional as F


def call_bn(bn, x, update_batch_stats=True):
    if bn.training is False:
        return bn(x)
    elif not update_batch_stats:
        return F.batch_norm(x, None, None, bn.weight, bn.bias, True,
                            bn.momentum, bn.eps)
    else:
        return bn(x)


# Following this repository
# https://github.com/musyoku/vat
class AllFCNet(nn.Module):
    def __init__(self, n_class, n_ch, res):
        super(AllFCNet, self).__init__()
        self.input_len = n_ch * res * res
        self.fc1 = nn.Linear(self.input_len, 1200)
        self.fc2 = nn.Linear(1200, 600)
        self.fc3 = nn.Linear(600, n_class)

        self.bn_fc1 = nn.BatchNorm1d(1200)
        self.bn_fc2 = nn.BatchNorm1d(600)

    def __call__(self, x, update_batch_stats=True):
        h = F.relu(call_bn(self.bn_fc1, self.fc1(x.view(-1, self.input_len)),
                           update_batch_stats))
        h = F.relu(call_bn(self.bn_fc2, self.fc2(h), update_batch_stats))
        return self.fc3(h)
