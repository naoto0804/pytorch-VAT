import torch.nn as nn
import torch.nn.functional as F


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

    def __call__(self, x):
        h = F.relu(self.bn_fc1(self.fc1(x.view(-1, self.input_len))))
        h = F.relu(self.bn_fc2(self.fc2(h)))
        return self.fc3(h)


class LeNet(nn.Module):
    def __init__(self, n_class, n_ch, res, use_bn=True):
        super(LeNet, self).__init__()
        self.use_source_extractor = False
        self.conv1 = nn.Conv2d(n_ch, 32, 5)
        self.conv2 = nn.Conv2d(32, 48, 5)
        self.fc_input_len = (((res - 4) // 2 - 4) // 2) ** 2 * 48
        self.fc1 = nn.Linear(self.fc_input_len, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, n_class)
        self.use_bn = use_bn

        self.bn_conv1 = nn.BatchNorm2d(32)
        self.bn_conv2 = nn.BatchNorm2d(48)
        self.bn_fc1 = nn.BatchNorm1d(100)
        self.bn_fc2 = nn.BatchNorm1d(100)

    def __call__(self, x):
        h = self.conv1(x)
        if self.use_bn: h = self.bn_conv1(h)
        h = F.max_pool2d(F.relu(h), 2, stride=2)

        h = self.conv2(h)
        if self.use_bn: h = self.bn_conv2(h)
        h = F.max_pool2d(F.relu(h), 2, stride=2)

        h = self.fc1(h.view(x.size(0), -1))
        if self.use_bn: h = self.bn_fc1(h)
        h = F.relu(h)

        h = self.fc2(h)
        if self.use_bn: h = self.bn_fc2(h)
        h = F.relu(h)

        return self.fc3(h)
