import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _l2_normalize(d):
    d = d.cpu().numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape(
        (-1, 1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)


def _entropy(logits):
    p = F.softmax(logits, dim=1)
    return -torch.mean(torch.sum(p * F.log_softmax(logits, dim=1), dim=1))


def _switch_bn_to_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') == 0:
        m.eval()


def _switch_bn_to_train(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') == 0:
        m.train()


class VAT(object):
    def __init__(self, model, device, eps, xi, k=1, use_entmin=False):
        self.model = model
        self.device = device
        self.xi = xi
        self.eps = eps
        self.k = k
        self.kl_div = nn.KLDivLoss(size_average=False, reduce=False).to(device)
        self.use_entmin = use_entmin

    def __call__(self, X):
        # do *not* update batch statistics
        self.model.apply(_switch_bn_to_eval)

        logits = self.model(X)
        prob_logits = F.softmax(logits.detach(), dim=1)
        d = _l2_normalize(torch.randn(X.size())).to(self.device)

        for ip in range(self.k):
            X_hat = X + d * self.xi
            X_hat.requires_grad = True
            logits_hat = self.model(X_hat)

            adv_distance = torch.mean(self.kl_div(
                F.log_softmax(logits_hat, dim=1), prob_logits).sum(dim=1))
            adv_distance.backward()
            d = _l2_normalize(X_hat.grad).to(self.device)

        logits_hat = self.model(X + self.eps * d)
        LDS = torch.mean(self.kl_div(
            F.log_softmax(logits_hat, dim=1), prob_logits).sum(dim=1))

        if self.use_entmin:
            LDS += _entropy(logits_hat)

        self.model.apply(_switch_bn_to_train)
        return LDS
