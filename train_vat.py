import click
import torch
import torch.cuda
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from dataset import SubsetDataset
from evaluate import evaluate_classifier
from net import AllFCNet
from sampler import InfiniteSampler
from vat import VAT

torch.backends.cudnn.benchmark = True


@click.command()
@click.option('--base_lr', type=float, default=1e-3)
@click.option('--num_iterations', type=int, default=50000)
@click.option('--alpha', type=float, default=1.0)
@click.option('--eps', type=float, default=1.0)
@click.option('--xi', type=float, default=10.0)
@click.option('--n_label', type=int, default=100)
@click.option('--n_val', type=int, default=10000)
@click.option('--use_entmin', is_flag=True)
def experiment(base_lr, num_iterations, alpha, eps, xi, n_label, n_val,
               use_entmin):
    device = torch.device('cuda')

    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)  # [-1.0, 1.0]
    base_tfs = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    tfs = transforms.Compose(base_tfs)

    train_all = MNIST('./data/mnist', 'train', transform=tfs, download=True)
    train_l = SubsetDataset(train_all, list(range(n_label)))
    train_ul = SubsetDataset(train_all, list(range(len(train_all) - n_val)))
    test = MNIST('./data/mnist', 'test', transform=tfs, download=True)

    print(len(train_l), len(train_ul))

    batch_size_l = 32
    batch_size_ul = 128

    cls = AllFCNet(10, 1, 28).to(device)
    cls.train()

    optimizer = Adam(list(cls.parameters()), lr=base_lr)

    vat_criterion = VAT(cls, device, eps, xi, use_entmin=use_entmin)

    l_train_iter = iter(DataLoader(train_l, batch_size_l, num_workers=4,
                                   sampler=InfiniteSampler(
                                       len(train_l))))
    ul_train_iter = iter(
        DataLoader(train_ul, batch_size_ul, num_workers=4,
                   sampler=InfiniteSampler(len(train_ul))))
    test_loader = DataLoader(test, 1000, num_workers=4)
    print('Training...')

    for niter in range(1, 1 + num_iterations):
        l_x, l_y = next(l_train_iter)
        l_x, l_y = l_x.to(device), l_y.to(device)
        ul_x, _ = next(ul_train_iter)
        ul_x = ul_x.to(device)

        sup_loss = F.cross_entropy(cls(l_x), l_y)
        unsup_loss = alpha * vat_criterion(ul_x)
        loss = sup_loss + unsup_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if niter % 100 == 0:
            n_err = evaluate_classifier(cls, test_loader, device)
            print('Iter {} Err {:.3} Sup {:.3} Unsup {:.3} LR {}'.format( \
                niter, n_err / len(test), sup_loss.item(), unsup_loss.item(),
                base_lr))


if __name__ == '__main__':
    experiment()
