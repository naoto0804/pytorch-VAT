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

torch.backends.cudnn.benchmark = True


@click.command()
@click.option('--base_lr', type=float, default=1e-3)
@click.option('--num_iterations', type=int, default=50000)
@click.option('--n_label', type=int, default=100)
def experiment(base_lr, num_iterations, n_label):
    device = torch.device('cuda')

    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)  # [-1.0, 1.0]
    base_tfs = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    tfs = transforms.Compose(base_tfs)

    train_all = MNIST('./data/mnist', 'train', transform=tfs, download=True)
    train_l = SubsetDataset(train_all, list(range(n_label)))
    test = MNIST('./data/mnist', 'test', transform=tfs, download=True)

    batch_size_l = 32

    cls = AllFCNet(10, 1, 28).to(device)
    cls.train()

    optimizer = Adam(list(cls.parameters()), lr=base_lr)

    l_train_iter = iter(DataLoader(train_l, batch_size_l, num_workers=4,
                                   sampler=InfiniteSampler(
                                       len(train_l))))
    test_loader = DataLoader(test, 1000, num_workers=4)
    print('Training...')

    for niter in range(1, 1 + num_iterations):
        l_x, l_y = next(l_train_iter)
        l_x, l_y = l_x.to(device), l_y.to(device)

        sup_loss = F.cross_entropy(cls(l_x), l_y)

        optimizer.zero_grad()
        sup_loss.backward()
        optimizer.step()

        if niter % 100 == 0:
            n_err = evaluate_classifier(cls, test_loader, device)
            print('Iter {} Err {:.3} Sup {:.3} LR {}'.format( \
                niter, n_err / len(test), sup_loss.item(), base_lr))


if __name__ == '__main__':
    experiment()
