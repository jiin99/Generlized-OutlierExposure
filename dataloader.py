import os
import torch
import numpy as np
import torchvision.datasets as dset

from torchvision import transforms
from torch.utils.data import DataLoader
from skimage.filters import gaussian as gblur
from tinyimages_80mn_loader import TinyImages_valid

def out_dist_loader_cifar100(data, batch_size, mode,args, transform=False):

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    stdv = [x / 255 for x in [63.0, 62.1, 66.7]]

    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=mean,
                                            std=stdv)])
    if transform == True:
        test_transform = transforms.Compose(
            [transforms.RandomCrop((32, 32), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                std=stdv)])
    if (mode == 'valid') and (data == '80mn'):
        out_dataset = TinyImages_valid(transform = test_transform, args = args)
    elif data == 'blobs':
        out_dataset = blobs()
    elif data == 'cifar10':
        out_dataset = dset.CIFAR10(root='./data', train=False, transform=test_transform)
        out_dataset.targets = [0]*len(out_dataset.targets)
    elif data == 'cifar100':
        out_dataset = dset.CIFAR100(root='./data', train=False, transform=test_transform)
        out_dataset.targets = [0]*len(out_dataset.targets)
    elif data == 'gaussian-noise' :
        out_dataset = gaussian_noise()

    out_loader = torch.utils.data.DataLoader(out_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=4)
    return out_loader

class gaussian_noise(object):
    def __init__(self, transform = None):
        self.samples = torch.from_numpy(np.float32(np.clip(
        np.random.normal(size=(10000, 3, 32, 32), scale=0.5), -1, 1)))
        self.targets = torch.ones(size=(10000,))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        imgs = self.samples[idx]
        target = self.targets[idx].long()
        if self.transform is not None:
            imgs = self.transform(toPIL(imgs))

        return imgs, target

class blobs(object):
    def __init__(self, transform = None):
        ood_data = np.float32(np.random.binomial(n=1, p=0.7, size=(10000, 32, 32, 3)))
        for i in range(10000):
            ood_data[i] = gblur(ood_data[i], sigma=1.5, multichannel=False)
            ood_data[i][ood_data[i] < 0.75] = 0.0
        self.samples = torch.from_numpy(ood_data.transpose((0, 3, 1, 2))) * 2 - 1
        self.targets = torch.ones(size=(10000,))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        imgs = self.samples[idx]
        target = self.targets[idx].long()
        if self.transform is not None:
            imgs = self.transform(toPIL(imgs))

        return imgs, target