import logging
import random
import traceback
import ast
import io
import os
from typing import List

import numpy as np
import torch
import torchvision.transforms as trn
from PIL import Image, ImageFile
from torchvision.transforms import InterpolationMode

from torch.utils.data import Dataset
from pathlib import Path

from torch.utils.data import DataLoader
import torchvision.transforms as trn
import torchvision.datasets as dset
from torch.utils.data.sampler import SubsetRandomSampler

class BaseDataset(Dataset):
    def __init__(self, psudo_index=-1, skip_broken=False, new_index="next"):
        super(BaseDataset, self).__init__()
        self.psudo_index = -1
        self.skip_broken = skip_broken
        self.new_index = new_index
        if new_index not in ("next", "rand"):
            raise ValueError('new_index not one of ("next", "rand")')

    def __getitem__(self, index):
        # in some pytorch versions, input index will be torch.Tensor
        index = int(index)

        # if sampler produce psudo_index, randomly sample an index, and mark it as psudo
        if index == self.psudo_index:
            index = random.randrange(len(self))
            psudo = 1
        else:
            psudo = 0

        while True:
            try:
                sample = self.getitem(index)
                break
            except Exception as e:
                if self.skip_broken and not isinstance(e, NotImplementedError):
                    if self.new_index == "next":
                        new_index = (index + 1) % len(self)
                    else:
                        new_index = random.randrange(len(self))
                    logging.warn(
                        "skip broken index [{}], use next index [{}]".format(
                            index, new_index
                        )
                    )
                    index = new_index
                else:
                    logging.error("index [{}] broken".format(index))
                    traceback.print_exc()
                    logging.error(e)
                    raise e

        sample["index"] = index
        sample["psudo"] = psudo
        return sample

    def getitem(self, index):
        raise NotImplementedError



# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Convert:
    def __init__(self, mode="RGB"):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
# [mean, std]
dataset_stats = {
    # Training stats
    "cifar10": [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]],
    "cifar100": [[0.507, 0.487, 0.441], [0.267, 0.256, 0.276]],
    "tin": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    "test" : [mean, std]
}


def get_transforms(
    mean: List[float],
    std: List[float],
    stage: str,
    interpolation: str = "bilinear",
    name : str = 'cifar10',
    model : str = None
):
    interpolation_modes = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
    }
    color_mode = "RGB"
    interpolation = interpolation_modes[interpolation]
    if stage == "train":
        if (model == 'vit') or (model == 'deit'):
            if model == 'vit':
                mean =  (0.5, 0.5, 0.5)
                std= (0.5, 0.5, 0.5)
            elif model == 'deit':
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)

            scale = (0.2, 1.)
            return trn.Compose([
                        trn.RandomResizedCrop(size=224, scale=scale),
                        trn.RandomHorizontalFlip(),
                        trn.ToTensor(),
                        trn.Normalize(mean, std),
                        #transforms.RandomErasing(p=0.2)
                    ])
        else :  
            return trn.Compose(
                [
                    Convert(color_mode),
                    trn.Resize(32, interpolation=interpolation),
                    trn.CenterCrop(32),
                    trn.RandomHorizontalFlip(),
                    trn.RandomCrop(32, padding=4),
                    trn.ToTensor(),
                    trn.Normalize(mean, std),
                ]
            )

    elif stage == "test":
        if (model == 'vit') or (model == 'deit'):
            if model == 'vit' :
                mean =  (0.5, 0.5, 0.5)
                std= (0.5, 0.5, 0.5)
            elif model == 'deit':
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)
            scale = (0.2, 1.)
            return trn.Compose([
                    trn.Resize((224,224)),
                    trn.ToTensor(),
                    trn.Normalize(mean, std)
                ])
        else : 
            return trn.Compose(
            [
                # Convert(color_mode),
                trn.Resize((32,32)),
                # trn.CenterCrop(32),
                trn.ToTensor(),
                trn.Normalize(mean, std),
            ]
        )


class ImagenameDataset(BaseDataset):
    def __init__(
        self,
        name,
        stage,
        interpolation,
        imglist,
        root,
        num_classes=10,
        maxlen=None,
        dummy_read=False,
        dummy_size=None,
        model=None,
        **kwargs
    ):
        super(ImagenameDataset, self).__init__(**kwargs)
        self.name = name

        with open(imglist) as imgfile:
            self.imglist = imgfile.readlines()
        self.root = root

        mean, std = dataset_stats[stage] if stage == "test" else dataset_stats[name]
        self.transform_image = get_transforms(mean, std, stage, interpolation, name, model)
        # basic image transformation for online clustering (without augmentations)
        self.transform_aux_image = get_transforms(mean, std, "test", interpolation, name, model)

        self.num_classes = num_classes
        self.maxlen = maxlen
        self.dummy_read = dummy_read
        self.dummy_size = dummy_size
        if dummy_read and dummy_size is None:
            raise ValueError("if dummy_read is True, should provide dummy_size")

        self.cluster_id = np.zeros(len(self.imglist), dtype=int)
        self.cluster_reweight = np.ones(len(self.imglist), dtype=float)

        # use pseudo labels for unlabeled dataset during training
        self.pseudo_label = np.array(-1 * np.ones(len(self.imglist)), dtype=int)
        self.ood_conf = np.ones(len(self.imglist), dtype=float)

    def __len__(self):
        if self.maxlen is None:
            return len(self.imglist)
        else:
            return min(len(self.imglist), self.maxlen)

    def getitem(self, index):
        line = self.imglist[index].strip("\n")
        tokens = line.split(" ", 1)
        image_name, extra_str = tokens[0], tokens[1]
        if self.root != "" and image_name.startswith("/"):
            raise RuntimeError('root not empty but image_name starts with "/"')
        path = os.path.join(self.root, image_name)
        sample = dict()
        sample["image_name"] = image_name
        try:
            if not self.dummy_read:
                with open(path, "rb") as f:
                    content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
            if self.dummy_size is not None:
                sample["data"] = torch.rand(self.dummy_size)
            else:
                image = Image.open(buff).convert("RGB")
                sample["data"] = self.transform_image(image)
                sample["plain_data"] = self.transform_aux_image(image)
            extras = ast.literal_eval(extra_str)
            try:
                for key, value in extras.items():
                    sample[key] = value
            except AttributeError:
                sample["label"] = int(extra_str)
            # Generate Soft Label
            soft_label = torch.Tensor(self.num_classes)
            if sample["label"] < 0:
                soft_label.fill_(1.0 / self.num_classes)
            else:
                soft_label.fill_(0)
                soft_label[sample["label"]] = 1
            sample["soft_label"] = soft_label
            # Deep Clustering Aux Label Assignment for both labeled/unlabeled data
            sample["cluster_id"] = self.cluster_id[index]
            sample["cluster_reweight"] = self.cluster_reweight[index]

            # Deep Clustering Pseudo Label Assignment for unlabeled data
            sample["pseudo_label"] = self.pseudo_label[index]
            soft_pseudo_label = torch.Tensor(len(sample["soft_label"]))
            if sample["pseudo_label"] == -1:
                soft_pseudo_label.fill_(1.0 / len(sample["soft_label"]))
            else:
                soft_pseudo_label.fill_(0.0)
                soft_pseudo_label[sample["pseudo_label"]] = 1.0
            sample["pseudo_softlabel"] = soft_pseudo_label
            sample["ood_conf"] = self.ood_conf[index]
        except Exception as e:
            logging.error("[{}] broken".format(path))
            raise e
        return sample



def get_dataset(
    root_dir: str = "data",
    benchmark: str = "cifar100",
    num_classes: int = 100,
    name: str = "cifar100",
    stage: str = "train",
    interpolation: str = "bilinear",
    model : str = None
):
    # root_dir = Path(root_dir)
    root_dir = Path(os.path.join('/data',root_dir))
    data_dir = "data" / root_dir / "images"
    imglist_dir = "data" / root_dir / "imglist" / f"benchmark_{benchmark}"
    return ImagenameDataset(
        name=name,
        stage=stage,
        interpolation=interpolation,
        imglist=imglist_dir / f"{stage}_{name}.txt",
        root=data_dir,
        num_classes=num_classes,
        model = model,
    )


def get_dataloader(
    root_dir: str = "sc-ood",
    benchmark: str = "cifar100",
    num_classes: int = 100,
    name: str = "cifar100",
    stage: str = "test",
    interpolation: str = "bilinear",
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 4,
    model :str = None
):
    dataset = get_dataset(root_dir, benchmark, num_classes, name, stage, interpolation, model)

    return DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

def get_trainloader(args):

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    if (args.model == 'vit') or (args.model == 'deit'):
        if args.model == 'vit':
            mean =  (0.5, 0.5, 0.5)
            std= (0.5, 0.5, 0.5)
        elif args.model == 'deit':
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        scale = (0.2, 1.)
        train_transform = trn.Compose([
                    trn.RandomResizedCrop(size=224, scale=scale),
                    trn.RandomHorizontalFlip(),
                    trn.ToTensor(),
                    trn.Normalize(mean, std),
                    #transforms.RandomErasing(p=0.2)
                ])
        test_transform = trn.Compose([
                trn.Resize((224,224)),
                trn.ToTensor(),
                trn.Normalize(mean, std)
            ])
    else : 
        train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                                trn.ToTensor(), trn.Normalize(mean, std)])
        test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

    valid_idx = []
    if args.dataset == 'cifar10':
        train_data = dset.CIFAR10(f'/{args.datapath}/cifar', train=True, transform=test_transform, download = True)
        with open('./data_indices/valid_idx_cifar10.txt', 'r') as idxs:
            for idx in idxs:
                valid_idx.append(int(idx))

    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(f'/{args.datapath}/cifar', train=True, transform=test_transform, download = True)
        with open('./data_indices/valid_idx_cifar100.txt', 'r') as idxs:
            for idx in idxs:
                valid_idx.append(int(idx))

    num_train = len(train_data)
    indices = list(range(num_train))

    train_idx = set(indices) - set(valid_idx)
    train_idx = list(train_idx)

    train_sampler = SubsetRandomSampler(train_idx)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False,
                                           sampler=train_sampler,num_workers=4)
    return train_loader