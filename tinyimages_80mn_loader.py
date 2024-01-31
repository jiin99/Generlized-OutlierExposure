import numpy as np
import torch
import torchvision.transforms as trn
from PIL import Image

class TinyImages(torch.utils.data.Dataset):

    def __init__(self, transform=None, exclude_cifar=True, args=None):

        data_file = open(f'/{args.datapath}/80M-tinyimage/tiny_images.bin', "rb")

        def load_image(idx):
            data_file.seek(idx * 3072)
            data = data_file.read(3072)

            return np.fromstring(data, dtype='uint8').reshape(32, 32, 3, order="F")

        self.load_image = load_image
        self.offset = 0     # offset index
        self.transform = transform
        self.exclude_cifar = args.exclude
        self.val_idxs = []
        print(self.exclude_cifar)
        print(args.dataset)
        with open(f'/{args.datapath}/80M-tinyimage/valid_idx_{args.dataset}.txt', 'r') as idxs:
            for idx in idxs : 
                self.val_idxs.append(int(idx))

        if exclude_cifar:
            self.cifar_idxs = []
            with open(f'/{args.datapath}/80M-tinyimage/80mn_cifar_idxs.txt', 'r') as idxs:
                for idx in idxs:
                    # indices in file take the 80mn database to start at 1, hence "- 1"
                    self.cifar_idxs.append(int(idx) - 1)

            # hash table option
            self.cifar_idxs = set(self.cifar_idxs)
            self.in_cifar = lambda x: x in self.cifar_idxs
        self.val_idxs = set(self.val_idxs)
        self.in_val = lambda x: x in self.val_idxs

        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        stdv = [x / 255 for x in [63.0, 62.1, 66.7]]

        if args.dataset == 'cifar100':
            self.num_classes = 100
        else : 
            self.num_classes = 10

    def __getitem__(self, index):
        index = (index + self.offset) % 79302016
        while self.in_val(index):
            index = np.random.randint(79302017)
        
        if self.exclude_cifar:
            while self.in_cifar(index):
                index = np.random.randint(79302017)

        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)

        return img, 0, index  # 0 is the class


    def __len__(self):
        return 79302017

class TinyImages_valid(torch.utils.data.Dataset):

    def __init__(self, transform=None, args=None):

        data_file = open(f'/{args.datapath}/80M-tinyimage/tiny_images.bin', "rb")

        def load_image(idx):
            data_file.seek(idx * 3072)
            data = data_file.read(3072)
            return np.fromstring(data, dtype='uint8').reshape(32, 32, 3, order="F")

        self.load_image = load_image
        self.offset = 0     
        self.targets = 0
        self.transform = transform
        print(args.dataset)
        self.val_idxs = []
        with open(f'/{args.datapath}/80M-tinyimage/valid_idx_{args.dataset}.txt', 'r') as idxs:
            for idx in idxs : 
                self.val_idxs.append(int(idx))
        idx = self.val_idxs
        
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        stdv = [x / 255 for x in [63.0, 62.1, 66.7]]

        if args.dataset == 'cifar100':
            self.num_classes = 100
        else : 
            self.num_classes = 10
            
        self.di_idx = dict(zip(range(5000),idx))
        

    def __getitem__(self, index):
        sample = dict()
        index = self.di_idx[index]

        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)

        return img, 0 # 0 is the class

    def __len__(self):
        return 5000

if __name__ == "__main__":
    ood_data = TinyImages()