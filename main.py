import numpy as np
import os
import argparse
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from models.wrn import WideResNet
from models.densenet_bc import DenseNet3
from models.resnet import ResNet34

from tinyimages_80mn_loader import TinyImages
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import MultiStepLR

import utils

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier with G-OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--model', '-m', type=str, default='wrn',
                    help='Choose architecture.')
parser.add_argument('--exclude', action='store_true', help='tiny imagenet exclude cifar flag.')
parser.add_argument('--exp-type', default= 'oe-mixup',#['baseline', 'oe', 'mixup-oe'],
                    help='experiment methods.')
parser.add_argument('--weight', type=float, default=0.5) 
parser.add_argument('--oe-weight', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=1.0, help='beta distribution param for mixup.') 
parser.add_argument('--beta', type=float, default=1.0, , help='beta distribution param for mixup.') 
parser.add_argument('--strategy', type=str, default='static', help='type of filtering strategy (static or adaptive).') 
parser.add_argument('--estimation-func', type=str, default='msp', help='uncertainty estimation fucntion')
parser.add_argument('--filtered_num', default=20, type=int, help='number of filtering samples, k')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--learning-rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch-size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--oe-batch-size', type=int, default=256, help='Batch size.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers.')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor.')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability.')
# Checkpoints
parser.add_argument('--save', '-s', type=str, 
                    default='./snapshots', 
                     help='Folder to save checkpoints.')
parser.add_argument('--trial', 
                    default='01', type=str)
# Acceleration
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
# GPU
parser.add_argument('--gpu-id', default='0', 
                    type=str, help='gpu number.')
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

torch.manual_seed(1)
np.random.seed(1)

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

class train_data_cifar(Dataset):

    def __init__(self, args = None):
        if args.dataset == 'cifar10':
            self.data = dset.CIFAR10('/data/cifar', train=True, 
                                    transform=train_transform, 
                                    download = True)
        elif args.dataset == 'cifar100':
            self.data = dset.CIFAR100('/data/cifar', train=True, 
                                    transform=train_transform, 
                                    download = True)

    def __getitem__(self, index):
        x, y = self.data[index]
        
        return x, y, index

    def __len__(self):
        return len(self.data)

class test_data_cifar(Dataset):

    def __init__(self, args = None):
        if args.dataset == 'cifar10':
            self.data = dset.CIFAR10('/data/cifar', train=False, 
                                    transform=test_transform, 
                                    download = True)
        elif args.dataset == 'cifar100':
            self.data = dset.CIFAR100('/data/cifar', train=False, 
                                    transform=test_transform, 
                                    download = True)
    def __getitem__(self, index):
        x, y = self.data[index]
        
        return x, y, index

    def __len__(self):
        return len(self.data)

valid_idx = []

if args.dataset == 'cifar10':
    train_data_in = train_data_cifar(args = args)
    test_data = test_data_cifar(args = args)
    num_classes = 10
    with open('./valid_idx_cifar10.txt', 'r') as idxs:
        for idx in idxs:
            valid_idx.append(int(idx))
else:
    train_data_in = train_data_cifar(args = args)
    test_data = test_data_cifar(args = args)
    num_classes = 100
    with open('./valid_idx_cifar100.txt', 'r') as idxs:
        for idx in idxs:
            valid_idx.append(int(idx))

args.num_classes = num_classes

ood_data = TinyImages(
    transform=trn.Compose(
    [trn.ToTensor(), trn.ToPILImage(), trn.RandomCrop(32, padding=4),
    trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)]),
    exclude_cifar = args.exclude, args=args
    )

num_train = len(train_data_in)
indices = list(range(num_train))


train_idx = set(indices) - set(valid_idx)
train_idx = list(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)
train_sampler = SubsetRandomSampler(train_idx)

train_loader_in = torch.utils.data.DataLoader(
    train_data_in,
    batch_size=args.batch_size, shuffle=False, sampler=train_sampler,
    num_workers=args.prefetch)

train_loader_out = torch.utils.data.DataLoader(
    ood_data,
    batch_size=args.oe_batch_size, shuffle=False,
    num_workers=args.prefetch)

test_loader = torch.utils.data.DataLoader(
    train_data_in,
    batch_size=args.batch_size, shuffle=False, sampler=valid_sampler,
    num_workers=args.prefetch)

# Create model
if args.model == 'dense':
    net = nn.DataParallel(DenseNet3(depth=100,
                    num_classes=num_classes,
                    growth_rate=12,
                    reduction=0.5,
                    bottleneck=True,
                    dropRate=0)).cuda()

elif args.model == 'wrn':
    net = WideResNet(args.layers, 
                     num_classes, 
                     args.widen_factor, 
                     dropRate=args.droprate).cuda()
                    
elif args.model == 'res34':
    net = ResNet34(num_classes=num_classes).cuda()

cudnn.benchmark = True  # fire on all cylinders

optimizer = torch.optim.SGD(
    net.parameters(), state['learning_rate'], 
    momentum=state['momentum'],
    weight_decay=state['decay'],
    nesterov=True)

scheduler = MultiStepLR(optimizer,
                        milestones=[120,160],
                        gamma=0.1)

class CrossEntropy(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, preds, target, mask):
        n = preds.size()[-1]
        loss = mask * (target * - F.log_softmax(preds, dim=1)).sum(dim=1)
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        
        return loss

criterion_in = nn.CrossEntropyLoss().cuda()
criterion = CrossEntropy(reduction='mean').cuda()

def mixup(x, idx1, idx2, alpha=1.0, beta = 1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, beta)
    else:
        lam = 1

    batch1 = x[idx1,:]
    batch2 = x[idx2,:]

    mixed_x = lam * batch1 + (1 - lam) * batch2
    
    return mixed_x, lam


def mixup_criterion(criterion, output, y, idx1, idx2, lam, mixed_x_mask):
    label_a = y[idx1]
    label_b = y[idx2]

    return lam * criterion(output, label_a, mixed_x_mask) + (1-lam) * criterion(output, label_b, mixed_x_mask)

def add_loss(criterion, output, targets, out_size, args, mask):
    loss = criterion(output[:targets.shape[0]], targets) + \
            args.oe_weight * -(mask * (output[-out_size:].mean(1) - torch.logsumexp(output[-out_size:], dim=1))).mean()
    
    return loss

def filtering_strategy(strategy, estimation_func, net, inputs_in, out_data, targets, args) :
    net.eval() 
    with torch.no_grad():
        data = torch.cat((inputs_in,out_data),0)
        x = net(data)
        in_softmax = F.softmax(x[:len(targets)],dim=1)
        out_softmax = F.softmax(x[len(targets):],dim=1)

        if estimation_func == 'msp':
            in_value, _ = in_softmax.max(axis=1)
            out_value, _ = out_softmax.max(axis=1)
        
        elif estimation_func == 'margin':
            in_margin, _ = torch.topk(in_softmax, 2, dim = 1)
            in_value = in_margin[:,0] - in_margin[:,1]
            out_margin, _ = torch.topk(out_softmax, 2, dim = 1)
            out_value = out_margin[:,0] - out_margin[:,1]
        
        elif estimation_func == 'entropy':
            log_in_softmax = F.log_softmax(x[:len(targets)], dim=1)
            entropy_in = in_softmax * log_in_softmax
            in_value = -1.0 * entropy_in.sum(dim=1)
            log_out_softmax = F.log_softmax(x[len(targets):], dim=1)
            entropy_out = out_softmax * log_out_softmax
            out_value = -1.0 * entropy_out.sum(dim=1)
            # multiply -1 to use negative of entropy
            in_value = -1 * in_value
            out_value = -1 * out_value
        
        if strategy == 'static' : 
            thread = in_value.sort()[0][:args.filtered_num][-1]
        else : 
            thread = in_value.mean()

        cut_value = out_value.ge(thread)

    return cut_value

# /////////////// Training ///////////////
def train():
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    end = time.time()
    
    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
    for num,(in_set, out_set) in enumerate(zip(train_loader_in, train_loader_out)):
        inputs_in = in_set[0]
        targets = in_set[1]
        out_data = out_set[0].cuda()
        out_size = out_set[0].size(0)
        inputs_in, targets = inputs_in.cuda(), targets.cuda()
        # Apply filtering strategy
        cut_value = filtering_strategy(args.strategy, args.estimation_func, net, inputs_in, out_data, targets, args)
        discard_indices = torch.where(cut_value)[0].cpu()
        discarded_indices = [i + inputs_in.size(0) for i in discard_indices if i < inputs_in.size(0)]
    
        net.train()  # enter train mode

        # Apply Random Mixup
        # get ID and OoD label(unform label for OoD) for Mixup
        true_label = torch.zeros((targets.size(0),args.num_classes), device=targets.device)
        true_label.scatter_(1, targets.data.unsqueeze(1),1)
        uni_label = torch.zeros((out_size,args.num_classes), device=targets.device)
        uni_label.fill_(1/args.num_classes)

        new_batch = torch.cat((inputs_in,out_data[:inputs_in.size(0)]),0)
        new_label = torch.cat((true_label,uni_label[:inputs_in.size(0)]),0)        

        idx1 = torch.randperm(new_batch.size(0))
        idx2 = torch.randperm(new_batch.size(0))
        # Random Mixup
        mixed_x, lam = mixup(new_batch,idx1,idx2,args.alpha,args.beta)

        data = torch.cat((inputs_in,mixed_x,out_data),0).cuda()

        # find the index of discarded samples
        idx1_discard = [torch.where(idx1 == i)[0][0] for i in discarded_indices]
        idx2_discard = [torch.where(idx2 == i)[0][0] for i in discarded_indices]

        # if discarded sample is included, 0 and else, 1
        mixed_x_mask = torch.ones(mixed_x.shape[0])
        mixed_x_mask[[idx1_discard]] = 0
        mixed_x_mask[[idx2_discard]] = 0

        out_mask = torch.ones(out_size)
        out_mask[[discard_indices]] = 0
        
        mixed_x_mask, out_mask = mixed_x_mask.cuda(), out_mask.cuda()
        
        # forward
        x = net(data)
        prec, _ = utils.accuracy(x[:targets.shape[0]], targets)
        
        optimizer.zero_grad()

        loss = args.weight * mixup_criterion(criterion, x[len(targets):len(targets)+mixed_x.size(0)],new_label,idx1,idx2, lam, mixed_x_mask)
        loss += add_loss(criterion_in,x,targets,out_data.size(0),args,out_mask)
        
        # backward
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), in_set[0].size(0))
        top1.update(prec.item(), in_set[0].size(0))

    state['train_loss'] = losses.avg
    train_logger.write([epoch, losses.avg, top1.avg])

# test function
def test():
    criterion = nn.CrossEntropyLoss().cuda()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    net.eval()
    with torch.no_grad():
        for data, targets, idx in test_loader:
            data, targets = data.cuda(), targets.cuda()
            # forward
            output = net(data)
            loss = F.cross_entropy(output, targets)

            prec, _ = utils.accuracy(output, targets)
            losses.update(loss.item(), data.size(0))
            top1.update(prec.item(), data.size(0))

    state['test_loss'] = losses.avg
    state['test_accuracy'] = top1.avg
    test_logger.write([epoch, losses.avg, top1.avg])

    return top1.avg

args.save = os.path.join(args.save, args.exp_type + '_' + \
                        args.dataset + '_' + 'exc' + str(args.exclude) + \
                        '_' + args.model, f'trial_{args.trial}')

# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

train_logger = utils.Logger(os.path.join(args.save, 'train_oe.log'))
test_logger = utils.Logger(os.path.join(args.save, 'test_oe.log'))

print('Beginning Training\n')

# Main loop
best_acc = 0
for epoch in range(1, args.epochs+1):
    state['epoch'] = epoch

    begin_epoch = time.time()
    train()
    acc = test()
    scheduler.step()


    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
        (epoch),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        100 - state['test_accuracy'])
    )
    if acc > best_acc : 
        best_acc = acc
        torch.save(net.state_dict(),
            os.path.join(args.save ,f'model_best.pth'))

torch.save(net.state_dict(),
            os.path.join(args.save ,f'model_final.pth'))

