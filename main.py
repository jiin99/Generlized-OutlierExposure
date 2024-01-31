import numpy as np
import os
import pickle
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
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import MultiStepLR

from tinyimages_80mn_loader import TinyImages
from models import resnet, wrn, densenet_bc


import utils


parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier with GeneralizedOE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', 
                    type=str, choices=['cifar10', 'cifar100'], help='Choose between CIFAR-10, CIFAR-100')
parser.add_argument('--datapath', 
                    type=str, default='data', help='Path of dataset')
parser.add_argument('--model', '-m', 
                    type=str, default='wrn', help='Choose architecture')
parser.add_argument('--exclude', 
                    action='store_true', help='tiny imagenet exclude cifar flag')
parser.add_argument('--exp-type', 
                    choices=['G-OE', 'OE', 'E-OE', 'Baseline'], help='experiment methods')
parser.add_argument('--weight', 
                    type=float, default=0.5) #weight : [0.5, 1.0, 2.0, 5.0]
parser.add_argument('--oe-weight', 
                    type=float, default=0.5)
parser.add_argument('--alpha', 
                    type=float, default=1.0) 
parser.add_argument('--beta', 
                    type=float, default=1.0) 

# Optimization options
parser.add_argument('--epochs', '-e', 
                    type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--learning-rate', '-lr', 
                    type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch-size', '-b', 
                    type=int, default=128, help='Batch size of ID')
parser.add_argument('--oe-batch-size', 
                    type=int, default=256, help='Batch size of OoD')
parser.add_argument('--momentum', 
                    type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', 
                    type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', 
                    default=40, type=int, help='Total number of layers')
parser.add_argument('--widen-factor', 
                    default=2, type=int, help='Widen factor')
parser.add_argument('--droprate', 
                    default=0.3, type=float, help='Dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', 
                    type=str, default='./snapshots-', help='Folder to save checkpoints.')
parser.add_argument('--filtered_num', 
                    default=8, type=int, help='Number of samples to filter out')
parser.add_argument('--trial', 
                    default='01', type=str)
# EG specific
parser.add_argument('--m_in', 
                    type=float, default=-25., help='Margin for in-distribution; above this value will be penalized')
parser.add_argument('--m_out', 
                    type=float, default=-7., help='Margin for out-distribution; below this value will be penalized')

# Acceleration
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
parser.add_argument('--gpu-id', default='0', 
                    type=str, help='gpu number')

args = parser.parse_args()


state = {k: v for k, v in args._get_kwargs()}
print(state)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

torch.manual_seed(1)
np.random.seed(1)

if args.exp_type == 'G-OE':
    args.save = args.save + args.exp_type + '-' + str(args.filtered_num)
else:
    args.save = args.save + args.exp_type 

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]


train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                            trn.ToTensor(), trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
ood_transform = trn.Compose(
[trn.ToTensor(), trn.ToPILImage(), trn.RandomCrop(32, padding=4),
    trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)])

class train_data_cifar(Dataset):
    def __init__(self,args = None):
        if args.dataset == 'cifar10':
            self.data = dset.CIFAR10(f'/{args.datapath}/cifar', train=True, transform=train_transform, download = True)
        elif args.dataset == 'cifar100':
            self.data = dset.CIFAR100(f'/{args.datapath}/cifar', train=True, transform=train_transform, download = True)
    def __getitem__(self, index):
        x, y = self.data[index]

        return x, y, index

    def __len__(self):
        return len(self.data)

class test_data_cifar(Dataset):
    def __init__(self,args = None):
        if args.dataset == 'cifar10':
            self.data = dset.CIFAR10(f'/{args.datapath}/cifar', train=False, transform=test_transform, download = True)
        elif args.dataset == 'cifar100':
            self.data = dset.CIFAR100(f'/{args.datapath}/cifar', train=False, transform=test_transform, download = True)
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
    with open('./data_indices/valid_idx_cifar10.txt', 'r') as idxs:
        for idx in idxs:
            valid_idx.append(int(idx))
else:
    train_data_in = train_data_cifar(args = args)
    test_data = test_data_cifar(args = args)
    num_classes = 100
    with open('./data_indices/valid_idx_cifar100.txt', 'r') as idxs:
        for idx in idxs:
            valid_idx.append(int(idx))
args.num_classes = num_classes
ood_data = TinyImages(transform=ood_transform,exclude_cifar = args.exclude, args=args)

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
    net = nn.DataParallel(densenet_bc.DenseNet3(depth=100,
                    num_classes=num_classes,
                    growth_rate=12,
                    reduction=0.5,
                    bottleneck=True,
                    dropRate=0)).cuda()
elif args.model == 'wrn':
    net = wrn.WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate).cuda()
elif args.model == 'res34':
    net = resnet.ResNet34(num_classes=num_classes).cuda()


cudnn.benchmark = True  # fire on all cylinders
optimizer = torch.optim.SGD(
    net.parameters(), state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler = MultiStepLR(optimizer,
                            milestones=[120,160],
                            gamma=0.1)

class LabelSmoothingCrossEntropy(nn.Module):
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

criterion = LabelSmoothingCrossEntropy(reduction='mean').cuda()

def mixup(x,idx1,idx2, alpha=1.0, beta = 1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, beta)
    else:
        lam = 1

    batch1 = x[idx1,:]
    batch2 = x[idx2,:]

    mixed_x = lam * batch1 + (1 - lam) * batch2
    
    return mixed_x, lam


def mixup_criterion(criterion, output,y,idx1,idx2, lam, mixed_x_mask):
    label_a = y[idx1]
    label_b = y[idx2]

    return lam * criterion(output, label_a, mixed_x_mask) + (1-lam) * criterion(output, label_b, mixed_x_mask)

def get_discard_idx(net, out_data, inputs_in):
    if args.filtered_num > 0 : 
        net.eval() 

        with torch.no_grad():
            x2 = net(out_data)
            out_softmax = F.softmax(x2, dim=1)
            out_mcp,_ = out_softmax.max(axis=1)
            out_margin = -out_mcp
            fitering_value = out_margin.sort()[0][:args.filtered_num][-1]
            cut_margin = out_margin.le(fitering_value)

        discard_idx = torch.where(cut_margin)[0].cpu()
        discard_idx_out = [i + inputs_in.size(0) for i in discard_idx if i < inputs_in.size(0)]
    else :
        discard_idx = []
        discard_idx_out = []

    return discard_idx, discard_idx_out

def make_mixup_input(net,targets,inputs_in,out_data,discard_idx,discard_idx_out):

    net.train()  # enter train mode

    true_label = torch.zeros((targets.size(0),args.num_classes), device=targets.device)
    true_label.scatter_(1, targets.data.unsqueeze(1),1)

    uni_label = torch.zeros((out_data.size(0),args.num_classes), device=targets.device)
    uni_label.fill_(1/args.num_classes)

    new_batch = torch.cat((inputs_in,out_data[:inputs_in.size(0)]),0)
    new_label = torch.cat((true_label,uni_label[:inputs_in.size(0)]),0)
    
    idx1 = torch.randperm(new_batch.size(0))
    idx2 = torch.randperm(new_batch.size(0))

    mixed_x, lam = mixup(new_batch,idx1,idx2,args.alpha,args.beta)
    data = torch.cat((inputs_in,mixed_x,out_data),0).cuda()

    idx1_discard = [torch.where(idx1 == i)[0][0] for i in discard_idx_out]
    idx2_discard = [torch.where(idx2 == i)[0][0] for i in discard_idx_out]

    mixed_x_mask = torch.ones(mixed_x.shape[0])

    mixed_x_mask[[idx1_discard]] = 0
    mixed_x_mask[[idx2_discard]] = 0

    out_mask = torch.ones(out_data.size(0))
    out_mask[[discard_idx]] = 0
    
    mixed_x_mask, out_mask = mixed_x_mask.cuda(), out_mask.cuda()

    return mixed_x, mixed_x_mask, out_mask, data, new_label ,idx1 ,idx2, lam


# /////////////// Training ///////////////
def train():
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    end = time.time()
    loss_avg = 0.0
    correct = 0
    if args.exp_type != 'G-OE':
        net.train()
    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
    for in_set, out_set in zip(train_loader_in, train_loader_out):

        inputs_in = in_set[0]
        targets = in_set[1]
        out_data = out_set[0]
        out_size = out_data.size(0)

        ind = out_set[2].cpu().numpy().transpose()
        data = torch.cat((inputs_in, out_data), 0)
        inputs_in, out_data, targets, data = inputs_in.cuda(), out_data.cuda(), targets.cuda(), data.cuda()
        
        if args.exp_type == 'G-OE':
            # get filtering idxs
            discard_idx, discard_idx_out = get_discard_idx(net, out_data, inputs_in)
            # apply mixup
            mixed_x, mixed_x_mask, out_mask, data, new_label ,idx1 ,idx2, lam = \
                    make_mixup_input(net, targets, inputs_in, out_data, discard_idx, discard_idx_out)
       
        # forward
        x = net(data)
        prec, _ = utils.accuracy(x[:targets.shape[0]], targets)
        # backward
        optimizer.zero_grad()
        loss = F.cross_entropy(x[:len(inputs_in)], targets)
        # cross-entropy from softmax distribution to uniform distribution
        if args.exp_type == 'G-OE': 
            loss += args.weight * mixup_criterion(criterion, x[len(targets):len(targets)+mixed_x.size(0)], \
                                                    new_label, idx1, idx2, lam, mixed_x_mask)
            loss += args.oe_weight * -(out_mask * (x[-out_size:].mean(1) - torch.logsumexp(x[-out_size:], dim=1))).mean()

        elif args.exp_type == 'OE':
            loss += args.oe_weight * -(x[len(inputs_in):].mean(1) - torch.logsumexp(x[len(inputs_in):], dim=1)).mean() 
        
        elif args.exp_type == 'E-OE':
            Ec_out = -torch.logsumexp(x[len(inputs_in):], dim=1)
            Ec_in = -torch.logsumexp(x[:len(inputs_in)], dim=1)
            loss += 0.1*(torch.pow(F.relu(Ec_in-args.m_in), 2).mean() + torch.pow(F.relu(args.m_out-Ec_out), 2).mean())
        
        loss.backward()
        optimizer.step()
       
        losses.update(loss.item(), in_set[0].size(0))
        top1.update(prec.item(), in_set[0].size(0))

    state['train_loss'] = losses.avg
    train_logger.write([epoch, losses.avg, top1.avg])

# test function
def test():
    criterion = nn.CrossEntropyLoss().cuda()
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, targets, idx in test_loader:
            data, targets = data.cuda(), targets.cuda()
            # forward
            output = net(data)
            loss = F.cross_entropy(output, targets)

            # test loss average
            loss_avg += float(loss.data)

            prec, _ = utils.accuracy(output, targets)
            losses.update(loss.item(), data.size(0))
            top1.update(prec.item(), data.size(0))


    state['test_loss'] = losses.avg
    state['test_accuracy'] = top1.avg
    test_logger.write([epoch, losses.avg, top1.avg])
    return top1.avg

args.save = os.path.join(args.save, \
                        args.exp_type + '_' + \
                        args.dataset + '_' + \
                        'exc' + str(args.exclude) + '_'\
                        + args.model + '_' \
                        + str(args.weight)+ '_oe_'\
                        + str(args.oe_weight) + '_'\
                        + f'trial_{args.trial}')

# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

train_logger = utils.Logger(os.path.join(args.save, 'train_oe.log'))
test_logger = utils.Logger(os.path.join(args.save, 'test_oe.log'))
epoch_logger = utils.Logger(os.path.join(args.save , 'best_epoch.log'))

print('Beginning Training\n')

# Main loop
best_acc = 0
start_epoch = 1

for epoch in range(start_epoch, args.epochs+start_epoch):
    state['epoch'] = epoch

    begin_epoch = time.time()
    train()
    acc = test()
    scheduler.step()

    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Accuracy {2:.2f}'.format(
        (epoch),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        state['test_accuracy'])
    )

    if acc > best_acc : 
        best_acc = acc
        epoch_logger.write([epoch, best_acc])
        torch.save(net.state_dict(),
            os.path.join(args.save, f'model_best.pth'))

torch.save(net.state_dict(),
            os.path.join(args.save, f'model_{args.epochs}.pth'))

