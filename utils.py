from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import argparse
import matplotlib
import numpy as np
matplotlib.use('agg')
from collections import Iterable
import gc

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 'True', 'T', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'F', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

        
def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):
    def __init__(self, path, int_form=':04d', float_form=':.6f'):
        self.path = path
        self.int_form = int_form
        self.float_form = float_form
        self.width = 0

    def __len__(self):
        try: return len(self.read())
        except: return 0

    def write(self, values):
        if not isinstance(values, Iterable):
            values = [values]
        if self.width == 0:
            self.width = len(values)
        assert self.width == len(values), 'Inconsistent number of items.'
        line = ''
        for v in values:
            if isinstance(v, int):
                line += '{{{}}} '.format(self.int_form).format(v)
            elif isinstance(v, float):
                line += '{{{}}} '.format(self.float_form).format(v)
            elif isinstance(v, str):
                line += '{} '.format(v)
            else:
                raise Exception('Not supported type.')
        with open(self.path, 'a') as f:
            f.write(line[:-1] + '\n')

    def read(self):
        with open(self.path, 'r') as f:
            log = []
            for line in f:
                values = []
                for v in line.split(' '):
                    try:
                        v = float(v)
                    except:
                        pass
                    values.append(v)
                log.append(values)

        return log

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    
    return res[0], correct.squeeze()

def get_values(loader, net, criterion, args, conf_func='mcp', benchmark = 'sc-ood',train_loader=None):
    net.eval()

    with torch.no_grad():
        total_loss = 0
        arr_corr = 0
        arr_logit = np.empty(shape=(0, args.num_classes))
        arr_softmax = np.empty(shape=(0, args.num_classes))
        target_li = np.array([])
        sc_target_li = np.array([])

        if benchmark == 'sc-ood' : 
            for i, sample in enumerate(loader):
                input = sample['data'].cuda()
                target = sample['label'].cuda()
                target_sc = sample['sc_label'].cuda()
                if target.sum() < 0:
                    target = torch.zeros(target.shape[0], dtype=torch.long).cuda()
                output = net(input)
                
                softmax = F.softmax(output, dim=1)

                arr_logit = np.concatenate((arr_logit, output.cpu().numpy()))
                arr_softmax = np.concatenate((arr_softmax, softmax.cpu().numpy()))
                target_li = np.append(target_li, target.cpu())
                sc_target_li = np.append(sc_target_li, target_sc.cpu())

        elif benchmark == 'synthetic':
            for i, (input, target) in enumerate(loader):
                input = input.cuda()
                target = target.cuda()
                
                output = net(input)
                loss = criterion(output, target).cuda()
                total_loss += loss.mean().item()
                
                softmax = F.softmax(output, dim=1)

                arr_logit = np.concatenate((arr_logit, output.cpu().numpy()))
                arr_softmax = np.concatenate((arr_softmax, softmax.cpu().numpy()))
        
        arr_pred = arr_softmax.argmax(axis=1)
        arr_corr = (arr_pred == np.array(target_li)) * 1
        if conf_func == 'mcp':
            confidence = arr_softmax.max(axis=1)
        elif conf_func == 'entropy':
            softmax = F.softmax(torch.Tensor(arr_logit), dim=1)
            log_softmax = F.log_softmax(torch.Tensor(arr_logit), dim=1)
            entropy = softmax * log_softmax
            entropy = -1.0 * entropy.sum(dim=1)
            confidence = np.array(-entropy)
        elif conf_func == 'margin' : 
            softmax = F.softmax(torch.Tensor(arr_logit), dim=1)
            margin, _ = torch.topk(softmax, 2, dim = 1)
            margin = margin[:,0] - margin[:,1]
            confidence = np.array(margin)
        elif conf_func == 'energy' : 
            confidence = 1 * torch.logsumexp(torch.tensor(arr_logit) / 1, dim=1)
            confidence = np.array(confidence)

    return arr_corr, arr_pred, confidence, target_li, sc_target_li

# write logger
def log_record(logger, scores, task):
    li_key = []
    li_value = []
    for key in scores.keys():
        li_key.append(key)
        if len(key) < 7:
            li_key.append('\t\t')
        else:
            li_key.append('\t')

    if len(task)<=10:
        li_key.insert(0, '{0}\t\t\t'.format(task))
    elif 10<len(task)<20:
        li_key.insert(0, '{0}\t\t'.format(task))
    else:
        li_key.insert(0, '{0}\t'.format(task))

    if len(li_key) != 15:
        for i in range(15-len(li_key)):
            li_key.append('')
            li_value.append('')

    logger.write(li_key)
    logger.write(li_value)