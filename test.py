import os
import sys
import argparse
sys.path.append("../")
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F

import utils
import metrics_id, metrics_ood
from dataset_sc_ood import get_dataloader

from models.wrn import WideResNet
from models.densenet_bc import DenseNet3
from models.resnet import ResNet34

parser = argparse.ArgumentParser(description='Generalized OE')
parser.add_argument('--batch-size', default=64,
                    type=int,
                    help='batch size')
parser.add_argument('--model', default='res34',
                    type=str,
                    help='architecture choice')
parser.add_argument('--exclude', action='store_true', 
                    help='tiny imagenet exclude cifar flag.')
parser.add_argument('--exp-type', default= 'oe-mixup',
                    help='experiment methods')
parser.add_argument('--dataset', default='cifar',
                    type=str,
                    help='ID dataset')
parser.add_argument('--conf', default='mcp',
                    choices=['mcp', 'entropy', 'margin', 'energy'],
                    type=str,
                    help='confidence function')
parser.add_argument('--save-path', default='/nas/home/jiin9/oe_mixup/snapshots',
                    type=str,
                    help='save path')
parser.add_argument('--print-freq', default=10,
                    type=int,
                    metavar='N',
                    help='print frequency')
parser.add_argument('--gpu-id', default='5',
                    type=str,
                    help='gpu number')
parser.add_argument('--pick', default='best',
                    type=str,
                    help='pick best model or final model')


args = parser.parse_args()

args.save_path = os.path.join(args.save_path, args.exp_type + '_' + args.dataset + '_' + 'exc' + str(args.exclude) + '_' + args.model)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    cudnn.benchmark = True

    if args.dataset == 'cifar10':
        num_classes = 10
        last_epoch = 'final'
        out_data = ['cifar100', 'texture', 'svhn', 'tin', 'lsun', 'places365', 'blobs', 'gaussian-noise']
    elif args.dataset == 'cifar100':
        num_classes = 100
        last_epoch = 'final'
        out_data = ['cifar10', 'texture', 'svhn', 'tin', 'lsun', 'places365', 'blobs', 'gaussian-noise']

    
    args.num_classes = num_classes
    
    if args.model == 'wrn':
        net = WideResNet(40, num_classes, 2, dropRate=0.3).cuda()
    elif args.model == 'res34':
        net = ResNet34(num_classes=num_classes).cuda()
    elif args.model == 'dense':
        net = nn.DataParallel(DenseNet3(depth=100,
                        num_classes=num_classes,
                        growth_rate=12,
                        reduction=0.5,
                        bottleneck=True,
                        dropRate=0)).cuda()


    md_li = {'miscls' : np.empty((0,8),float)}
    ood_li = {i : np.empty((0,8),float) for i  in  out_data}
    for trial in ['01','02','03','04','05'] : 

        try : 
            if args.pick == 'best' :
                state_dict = torch.load('{0}/trial_{1}/model_best.pth'.format(args.save_path,trial))
            elif args.pick == 'final' : 
                state_dict = torch.load('{0}/trial_{1}/model_final.pth'.format(args.save_path,trial))
        except FileNotFoundError:
            continue
        
        try:
            net.load_state_dict(state_dict)
        except RuntimeError:
            net.module.load_state_dict(state_dict)

        criterion = nn.CrossEntropyLoss().cuda()
        metric_logger = utils.Logger(os.path.join(
                                    '{0}/trial_{1}'.format(args.save_path,trial), 
                                    '{0}-scores.log'.format(args.conf)))

        ''' Misclassification Detection '''
        test_loader = get_dataloader(benchmark = args.dataset,num_classes = args.num_classes, name = args.dataset)
        correct, _, scores, _, _ = utils.get_values_sc(test_loader,
                                                     net,
                                                     criterion,args,
                                                     args.conf)
        acc = len(np.where(np.array(correct) == 1.0)[0]) / len(correct)
        print('* ACC\t\t{0}'.format(round(acc * 100, 2)))

        # auroc
        _auroc = metrics_id.auroc(scores, correct)

        # aurc, e-aurc
        conf_corr = sorted(zip(scores, correct), key=lambda x: x[0], reverse=True)
        sorted_conf, sorted_corr = zip(*conf_corr)
        aurc, eaurc = metrics_id.aurc_eaurc(sorted_conf, sorted_corr)

        # aupr error
        _aupr_err = metrics_id.aupr_err(scores, correct)

        # frp@95%tpr
        _fpr95tpr = metrics_id.fpr95tpr(scores, correct)

        # ece
        _ece, _, _ = metrics_id.ece(scores, correct, bins=15)
        print('-----------------------')

        md_scores = {'TRIAL' : trial,
                    'ACC': acc * 100,
                    'AUROC': _auroc * 100,
                    'AURC': aurc * 1000,
                    'E-AURC': eaurc * 1000,
                    'AUPR-ERR': _aupr_err * 100,
                    'FPR@95%TPR': _fpr95tpr * 100,
                    'ECE': _ece * 100}
        md_ = np.array([[trial,
                    acc * 100,
                    _auroc * 100,
                    aurc * 1000,
                    eaurc * 1000,
                     _aupr_err * 100,
                     _fpr95tpr * 100,
                     _ece * 100
                     ]])
        md_li['miscls'] = np.append(md_li['miscls'], md_, axis = 0)


        ''' OoD Detection '''
        for name in out_data: 
            mode = 'test'
            ood_scores = metrics_ood.ood_metrics_cifar100_sc(name,
                                                trial,
                                                mode,
                                                net,
                                                criterion,
                                                args)
            ood_ = np.array([[i for i in ood_scores.values()]])
            ood_li[name] = np.append(ood_li[name], ood_, axis = 0)
    
    mis_row = pd.DataFrame([['TRIAL', 'ACC', 'AUROC', 'AURC', 'E-AURC', 'AUPR-ERR', 'FPR@95%TPR', 'ECE']])
    ood_row = pd.DataFrame([['TRIAL', 'FPR@95%TPR', 'DETEC-ERR', 'AUROC', 'AUPR-IN', 'AUPR-OUT', 'F1-SC0RE']])
    mis_df = pd.DataFrame(md_li['miscls'])
    mis_df = pd.concat([mis_row, mis_df])
    mis_df.insert(0,'type', ['' for i in range(mis_df.shape[0])])
    mis_df.iloc[0,0] = f'miscls-{args.dataset}'

    for name in out_data : 

        ood_li[name] = pd.DataFrame(ood_li[name])
        ood_li[name] = pd.concat([ood_row, ood_li[name]])
        ood_li[name].insert(0,'type', ['' for i in range(ood_li[name].shape[0])])
        ood_li[name].iloc[0,0] = name
    
    df = mis_df
    for name in out_data : 
        df = pd.concat([df, ood_li[name]])
    
    df.to_csv(f'{args.save_path}/{args.pick}model_scores_SC_{args.conf}_delete.csv', header = None, index = None)

if __name__ == "__main__":
    main()