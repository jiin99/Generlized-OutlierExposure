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
from dataset_sc_ood import get_dataloader, get_trainloader
from models import resnet, wrn, densenet_bc
import vit
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
                    choices=['mcp', 'entropy', 'margin', 'energy', 'mahal'],
                    type=str,
                    help='confidence function')
parser.add_argument('--save-path', default='/nas/home/jiin9/oe_mixup',
                    type=str,
                    help='save path')
parser.add_argument('--datapath', type=str, default='data')

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
parser.add_argument('--weight', type=float, default=0.5)
parser.add_argument('--learning-rate', '-lr', type=float, default=0.01, help='The initial learning rate.')
parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--cl-type', type=str, choices=['ideal', 'mcp','percentile5','percentile10','3mcp'], help='ideal|euclidean')

args = parser.parse_args()

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }
args_dict = namespace_to_dict(args)

# args.save_path = os.path.join(args.save_path, args.dataset + '_' + args.model + '_' + str(args.weight)+ '_lr_' + str(args.learning_rate) + '_epoch_' + str(args.epochs))
#str(args.weight) + '_' ) #percentile5

# s_path = '/nas/home/jiin9/oe_mixup/EnergyGOE-finetune'
# args.save_path = os.path.join(s_path, args.dataset + '_' + args.model + '_lr_' + str(args.learning_rate) + '_epoch_10')
#import pdb; pdb.set_trace()

# args.save_path = '/nas/home/jiin9/oe_mixup/performance_verification_220802_energy_final/oe-mixup_cifar10_excFalse_wrn_0.1_oe_0.5_ep_10'

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
        net = wrn.WideResNet(40, num_classes, 2, dropRate=0.3).cuda()

    elif args.model == 'res34':
        net = resnet.ResNet34(num_classes=num_classes).cuda()

    elif args.model == 'dense':
        net = nn.DataParallel(densenet_bc.DenseNet3(depth=100,
                        num_classes=num_classes,
                        growth_rate=12,
                        reduction=0.5,
                        bottleneck=True,
                        dropRate=0)).cuda()

    elif args.model == 'vit':
        net = nn.DataParallel(vit.VisionTransformer(
             image_size=(224, 224),
             patch_size=(16, 16),
             emb_dim=768,
             mlp_dim=3072,
             num_heads=12,
             num_layers=12,
             num_classes=num_classes,
             attn_dropout_rate=0.0,
             dropout_rate=0.1,
             contrastive=False,
             timm=True,
             head=None)).cuda()

    elif args.model == 'deit':
        net = nn.DataParallel(vit.VisionTransformer(
             image_size=(224, 224),
             patch_size=(16, 16),
             emb_dim=384,
             mlp_dim=1536,
             num_heads=6,
             num_layers=12,
             num_classes=num_classes,
             attn_dropout_rate=0.0,
             dropout_rate=0.1,
             contrastive=False,
             timm=True,
             head=None)).cuda()


    md_li = {'miscls' : np.empty((0,8),float)}
    ood_li = {i : np.empty((0,8),float) for i  in  out_data}
    for nums, trial in enumerate(['01','02','03','04','05']) : 
        try : 
            if args.pick == 'best' :
                # vit, deit
                if (args.model == 'vit') or (args.model == 'deit'):
                    dir_li=[]
                    for item in sorted(os.listdir(args.save_path)) : 
                        sub_folder = os.path.join(args.save_path, item)
                        if os.path.isdir(sub_folder):
                            dir_li.append(item)

                    state_dict = torch.load('{0}/{1}/checkpoints/ckpt_epoch_best.pth'.format(args.save_path,dir_li[nums]), map_location=torch.device("cpu"))['state_dict']
                    # state_dict = torch.load('{0}/checkpoints/ckpt_epoch_best.pth'.format(args.save_path), map_location=torch.device("cpu"))['state_dict']
                else: 
                    state_dict = torch.load('{0}/trial_{1}/model_best.pth'.format(args.save_path,trial))
            elif args.pick == 'final' : 
                # vit, deit
                if (args.model == 'vit') or (args.model == 'deit'):
                    dir_li=[]
                    for item in sorted(os.listdir(args.save_path)) : 
                        sub_folder = os.path.join(args.save_path, item)
                        if os.path.isdir(sub_folder):
                            dir_li.append(item)

                    state_dict = torch.load('{0}/{1}/checkpoints/ckpt_epoch_current.pth'.format(args.save_path,dir_li[nums]), map_location=torch.device("cpu"))['state_dict']
                else : 
                    state_dict = torch.load('{0}/trial_{1}/model_final.pth'.format(args.save_path,trial))
            elif args.pick == 'best-80':
                dir_li=[]
                for item in sorted(os.listdir(args.save_path)) : 
                    sub_folder = os.path.join(args.save_path, item)
                    if os.path.isdir(sub_folder):
                        dir_li.append(item)
                state_dict = torch.load('{0}/{1}/checkpoints/ckpt_epoch_current.pth'.format(args.save_path,dir_li[nums]), map_location=torch.device("cpu"))['state_dict']

        except: #FileNotFoundError:
            continue
        
        try:
            if args.model == 'vit':
                net.module.load_state_dict(state_dict, strict = False)
            else : 
                net.load_state_dict(state_dict)
        except RuntimeError:
            net.module.load_state_dict(state_dict)

        criterion = nn.CrossEntropyLoss().cuda()
        metric_logger = utils.Logger(os.path.join(
                                    '{0}/trial_{1}'.format(args.save_path,trial), 
                                    '{0}-scores.log'.format(args.conf)))

        ''' Misclassification Detection '''
        test_loader = get_dataloader(benchmark = args.dataset,num_classes = args.num_classes, name = args.dataset, model = args.model)
        train_loader = get_trainloader(args)
        correct, _, scores, _, _ = utils.get_values(test_loader,
                                                     net,
                                                     criterion,args,
                                                     args.conf,
                                                     train_loader = train_loader)
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
    
    df.to_csv(f'{args.save_path}/{args.pick}model_scores_SC_{args.conf}.csv', header = None, index = None)

if __name__ == "__main__":
    main()
