from __future__ import print_function

import numpy as np
from sklearn import metrics

import torch
import torchvision.transforms as trn
import torchvision.datasets as dset
from torch.utils.data.sampler import SubsetRandomSampler

from dataloader import out_dist_loader_cifar
from utils import get_values
from tinyimages_80mn_loader import TinyImages_valid
from dataset_sc_ood import get_dataloader, get_trainloader


def get_curve(in_scores, out_scores, stypes=['Baseline']):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()
    for stype in stypes:
        known = in_scores
        novel = out_scores

        known.sort()
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known), np.min(novel)])
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        fp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k + num_n):
            if k == num_k:
                tp[stype][l + 1:] = tp[stype][l]
                fp[stype][l + 1:] = np.arange(fp[stype][l] - 1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l + 1:] = np.arange(tp[stype][l] - 1, -1, -1)
                fp[stype][l + 1:] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l + 1] = tp[stype][l]
                    fp[stype][l + 1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l + 1] = tp[stype][l] - 1
                    fp[stype][l + 1] = fp[stype][l]
        tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
        fpr_at_tpr95[stype] = fp[stype][tpr95_pos] / num_n

    return tp, fp, fpr_at_tpr95


def ood_metrics_cifar100_sc(name, trial, mode, net, criterion, args, stypes=['result'], params=(0,0)):

    print('')
    print('OoD Detection')
    print(f'ID: {args.dataset}')
    print(f"OoD: {name}")
    print('')


    in_loader = get_dataloader(benchmark = args.dataset, num_classes = args.num_classes, name = args.dataset, model = args.model)
    train_loader = get_trainloader(args)
    
    if (name == 'blobs') or (name == 'gaussian-noise') : 
        out_loader = out_dist_loader_cifar(
                                 name,
                                 args.batch_size,
                                 mode,
                                 args)
    else : 
        out_loader = get_dataloader(name = name, model = args.model)
    
    _, id_pred, id_conf, id_ddood, id_scood = get_values(in_loader,
                                    net,
                                    criterion,
                                    args,
                                    conf_func=args.conf,
                                    train_loader = train_loader)
    in_scores = id_conf

    if (name == 'blobs') or (name == 'gaussian-noise') : 
        _, _, out_scores, _, _ = get_values(out_loader,
                                        net,
                                        criterion,
                                        args,
                                        conf_func=args.conf,
                                        benchmark='synthetic',
                                        train_loader = train_loader)
    else : 
        _, ood_pred, ood_conf, ood_ddood, ood_scood = get_values(out_loader,
                                        net,
                                        criterion,
                                        args,
                                        conf_func=args.conf,
                                        train_loader = train_loader)

        pred = np.concatenate([id_pred, ood_pred])
        conf = np.concatenate([id_conf, ood_conf])
        ddood = np.concatenate([id_ddood, ood_ddood])
        scood = np.concatenate([id_scood, ood_scood])
        label = scood
        if (name == 'cifar10') or (name == 'cifar100') : 
            out_scores = ood_conf
        else : 
            out_scores = conf[label == -1]
           
    tp, fp, fpr_at_tpr95 = get_curve(in_scores, out_scores, stypes)
    results = dict()

    for stype in stypes:
        results = dict()

        mtype  = 'TRIAL'
        results[mtype] = trial

        # FPR@95%TPR
        mtype = 'FPR@95%TPR'
        results[mtype] = fpr_at_tpr95[stype] * 100

        # DTERR
        mtype = 'DETEC-ERR'
        results[mtype] = (1 - (.5 * (tp[stype] / tp[stype][0] + 1. - fp[stype] / fp[stype][0]).max())) * 100

        # AUROC
        mtype = 'AUROC'
        tpr = np.concatenate([[1.], tp[stype] / tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype] / fp[stype][0], [0.]])
        results[mtype] = -np.trapz(1. - fpr, tpr)  * 100

        # AUIN
        mtype = 'AUPR-IN'
        denom = tp[stype] + fp[stype]
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp[stype] / denom, [0.]])
        results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])  * 100

        # AUOUT
        mtype = 'AUPR-OUT'
        denom = tp[stype][0] - tp[stype] + fp[stype][0] - fp[stype]
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[stype][0] - fp[stype]) / denom, [.5]])
        results[mtype] = np.trapz(pout[pout_ind], 1. - fpr[pout_ind])  * 100

        
    f1, _, _, _, _ = f1_score(in_scores, out_scores)
    results['F1-SC0RE'] = f1  * 100

    results['SC-ACC'] = ''#acc * 100


    print('-----------------------')
    print(f"* FPR@95%TPR\t{round(results['FPR@95%TPR'], 2)}")
    print(f"* DETEC-ERR\t{round(results['DETEC-ERR'], 2)}")
    print(f"* AUROC\t\t{round(results['AUROC'], 2)}")
    print(f"* AUPR-IN\t{round(results['AUPR-IN'], 2)}")
    print(f"* AUPR-OUT\t{round(results['AUPR-OUT'], 2)}")
    print(f"* F1-SC0RE\t{round(results['F1-SC0RE'], 2)}")
    print('-----------------------')

    return results


def ood_metrics_cifar100_validation(out_data, trial, mode, net, criterion, args, stypes=['result'], params=(0,0)):

    print('')
    print('OoD Detection')
    print(f'ID: {args.dataset}')
    print(f"OoD: {out_data}")
    print('')

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
    train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])

    valid_idx = []
    if args.dataset == 'cifar10':
        valid_data = dset.CIFAR10('./data', train=True, transform=test_transform, download = True)
        with open('./valid_idx_cifar10.txt', 'r') as idxs:
            for idx in idxs:
                valid_idx.append(int(idx))

    elif args.dataset == 'cifar100':
        valid_data = dset.CIFAR100('./data', train=True, transform=test_transform, download = True)
        with open('./valid_idx_cifar100.txt', 'r') as idxs:
            for idx in idxs:
                valid_idx.append(int(idx))

    valid_sampler = SubsetRandomSampler(valid_idx)


    ood_data = TinyImages_valid(transform=trn.Compose(
    [trn.ToTensor(), trn.ToPILImage(),trn.ToTensor(),trn.Normalize(mean, std)]),args = args)

    in_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False,
                                           sampler=valid_sampler,num_workers=4)

    external_loader = torch.utils.data.DataLoader(
    ood_data,
    batch_size=args.batch_size, shuffle=False,
    num_workers=4)

    out_loader = out_dist_loader_cifar(
                                 out_data,
                                 args.batch_size,
                                 mode,
                                 args)
    
    _, _, in_scores, _, _ = get_values(in_loader,
                                    net,
                                    criterion,
                                    args,
                                    conf_func=args.conf,
                                    benchmark='synthetic')

    _, _, out_scores, _, _  = get_values(out_loader,
                                        net,
                                        criterion,
                                        args,
                                        conf_func=args.conf,
                                        benchmark='synthetic')
    
    tp, fp, fpr_at_tpr95 = get_curve(in_scores, out_scores, stypes)
    results = dict()

    for stype in stypes:
        results = dict()

        mtype  = 'TRIAL'
        results[mtype] = trial

        # FPR@95%TPR
        mtype = 'FPR@95%TPR'
        results[mtype] = fpr_at_tpr95[stype] * 100

        # DTERR
        mtype = 'DETEC-ERR'
        results[mtype] = (1 - (.5 * (tp[stype] / tp[stype][0] + 1. - fp[stype] / fp[stype][0]).max())) * 100

        # AUROC
        mtype = 'AUROC'
        tpr = np.concatenate([[1.], tp[stype] / tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype] / fp[stype][0], [0.]])
        results[mtype] = -np.trapz(1. - fpr, tpr)  * 100

        # AUIN
        mtype = 'AUPR-IN'
        denom = tp[stype] + fp[stype]
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp[stype] / denom, [0.]])
        results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])  * 100

        # AUOUT
        mtype = 'AUPR-OUT'
        denom = tp[stype][0] - tp[stype] + fp[stype][0] - fp[stype]
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[stype][0] - fp[stype]) / denom, [.5]])
        results[mtype] = np.trapz(pout[pout_ind], 1. - fpr[pout_ind])  * 100

    f1, _, _, _, _ = f1_score(in_scores, out_scores)
    results['F1-SC0RE'] = f1  * 100

    mtype = ''
    results[''] = ''
    print('-----------------------')
    print(f"* FPR@95%TPR\t{round(results['FPR@95%TPR'], 2)}")
    print(f"* DETEC-ERR\t{round(results['DETEC-ERR'], 2)}")
    print(f"* AUROC\t\t{round(results['AUROC'], 2)}")
    print(f"* AUPR-IN\t{round(results['AUPR-IN'], 2)}")
    print(f"* AUPR-OUT\t{round(results['AUPR-OUT'], 2)}")
    print(f"* F1-SC0RE\t{round(results['F1-SC0RE'], 2)}")
    print('-----------------------')

    
    return results



# f1 score
def f1_score(in_scores, out_scores, pos_label=1):
    in_scores = np.array(in_scores)
    in_label = np.ones(len(in_scores))

    out_scores = np.array(out_scores)
    out_label = np.zeros(len(out_scores))

    conf = np.append(in_scores, out_scores)
    labels = np.append(in_label, out_label)

    if conf.ndim != 1:
        conf_max = np.max(conf, 1)
    else:
        conf_max = conf

    conf_labels = sorted(zip(conf_max, labels),
                          key=lambda x: x[0], reverse=False)
    sorted_conf, sorted_labels = zip(*conf_labels)
    precision, recall, thresholds = metrics.precision_recall_curve(sorted_labels,
                                                                   sorted_conf,
                                                                   pos_label=pos_label)

    idx_thr_05 = np.argmin(np.abs(thresholds - 0.5))
    li_f1_score = np.zeros(len(precision))
    for i in range(len(precision)):
        if precision[i] + recall[i] == 0:
            li_f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-8)
        else:
            li_f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

    f1_score = 2 * (precision[idx_thr_05] * recall[idx_thr_05]) / (precision[idx_thr_05] + recall[idx_thr_05])
    if precision[idx_thr_05] + recall[idx_thr_05] == 0:
        f1_score = 2 * (precision[idx_thr_05] * recall[idx_thr_05]) / (precision[idx_thr_05] + recall[idx_thr_05] + 1e-8)
    # li_f1_score = 2 * (precision * recall) / (precision + recall)
    thresholds = np.append(thresholds, 1.0)

    return f1_score, li_f1_score, thresholds, precision, recall