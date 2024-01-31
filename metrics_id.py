from __future__ import print_function
import torch
import numpy as np
from sklearn import metrics
    
# get tpr and fpr over thresholds
def tpr_fpr(in_conf, out_conf, pos_label):
    in_conf = np.array(in_conf)
    in_label = np.zeros(len(in_conf))

    out_conf = np.array(out_conf)
    out_label = np.ones(len(out_conf))

    conf = np.append(in_conf, out_conf)
    labels = np.append(in_label, out_label)

    if conf.ndim != 1:
        conf_max = np.max(conf, 1)
    else:
        conf_max = conf

    tpr, fpr, thresholds = metrics.roc_curve(labels, conf_max, pos_label=pos_label)

    return tpr, fpr


# fpr@95%tpr
def fpr95tpr(conf, correct):
    confidence = np.array(conf)
    correct = np.array(correct)

    fpr, tpr, thresholds = metrics.roc_curve(correct, confidence)
    idx_tpr_95 = np.argmin(np.abs(tpr - 0.95))
    fpr_at_95_tpr = fpr[idx_tpr_95]

    print(f'* FPR@95%TPR\t{round(fpr_at_95_tpr*100, 2)}')

    return fpr_at_95_tpr


# aupr-in
def aupr_in(conf, correct):
    confidence = np.array(conf)
    correct = np.array(correct)

    aupr_in = metrics.average_precision_score(correct, confidence)

    print(f'* AUPR-IN\t{round(aupr_in*100, 2)}')

    return aupr_in


# aupr-error / aupr-out
def aupr_err(conf, correct):
    confidence = np.array(conf)
    correct = np.array(correct)

    aupr_err = metrics.average_precision_score(-1 * correct + 1,
                                               -1 * confidence)

    print(f'* AUPR-ERR\t{round(aupr_err*100, 2)}')

    return aupr_err


# auroc
def auroc(conf, correct):
    confidence = np.array(conf)
    correct = np.array(correct)

    auroc = metrics.roc_auc_score(correct, confidence)

    print(f'* AUROC\t\t{round(auroc*100, 2)}')

    return auroc


# aurc, e-aurc
def aurc_eaurc(rank_conf, rank_corr):
    li_risk = []
    li_coverage = []
    risk = 0
    for i in range(len(rank_conf)):
        coverage = (i + 1) / len(rank_conf)
        li_coverage.append(coverage)

        if rank_corr[i] == 0:
            risk += 1

        li_risk.append(risk / (i + 1))

    r = li_risk[-1]
    risk_coverage_curve_area = 0
    optimal_risk_area = r + (1 - r) * np.log(1 - r)
    for risk_value in li_risk:
        risk_coverage_curve_area += risk_value * (1 / len(li_risk))

    aurc = risk_coverage_curve_area
    eaurc = risk_coverage_curve_area - optimal_risk_area

    print(f'* AURC\t\t{round(aurc*1000, 2)}')
    print(f'* E-AURC\t{round(eaurc*1000, 2)}')

    return aurc, eaurc


# ece
def ece(conf, corr, bins=15):
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    if not isinstance(conf, torch.Tensor):
        conf = torch.Tensor(conf).clone().detach()

    if not torch.tensor(corr).dtype == torch.bool:
        corr = torch.tensor(corr, dtype=bool)

    ece = torch.zeros(1)

    li_acc = []
    li_count = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = conf.gt(bin_lower.item()) * conf.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0.0:
            accuracy_in_bin = corr[in_bin].float().mean()
            avg_confidence_in_bin = conf[in_bin].mean()
            li_count.append(len(corr[in_bin]))

            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        else:
            li_count.append(0)
            accuracy_in_bin = 0.0

        li_acc.append(accuracy_in_bin)

    print(f'* ECE\t\t{round(ece.item()*100, 2)}')

    return ece.item(), li_acc, li_count
    