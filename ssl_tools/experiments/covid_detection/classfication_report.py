

import lightning as L


from functools import wraps

import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, balanced_accuracy_score


def wrap_zero_div(func):
    @wraps(func)
    def run(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ZeroDivisionError:
            return 0.0
    return run

@wrap_zero_div
# sensitivity
def recall_score(tn, fp, fn, tp):
    return float(tp / (tp+fn))

@wrap_zero_div
def specificity_score(tn, fp, fn, tp):
    return float(tn / (tn+fp))

@wrap_zero_div
# ppv
def precision_score(tn, fp, fn, tp):
    return float(tp / (tp+fp))

@wrap_zero_div
# npv
def negative_precision_score(tn, fp, fn, tp):
    return float(tn / (tn+fn))

@wrap_zero_div
def accuracy_score(tn, fp, fn, tp):
    return float(tp+tn/(tp+tn+fp+fn))

@wrap_zero_div
def f1_score(tn, fp, fn, tp):
    precision = precision_score(tn, fp, fn, tp)
    recall = recall_score(tn, fp, fn, tp)
    return float(2 * ( (precision * recall) / (precision + recall) ))

@wrap_zero_div
def uar_score(tn, fp, fn, tp):
    specificity = specificity_score(tn, fp, fn, tp)
    sensitivity = recall_score(tn, fp, fn, tp)
    uar = (specificity + sensitivity)/2.0
    return float(uar)

@wrap_zero_div
def f2_score(tn, fp, fn, tp):
    return fbeta_score(tn, fp, fn, tp, beta=2.0)

def _roc_auc_score(y_true, y_pred, labels):
    try:
        return float(roc_auc_score(y_true, y_pred, labels=labels))
    except Exception as e:
        return None

def _matthews_corrcoef(y_true, y_pred, labels):
    try:
        return float(matthews_corrcoef(y_true, y_pred))
    except Exception as e:
        return None

def _balanced_accuracy_score(y_true, y_pred, labels):
    try:
        return float(balanced_accuracy_score(y_true, y_pred))
    except Exception as e:
        return None

@wrap_zero_div
def fbeta_score(tn, fp, fn, tp, beta=0.1):
    return float(((1+beta**2) * ((tp / (tp+fp)) * (tp / (tp+fn)))) / ((beta**2) * (tp / (tp+fp)) + (tp / (tp+fn))))

@wrap_zero_div
def f2_score(tn, fp, fn, tp):
    x = fbeta_score(tn, fp, fn, tp, beta=2.0)
    return x

def classification_report(y_true, y_pred, labels=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    tn, fp, fn, tp  = int(tn), int(fp), int(fn), int(tp) 
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": recall_score(tn, fp, fn, tp),
        "specificity": specificity_score(tn, fp, fn, tp),
        "precision": precision_score(tn, fp, fn, tp),
        "negative precision": negative_precision_score(tn, fp, fn, tp),
        "f1": f1_score(tn, fp, fn, tp),
        "fbeta": fbeta_score(tn, fp, fn, tp),
        "roc auc": _roc_auc_score(y_true, y_pred, labels=labels),
        "mcc": _matthews_corrcoef(y_true, y_pred, labels=labels),
        "uar": uar_score(tn, fp, fn, tp),
        "accuracy": accuracy_score(tn, fp, fn, tp),
        "balanced_accuracy": _balanced_accuracy_score(y_true, y_pred, labels=labels),
        "f2": f2_score(tn, fp, fn, tp),
        # "y_true": y_true.tolist(),
        # "y_pred": y_pred.tolist()
    }
