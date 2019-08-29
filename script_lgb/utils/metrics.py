import pandas as pd
import numpy as np
from numba import jit
from sklearn.metrics import f1_score,classification_report

@jit
def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc

def eval_auc(preds, dtrain):
    labels = dtrain.get_label()
    return 'auc', fast_auc(labels, preds), True

def calc_f1score(ans, pred, argmax_skip=False):
    if not argmax_skip:
        pred = pred.argmax(axis=1)
    cv_score = f1_score(ans, pred, average="weighted")
    return cv_score

def eval_f1(preds, dtrain):  
    labels = dtrain.get_label()
    pred_labels = preds.reshape(len(np.unique(labels)),-1).argmax(axis=0)
    f1 = f1_score(labels, pred_labels, average='weighted')
    return ('weightedF1', f1, True) 

