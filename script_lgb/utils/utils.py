import os
import sys
import time
import pickle
import itertools
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from contextlib import contextmanager
import matplotlib.pyplot as plt
from multiprocessing.reduction import ForkingPickler, AbstractReducer


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def reduce_memory(data, verbose = True):
    start_mem = data.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage of dataframe: {:.2f} MB'.format(start_mem))
    for col in data.columns:
        col_type = data[col].dtype
        if col_type != object:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)
    end_mem = data.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage after optimization: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))    
    return data

def save_pickle(obj, filepath):
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])

def load_pickle(filepath):
    max_bytes = 2**31 - 1
    try:
        input_size = os.path.getsize(filepath)
        bytes_in = bytearray(0)
        with open(filepath, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        obj = pickle.loads(bytes_in)
    except:
        return None
    return obj

def save_importances(importances_,save_dir="./", prefix=""):
    mean_gain = importances_[['gain', 'feature']].groupby('feature').mean()
    importances_['mean_gain'] = importances_['feature'].map(mean_gain['gain'])
    importances_.to_csv(os.path.join(save_dir, prefix+'feature_importances.csv'), index=False)
    summary = mean_gain.reset_index().sort_values("gain",ascending=False)
    summary.to_csv(os.path.join(save_dir, prefix+"feature_importance_summary.csv"), index=False)

def _plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(12,12))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return plt

def make_confusion_matrix(cv_preds, y, save_dir="./"):
    unique_y = np.unique(y)
    class_map = dict()
    for i,val in enumerate(unique_y):
        class_map[val] = i    
    y_map = np.zeros((y.shape[0],))
    y_map = np.array([class_map[val] for val in y])
    cnf_matrix = confusion_matrix(y_map, np.argmax(cv_preds,axis=-1))
    np.set_printoptions(precision=2)
    class_names = unique_y
    plot = _plot_confusion_matrix(cnf_matrix, classes=class_names,normalize=True,
                      title='Confusion matrix')
    plot.savefig(os.path.join(save_dir,"confusion_matrix.png"))


def save_list(save_path, array_list):
    with open(save_path,"w") as f:
        for row in array_list:
            f.write(row + "\n")

def create_logging(root_log_dir, filemode, description=""):
    os.makedirs(root_log_dir,exist_ok=True)
    if description is None:
        description = ""
    log_dir = os.path.join(root_log_dir, 
                '{0}_{1}_tmp'
                    .format(datetime.now().strftime("%Y%m%d_%H%M%S"),description)
                )
    os.mkdir(log_dir)
    log_path = os.path.join(log_dir, 'learn.log')
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    #return logging
    return log_dir

def fast_concat(df1, df2):
    assert(len(df1) == len(df2), "df1: {0}, df2:{1}".format(len(df1),len(df2)))
    for col in [c for c in df2.columns if c not in df1.columns]:
        df1[col] = df2[col].values
    return df1

def fast_merge(df1, df2, on):
    if isinstance(on, str):
        tmp = df1[[on]].merge(df2, how="left", on=on)
    elif isinstance(on, list):
        tmp = df1[on].merge(df2, how="left", on=on)
    else:
        raise("on is not valid type :{}".format(on))
    for col in [col for col in df2.columns if col != on]:
        df1[col] = tmp[col].values
    return df1
