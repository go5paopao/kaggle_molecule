import os,gc
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from abc import ABCMeta, abstractmethod
from pathlib import Path
import logging
from imblearn.over_sampling import SMOTE
from utils import utils

class CreateFeature(metaclass=ABCMeta):
    def __init__(self, reset=False):
        self.save_path = Path("../feature/")
        self.reset = reset
    
    def _check_before_cols(self, train_df, test_df):
        self.train_before_cols = train_df.columns.tolist()
        self.test_before_cols = test_df.columns.tolist()
        
    def _check_after_cols(self, train_df, test_df):
        self.train_after_cols = train_df.columns.tolist()
        self.test_after_cols = test_df.columns.tolist()
        
    def _get_new_cols(self, train_df, test_df):
        train_new_cols = np.setdiff1d(self.train_after_cols, 
                                                    self.train_before_cols)
        test_new_cols = np.setdiff1d(self.test_after_cols, 
                                                    self.test_before_cols)
        assert(len(np.setdiff1d(train_new_cols, test_new_cols))==0)
        return train_new_cols
        
    def _save_feature(self, train_df, test_df):
        new_cols = self._get_new_cols(train_df, test_df)
        self.train_path.parent.mkdir(exist_ok=True)
        self.test_path.parent.mkdir(exist_ok=True)
        utils.save_pickle(train_df[new_cols], self.train_path)
        utils.save_pickle(test_df[new_cols], self.test_path)
        
    def _exists_feature(self):
        class_name = self.__class__.__name__
        self.train_path = self.save_path / "train" / "{}.pkl".format(class_name)
        self.test_path = self.save_path / "test" / "{}.pkl".format(class_name)
        if self.train_path.is_file() and self.test_path.is_file():
            return True
        else:
            return False
        
    def load_and_merge(self, train_df, test_df, start):
        train_feat = utils.load_pickle(self.train_path)
        test_feat = utils.load_pickle(self.test_path)
        # カラム数によってconcat方法かえる（高速化のため）
        if train_df.shape[1] > 100:
            train_df = utils.fast_concat(train_df, train_feat)
            test_df = utils.fast_concat(test_df, test_feat)
        else:
            train_df = pd.concat([train_df, train_feat], axis=1)
            test_df = pd.concat([test_df, test_feat], axis=1)
        logging.info("load complete ... {:.1f}s".format(time.time()-start))
        return train_df, test_df

    def make(self, train_df, test_df, *args):
        # check whether feature has saved or not
        logging.info("******** {} ********".format(self.__class__.__name__))
        if self._exists_feature() and not self.reset:
            logging.info("loading...")
            start = time.time()
            return self.load_and_merge(train_df, test_df, start)
        else:
            logging.info("making...")
            start = time.time()
            # save before cols
            self._check_before_cols(train_df, test_df)
            # feature make 
            train_df, test_df = self.__call__(train_df, test_df, *args)
            # save after cols
            self._check_after_cols(train_df, test_df)
            # save feature
            self._save_feature(train_df, test_df)
            logging.info("make complete ... {:.1f}s".format(time.time()-start))
            return train_df, test_df
    
    @abstractmethod
    def __call__(self):
        raise NotImplementedError()


def feature_select(feature_cols, importance_df, threshold=None, use_num=None, reverse=False):
    """
    特徴量を選択する 
    以下のどちらかで選択する
     - importanceをしきい値で区切る
     - 使う特徴量の数を指定
    thresholdを使うときに、reverseだった場合はしきい値より大きいものをdrop対象にする
    """
    assert((threshold is not None) + (use_num is not None) == 1)
    original_cols_len = len(feature_cols)
    importance_df = importance_df.groupby("feature").mean()
    if threshold is not None:
        # threshold で決める
        if reverse:
            drop_cols = importance_df[importance_df["importance"] >= threshold].index.tolist()
        else:
            drop_cols = importance_df[importance_df["importance"] <= threshold].index.tolist()
    elif use_num is not None:
        # 使う特徴量の数で決める
        if use_num >= len(feature_cols):
            drop_cols = []
        else:
            drop_cols = importance_df.sort_values(ascending=False).iloc[use_num:].index.tolist()
    feature_cols = [c for c in feature_cols if c not in drop_cols]
    logging.info("feature num: {0} => {1}".format(original_cols_len,len(feature_cols)))
    return feature_cols

def get_low_importance_cols(csv_path, threshold=10):
    importance_df = pd.read_csv(csv_path)
    importance_df = importance_df.groupby("feature").mean()
    drop_cols = importance_df[importance_df["gain"] < threshold].index.tolist()
    return drop_cols

def permutation_importance(model,val_x, val_y,cv_score, cols, 
                        metric_func,suffix="", output=True, out_dir="./"):
    feat_scores = {}
    for i, col in enumerate(tqdm(cols)):
        shuffle_x = val_x.copy().as_matrix()
        np.random.shuffle(shuffle_x[:,i])
        score = metric_func(val_y, model.predict(shuffle_x, num_iteration=model.best_iteration))
        feat_scores[col] =  cv_score - score
        if i%10 == 0:
            gc.collect()
    if output:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "permu_importance{}.csv".format(suffix))
        with open(out_path,"w") as f:
            for key,value in feat_scores.items():
               f.write(key + "," + str(value) + "\n")
    
        
def smoteAdataset(Xig_train, yig_train):
    sm=SMOTE(random_state=1)
    Xig_train_res, yig_train_res = sm.fit_sample(Xig_train, yig_train.ravel())
    return Xig_train_res, pd.Series(yig_train_res)


def get_no_use_feature(df, importance_df):
    df_columns = df.columns
    use_features = set(feature_select(df_columns, importance_df, threshold=0, reverse=True))
    no_use_features = [c for c in df.columns if c not in use_features]
    return no_use_features


