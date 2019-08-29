import warnings
warnings.filterwarnings('ignore')
import os, sys, time, gc, pickle, argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime as dt
import logging
# original module
from utils import utils
import features
import atom_training
import atom_features
#import prediction
import compe_data
import settings
import validation
import param_tuning

def make_atomic_data(train, test, st_df):
    """
    Structureをベースにしたtrain,testを作成
    """
    train_molecules = train["molecule_name"].unique()
    test_molecules = test["molecule_name"].unique()
    train_df = st_df[st_df.molecule_name.isin(train_molecules)].reset_index(drop=True)
    test_df = st_df[st_df.molecule_name.isin(test_molecules)].reset_index(drop=True)
    return train_df, test_df

def learn():
    with utils.timer("load train and test data"):
        train_pair = compe_data.read_train()
        test_pair = compe_data.read_test()
        st_df = compe_data.read_structures()
    with utils.timer("make atomic data frame"):
        train_df, test_df = make_atomic_data(train_pair, test_pair, st_df)
    # set validation fold
    validation.set_fold(
            fold_type = "GroupKFold",
            fold_num = settings.fold_num,
            random_state = settings.fold_seed,
            shuffle_flg = True,
    )
    # validationのsplitは、本来のターゲットのindexに合わせる
    validation.make_splits(
            train_pair, 
            train_pair[features.TARGET_COL],
            group_col=features.GROUP_COL
    )
    validation.make_atom_splits(train_df, train_pair)
    # make feature
    with utils.timer("feature make"):
        train_df, test_df, cat_cols = atom_features.make(
            train_df,
            test_df,
            train_pair,
            test_pair
        )
    pd.set_option("max_columns",100)
    ## for debug
    train_df.head(10).to_csv(
            os.path.join(log_dir,"check_atom_train_df.csv"),index=False)
    # predict only train feature
    # training
    
    with utils.timer("atomic_meta feature training"):
        meta_train_df = pd.DataFrame()
        meta_test_df = pd.DataFrame()
        meta_features = compe_data.read_mulliken_charges() #仮。変わるかも
        for meta_col in ["mulliken_charge"]:
            meta_feat = meta_features[meta_col]
            logging.info("-------- learning {} ---------".format(meta_col))
            use_feats = [x for x in train_df.columns 
                            if not x in atom_features.EXCEPT_FEATURES]
            train_y = meta_feat
            # training per type
            test_preds = np.zeros(len(test_df))
            clfs, importances, val_score, oof_preds = \
                atom_training.train(
                    train_df,
                    train_y, 
                    use_feats = use_feats,
                    permutation=False, 
                    cat_cols = cat_cols,
                    log_dir = log_dir,
                    target_mode = "meta"
            )
            utils.save_importances(importances_=importances, 
                    save_dir=log_dir, prefix="meta_col_")
            for clf in clfs:
                test_preds += clf.predict(test_df.loc[:, use_feats], 
                                    num_iteration=clf.best_iteration) / len(clfs)
            meta_train_df[meta_col] = oof_preds
            meta_test_df[meta_col] = test_preds
            meta_train_df["molecule_name"] = train_df["molecule_name"]
            meta_train_df["atom_index"] = train_df["atom_index"]
            meta_test_df["molecule_name"] = test_df["molecule_name"]
            meta_test_df["atom_index"] = test_df["atom_index"]
        # merge train and test_df
        meta_train_df.to_pickle("../pickle/atomic_meta_train.pkl")
        meta_test_df.to_pickle("../pickle/atomic_meta_test.pkl")

def finish():
    #学習が全て終わったらディレクトリ名を変更する
    os.rename(log_dir, log_dir[:-4])

parser = argparse.ArgumentParser()
parser.add_argument('-permutation',type=bool, default=False)
parser.add_argument('-debug',type=bool, default=False)
parser.add_argument('-c','--comment', help="use log dir")
args = parser.parse_args()

ROOT_LOG_DIR = "../log/"
log_dir = None

if __name__ == "__main__":
    log_dir = utils.create_logging(ROOT_LOG_DIR,'w',description=args.comment)
    learn()
    finish()

