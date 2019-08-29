import warnings
warnings.filterwarnings('ignore')
import os, sys, time, gc, pickle, argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime as dt
import logging
# original module
from utils import utils, feature_util
import training
import features
import prediction
import compe_data
import settings
import validation
import param_tuning
from metrics import competition_metric

def learn():
    with utils.timer("load train and test data"):
        train_df = compe_data.read_train()
        test_df = compe_data.read_test()
        st_df = compe_data.read_structures()
    # set validation fold
    validation.set_fold(
            fold_type = "GroupKFold",
            fold_num = settings.fold_num,
            random_state = settings.fold_seed,
            shuffle_flg = True,
    )
    validation.make_splits(
            train_df, 
            train_df[features.TARGET_COL],
            group_col=features.GROUP_COL
    )
    # make feature
    with utils.timer("feature make"):
        train_df, test_df, cat_cols = features.make(
            train_df,
            test_df,
            st_df
        ) 
    ## for debug
    train_df.head(30).to_csv(
            os.path.join(log_dir,"check_train_df.csv"),index=False)

    # predict only train feature
    # training
    flg_use_scc_meta = True#False
    if flg_use_scc_meta:
        with utils.timer("meta feature training"):
            flg_meta_learn = False#True#True
            feature_selection = True
            if flg_meta_learn:
                meta_train_df = pd.DataFrame()
                meta_test_df = pd.DataFrame()
                meta_features = compe_data.read_scalar_coupling_contributions()
                for meta_col in ["fc","sd","pso","dso"]:
                    meta_feat = meta_features[meta_col]
                    logging.info("******************************")
                    logging.info("******** learning {} *********".format(meta_col))
                    logging.info("******************************")
                    use_feats = [x for x in train_df.columns 
                                    if not x in features.EXCEPT_FEATURES]
                    train_y = meta_feat#train_df[features.TARGET_COL]
                    # training per type
                    types = train_df["type"].unique()
                    oof = np.zeros(len(train_df))
                    test_preds = np.zeros(len(test_df))
                    types_clfs = {}
                    types_scores = {}
                    for type_name in types:
                        logging.info("----- training type == {} -----".format(type_name))
                        type_idx = train_df.type == type_name
                        test_type_idx = test_df.type == type_name
                        if feature_selection:
                            type_use_feats = feature_util.feature_select(
                                    use_feats,
                                    importance_df = pd.read_csv(f"./importance/permu_importance{type_name}.csv",
                                                        names=["feature","importance"]
                                    ),
                                    threshold = -0.05,
                                    reverse = True,
                            )
                        else:
                            type_use_feats = use_feats.copy()
                        select_feats = features.select_type_feats(use_feats,type_name)
                        clfs, importances, val_score, oof_preds = \
                            training.train(
                                train_df.loc[type_idx],
                                train_y.loc[type_idx], 
                                use_feats = select_feats,
                                train_type = type_name,
                                permutation=False, 
                                cat_cols = cat_cols,
                                log_dir = log_dir,
                                target_mode = "meta",
                                meta_col = meta_col
                            )
                        types_clfs[type_name] = clfs
                        types_scores[type_name] = val_score
                        utils.save_importances(importances_=importances, 
                                save_dir=log_dir, prefix=f"{meta_col}_{type_name}")
                        oof[type_idx] = oof_preds
                        for clf in clfs:
                            test_preds[test_type_idx] += \
                                    clf.predict(test_df.loc[test_type_idx, use_feats], 
                                                num_iteration=clf.best_iteration) / len(clfs)
                    #total_cv = training.metric(train_df, oof) 
                    #print("TotalCV = {:.5f}".format(total_cv))
                    meta_train_df[meta_col] = oof
                    meta_test_df[meta_col] = test_preds
                    logging.info("---------- meta {} types val score ----------".format(meta_col))
                    for type_name, score in types_scores.items():
                        logging.info("{0} : {1}".format(type_name, score))
                # merge train and test_df
                meta_train_df.to_pickle("../pickle/meta_train.pkl")
                meta_test_df.to_pickle("../pickle/meta_test.pkl")
            else:
                meta_train_df = pd.read_pickle("../pickle/gnn_meta_train.pkl")
                meta_test_df = pd.read_pickle("../pickle/gnn_meta_test.pkl")
            train_df = utils.fast_concat(train_df, meta_train_df)
            test_df = utils.fast_concat(test_df, meta_test_df)
    #with utils.timer("reduce memory"):
    #    train_df = utils.reduce_memory(train_df)
    #    test_df = utils.reduce_memory(test_df)
    with utils.timer("training"):
        feature_selection = True
        use_feats = [x for x in train_df.columns 
                        if not x in features.EXCEPT_FEATURES]
        train_y = train_df[features.TARGET_COL]
        # training per type
        types = train_df["type"].unique()
        #types = train_df["new_type"].unique() #hinokkiタイプ
        oof = np.zeros(len(train_df))
        types_clfs = {}
        types_scores = {}
        use_feats_dict = {}
        for type_name in types:
            logging.info("----- training type == {} -----".format(type_name))
            type_idx = train_df.type == type_name
            if feature_selection:
                type_use_feats = feature_util.feature_select(
                        use_feats,
                        importance_df = pd.read_csv(f"./importance/permu_importance{type_name}.csv", 
                                        names=["feature","importance"]
                        ),
                        #importance_path = f"./importance/{type_name}feature_importance_summary.csv",
                        threshold = -0.05,
                        reverse = True,
                )
            else:
                type_use_feats = use_feats.copy()
            select_use_feats = features.select_type_feats(type_use_feats, type_name)
            use_feats_dict[type_name] = select_use_feats
            clfs, importances, val_score, oof_preds = \
                training.train(
                    train_df.loc[type_idx],
                    train_y.loc[type_idx], 
                    use_feats = select_use_feats,
                    train_type = type_name,
                    #permutation=True,
                    permutation=False, 
                    cat_cols = cat_cols,
                    log_dir = log_dir,
            )
            types_clfs[type_name] = clfs
            types_scores[type_name] = val_score
            utils.save_importances(importances_=importances, 
                    save_dir=log_dir, prefix=type_name)
            oof[type_idx] = oof_preds
            oof_df = pd.DataFrame({"id":train_df.loc[type_idx,"id"],"oof_preds":oof_preds})
            oof_df.to_csv(os.path.join(log_dir, "oof_{}.csv".format(type_name)), index=False)
        for type_name, score in types_scores.items():
            logging.info("{0} : {1}".format(type_name, score))
        total_cv = competition_metric(train_df, oof) 
        logging.info("TotalCV = {:.5f}".format(total_cv))
    del train_df
    gc.collect()
    # prediction
    with utils.timer("prediction"):
        prediction.predict(
                types_clfs,
                test_df,
                use_feats_dict = use_feats_dict,
                val_score = total_cv,
                log_dir = log_dir
        )
    return total_cv


def finish(score=None):
    #学習が全て終わったらディレクトリ名を変更する
    if score is not None:
        os.rename(log_dir, log_dir[:-4]+"{:.5f}".format(score))
    else:
        os.rename(log_dir, log_dir[:-4])

parser = argparse.ArgumentParser()
#parser.add_argument('-debug',type=bool, default=False)
parser.add_argument('-c','--comment', help="use log dir")
args = parser.parse_args()

ROOT_LOG_DIR = "../log/"
log_dir = None

if __name__ == "__main__":
    log_dir = utils.create_logging(ROOT_LOG_DIR,'w',description=args.comment)
    cv_score = learn()
    #adversarial_validation()
    finish(score=cv_score)

