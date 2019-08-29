import os, sys, gc, time, logging
import pandas as pd
import numpy as np
import multiprocessing
import lightgbm as lgb
from lightgbm.callback import _format_eval_result
from sklearn.metrics import confusion_matrix ,mean_absolute_error
from numba import jit
# original module
from utils import utils,feature_util
from utils.metrics import fast_auc, eval_auc, calc_f1score, eval_f1 
import compe_data
import features
import settings
import validation
from metrics import competition_metric
from parameter import lgb_param_meta, lgb_param_target

n_round = 10000
n_verbose = 500
n_log_period = 100

def log_evaluation(logger, period=1, show_stdv=True, level=logging.DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            logger.log(level, '[{}]\t{}'.format(env.iteration+1, result))
    _callback.order = 10
    return _callback

def index_augment(x, y):
    """
    atom_index_0と1を入れ替えたデータを作成する
    """
    logging.info("before: {0}, {1}".format(x.shape, y.shape))
    aug_x = x.copy()
    change_cols0 = [c for c in x.columns if c[-2:] == "_0"]
    change_cols1 = [c for c in x.columns if c[-2:] == "_1"]
    #change_dict = {**{c0:c1 for c0,c1 in zip(change_cols0, change_cols1)} \
    #        , **{c1:c0 for c0,c1 in zip(change_cols0, change_cols1)}}
    #aug_x.rename(columns=change_dict, inplace=True)
    logging.info(len(change_cols0), len(change_cols1))
    val0 = aug_x[change_cols0].copy()
    val1 = aug_x[change_cols1].copy()
    aug_x[change_cols0] = val1
    aug_x[change_cols1] = val0
    new_x = pd.concat([x, aug_x], axis=0)
    new_y = pd.concat([y, y], axis=0)
    logging.info("after: {0}, {1}".format(new_x.shape, new_y.shape))
    return new_x, new_y

def get_param(train_type, meta_col):
    if meta_col is not None: 
        base_params = lgb_param_meta
        param_path = f"../optuna_result/oputuna_result_{meta_col}_{train_type}.csv"
        param_dict = pd.read_csv(param_path,names=["param","val"], index_col=0)["val"].to_dict()
        for key, val in param_dict.items():
            if key in ["max_depth", "min_data_in_leaf"]:
                base_params[key] = int(val)
            elif key != "num_leaves_ratio":
                base_params[key] = val
            else:
                base_params["num_leaves"] = int((2**param_dict["max_depth"]-1)*val )
    else:
        base_params = lgb_param_target
    return base_params


def train(
        train_df=None, y=None, use_feats=None, train_type=None,permutation=False,
        cat_cols=[], single_fold=False, log_dir="./", 
        target_mode="target",meta_col=None):
    cat_cols = [c for c in cat_cols if c in use_feats]
    full_train = train_df[use_feats]
    train_df_for_score = train_df[["type","scalar_coupling_constant"]]
    logger = logging.getLogger(__name__)
    log_callback = log_evaluation(logger, period=n_log_period)
    #if target_mode == "target":
    #    lgb_param = lgb_param_target
    #else:
    #    lgb_param = lgb_param_meta
    lgb_param = get_param(train_type, meta_col)
    clfs = []
    importances = pd.DataFrame()
    oof_preds = np.zeros(len(compe_data.read_train()))#np.zeros(len(full_train))
    oof_preds[:] = np.nan
    # save use feature
    utils.save_list(
            os.path.join(log_dir,"train_feats.csv"), 
            full_train.columns.tolist()
    )
    # get validation split index list
    split_idx_list = validation.get_split_list()[train_type]
    # fold training
    logging.info("------ {} fold learning --------".format(len(split_idx_list)))
    for fold_, (trn_, val_) in enumerate(split_idx_list):
        logging.info("-----------------")
        logging.info("{}th Fold Start".format(fold_+1))
        if 0:#target_mode == "target":
            train_x, train_y = index_augment(full_train.loc[trn_], y.loc[trn_])
        else:
            train_x = full_train.loc[trn_]
            train_y = y.loc[trn_]
        trn_data = lgb.Dataset(train_x,
                               label = train_y,
                               categorical_feature = cat_cols
                              )
        val_data = lgb.Dataset(full_train.loc[val_],
                               label = y.loc[val_],
                               categorical_feature = cat_cols
                              )
        logging.info("train shape:({0},{1})".format(full_train.shape[1], len(trn_)))
        
        clf = lgb.train(lgb_param,
                    trn_data,
                    n_round,
                    valid_sets = [trn_data, val_data],
                    verbose_eval=n_verbose,
                    early_stopping_rounds = 100,
                    #feval = eval_f1,
                    callbacks = [log_callback],
        )
        # oof prediction
        #mini_idx_srs = full_train.reset_index().iloc[full_train.loc[val_]]
        oof_preds[val_] = \
                    clf.predict(full_train.loc[val_], num_iteration=clf.best_iteration)
        # permutation importance
        if permutation:
            fold_cv_score = competition_metric(train_df_for_score.loc[val_], oof_preds[val_])
            feature_util.permutation_importance(
                    model=clf, 
                    val_x=full_train.loc[val_], 
                    val_y=train_df_for_score.loc[val_],
                    cv_score=fold_cv_score,
                    cols=full_train.columns,
                    metric_func=competition_metric,
                    suffix=train_type,
                    output=True,
                    out_dir=log_dir
            )
        # feature importance
        imp_df = pd.DataFrame()
        imp_df['feature'] = full_train.columns
        imp_df['gain'] = clf.feature_importance(importance_type="gain")
        imp_df['fold'] = fold_ + 1
        importances = pd.concat([importances, imp_df], axis=0, sort=False)
        clfs.append(clf)
        if permutation:
            #仮でtrainのidxのoof_predsも0で埋めておく
            oof_preds[trn_] = 0
            break
        if single_fold:
            break
    # oof predsがtype別になる前のサイズなのでnanを除くことでもとのサイズにする
    oof_preds = oof_preds[~np.isnan(oof_preds)]
    # total CV value
    if permutation:
        cv_score = 0
        logging.info("This is permutation importance so cv score skip!!")
    elif target_mode == "target":
        cv_score = competition_metric(train_df_for_score, oof_preds)
        logging.info("CV Score {}".format(cv_score))
    else:
        cv_score = mean_absolute_error(y.values, oof_preds)
        logging.info("CV Score {}".format(cv_score))
    # CV Predicitionの対象 = TimeValidのtrain期間
    return clfs, importances, cv_score, oof_preds


