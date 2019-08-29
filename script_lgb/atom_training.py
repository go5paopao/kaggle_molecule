import os, sys, gc, time, logging
import pandas as pd
import numpy as np
import multiprocessing
import lightgbm as lgb
from lightgbm.callback import _format_eval_result
from sklearn.metrics import confusion_matrix ,mean_absolute_error
from numba import jit
# original module
from utils import utils
from utils.metrics import fast_auc, eval_auc
import compe_data
import features
import settings
import validation
from parameter import lgb_param_atom

def log_evaluation(logger, period=1, show_stdv=True, level=logging.DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            logger.log(level, '[{}]\t{}'.format(env.iteration+1, result))
    _callback.order = 10
    return _callback

def train(
        train_df=None, y=None, use_feats=None, train_type=None,permutation=False,
        cat_cols=[], single_fold=False, log_dir="./", target_mode="target"):
    cat_cols = [c for c in cat_cols if c in use_feats]
    full_train = train_df[use_feats]
    #gc.collect()
    logger = logging.getLogger(__name__)
    log_callback = log_evaluation(logger, period=100)
    clfs = []
    importances = pd.DataFrame()
    oof_preds = np.zeros(len(full_train))
    # save use feature
    utils.save_list(
            os.path.join(log_dir,"atom_train_feats.csv"), 
            full_train.columns.tolist()
    )
    # get validation split index list
    split_idx_list = validation.get_atom_split_list()
    # fold training
    logging.info("------ {} fold learning --------".format(len(split_idx_list)))
    for fold_, (trn_, val_) in enumerate(split_idx_list):
        logging.info("-----------------")
        logging.info("{}th Fold Start".format(fold_+1))
        trn_data = lgb.Dataset(full_train.iloc[trn_],
                               label = y.iloc[trn_],
                               categorical_feature = cat_cols
                              )
        val_data = lgb.Dataset(full_train.iloc[val_],
                               label = y.iloc[val_],
                               categorical_feature = cat_cols
                              )
        logging.info("train shape:({0},{1})".format(full_train.shape[1], len(trn_)))
        num_round = 10000
        clf = lgb.train(lgb_param_atom,
                    trn_data,
                    num_round,
                    valid_sets = [trn_data, val_data],
                    verbose_eval=100,
                    early_stopping_rounds = 100,
                    callbacks = [log_callback],
                    )
        # oof prediction
        oof_preds[val_] = clf.predict(full_train.iloc[val_], num_iteration=clf.best_iteration)
        # permutation importance
        #if permutation:
        #    permutation_importance(clf,val_x,val_y,cv_score,
        #       full_train.columns, suffix=str(fold_))
        # feature importance
        imp_df = pd.DataFrame()
        imp_df['feature'] = full_train.columns
        imp_df['gain'] = clf.feature_importance(importance_type="gain")
        imp_df['fold'] = fold_ + 1
        importances = pd.concat([importances, imp_df], axis=0, sort=False)
        clfs.append(clf)
        if single_fold:
            break
    # total CV value
    cv_score = mean_absolute_error(y, oof_preds)
    logging.info("CV Score {}".format(cv_score))
    # CV Predicitionの対象 = TimeValidのtrain期間
    return clfs, importances, cv_score, oof_preds


