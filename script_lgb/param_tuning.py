# coding:utf-8
import warnings
warnings.filterwarnings('ignore')
import gc,os
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm.callback import _format_eval_result
import optuna
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report,mean_absolute_error
from numba import jit
import logging
# original module
from utils import utils, feature_util
import compe_data
import features
import settings
import validation
from metrics import competition_metric

n_trials = 100
n_log_period = 500
n_train_verbose = 500
early_stopping_rounds = 100
n_rounds = 1000

def log_evaluation(logger, period=1, show_stdv=True, level=logging.DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            logger.log(level, '[{}]\t{}'.format(env.iteration+1, result))
    _callback.order = 10
    return _callback

def param_tuning(
        train_df=None, y=None, use_feats=None, objective_metric=None,
        type_name=None, train_type=None, cat_cols=[]):

    def objective(trial):
        trn_data = lgb.Dataset(full_train.loc[trn_],
                               label = y.loc[trn_],
                               categorical_feature = cat_cols
                              )
        val_data = lgb.Dataset(full_train.loc[val_],
                               label = y.loc[val_],
                               categorical_feature = cat_cols
                              )
        max_depth = trial.suggest_int('max_depth', 4, 12)
        num_leaves_ratio = \
                trial.suggest_uniform("num_leaves_ratio", 0.1, 0.9)
        lgb_param = {
                 "metric":"mae",
                 'learning_rate': 0.1,
                 "boosting": "gbdt",
                 #'objective':'binary',
                 'objective':'regression',
                 'metric':'mae',
                 'num_leaves': int((2 ** max_depth - 1) * num_leaves_ratio)+1,
                 #'num_leaves': trial.suggest_int('num_leaves', 30, 500),
                 'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 30, 300), 
                 'max_depth': max_depth,
                 "feature_fraction": trial.suggest_uniform('feature_fraction', 0.3, 1.0),
                 "bagging_fraction": trial.suggest_uniform('bagging_fraction',0.3,1.0),
                 "lambda_l1": trial.suggest_uniform("lambda_l1",0.0, 1.0),
                 "lambda_l2": trial.suggest_uniform("lambda_l2",0.0, 1.0),
                 "bagging_freq": 1,
                 #"max_cat_to_onehot":trial.suggest_categorical('max_cat_to_onehot', [4, 8, 30]),
                 #"max_cat_threshold":trial.suggest_int("max_cat_threshold", 16, 256),
                 #"cat_smooth": trial.suggest_uniform("cat_smooth", 1, 30),
                 "seed": 222,
                 "verbosity": -1,
                 "num_threads": 16,
        }
        num_round = n_rounds
        clf = lgb.train(lgb_param,
                    trn_data,
                    num_round,
                    valid_sets = [trn_data, val_data],
                    verbose_eval=n_train_verbose,
                    early_stopping_rounds = early_stopping_rounds,
                    callbacks = [log_callback],
                    )
        # Time Valid val_から計算
        val_pred = clf.predict(full_train.loc[val_], 
                                num_iteration=clf.best_iteration)
        val_score = objective_metric(y.loc[val_], val_pred)
        logging.info("val score: {0:.5f}".format(val_score))
        #os.system("aws s3 mv {} s3://pao-kaggle/optuna_result/ --quiet".format(dbfile))
        return val_score
    result_path = os.path.join(OPTUNA_DIR,f"oputuna_result_{type_name}.csv")
    if os.path.exists(result_path):
        return
    full_train = train_df[use_feats]
    del train_df
    gc.collect()
    logger = logging.getLogger(__name__)
    log_callback = log_evaluation(logger, period=n_log_period)
    split_idx_list = validation.get_split_list()[train_type]
    # fold training
    trn_, val_ = split_idx_list[0]
    dbfile = os.path.join(OPTUNA_DIR, 'lgb_{}.db'.format(type_name))
    study = optuna.create_study()
    #study = optuna.create_study(storage='sqlite:///{}'.format(dbfile), 
    #        study_name=type_name, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)
    logging.info('Number of finished trials: {}'.format(len(study.trials)))
    logging.info('Best trial:')
    trial = study.best_trial
    logging.info('  Value: {}'.format(trial.value))
    logging.info('  Params: ')
    for key, value in trial.params.items():
        logging.info('    {}: {}'.format(key, value))
    # write csv
    with open(result_path, "w") as f:
        for key, value in trial.params.items():
            f.write(key + "," + str(value) + "\n")
    try:
        os.system("sudo aws s3 cp {} s3://pao-kaggle/ --quiet".format(result_path))
    except:
        return

def execute():
    with utils.timer("load train and test data"):
        train_df = compe_data.read_train()
        test_df = compe_data.read_test()
        st_df = compe_data.read_structures()
    # set validation fold
    validation.set_fold(
            fold_type = "GroupKFold",
            fold_num = 2, #settings.fold_num,
            random_state = 2222,
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
    del test_df
    gc.collect()
    # predict only train feature
    # training
    #with utils.timer("reduce memory"):
    #    train_df = utils.reduce_memory(train_df)
    #    #test_df = utils.reduce_memory(test_df)
    flg_use_scc_meta = True#False
    if flg_use_scc_meta:
        with utils.timer("meta feature training"):
            flg_meta_learn = True
            feature_selection = True
            if flg_meta_learn:
                meta_train_df = pd.DataFrame()
                meta_features = compe_data.read_scalar_coupling_contributions()
                for meta_col in ["fc","sd","pso","dso"]:
                    meta_feat = meta_features[meta_col]
                    logging.info("******************************".format(meta_col))
                    logging.info("******** learning {} *********".format(meta_col))
                    logging.info("******************************".format(meta_col))
                    use_feats = [x for x in train_df.columns 
                                    if not x in features.EXCEPT_FEATURES]
                    train_y = meta_feat#train_df[features.TARGET_COL]
                    # training per type
                    types = train_df["type"].unique()
                    oof = np.zeros(len(train_df))
                    types_clfs = {}
                    types_scores = {}
                    for type_name in types:
                        logging.info("----- training type == {} -----".format(type_name))
                        type_idx = train_df.type == type_name
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
                        param_tuning(
                            train_df = train_df.loc[type_idx],
                            y = train_y.loc[type_idx], 
                            use_feats = select_feats,
                            objective_metric = mean_absolute_error,#competition_metric,
                            train_type = type_name,
                            type_name = meta_col + "_" + type_name,
                            cat_cols = cat_cols,
                        )
                else:
                    meta_train_df = pd.read_pickle("../pickle/meta_train.pkl")
            #train_df = utils.fast_concat(train_df, meta_train_df)
            #test_df = utils.fast_concat(test_df, meta_test_df)

    """
    with utils.timer("training"):
        feature_selection = True
        use_feats = [x for x in train_df.columns 
                        if not x in features.EXCEPT_FEATURES]
        train_y = train_df[features.TARGET_COL]
        # training per type
        types = train_df["type"].unique()
        oof = np.zeros(len(train_df))
        types_clfs = {}
        types_scores = {}
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
                        threshold = 0,
                        reverse = True,
                )
            else:
                type_use_feats = use_feats.copy()
            clfs, importances, val_score, oof_preds = \
                training.train(
                    train_df.loc[type_idx],
                    train_y.loc[type_idx], 
                    use_feats = features.select_type_feats(type_use_feats,type_name),
                    type_name = type_name,
                    #permutation=True,
                    permutation=False, 
                    cat_cols = cat_cols,
                    log_dir = log_dir,
            )
    """

if __name__ == "__main__":
    ROOT_LOG_DIR = "../log/"
    OPTUNA_DIR = "../optuna_result/"
    os.makedirs(OPTUNA_DIR, exist_ok=True)
    log_dir = utils.create_logging(ROOT_LOG_DIR,'w',description="optuna_tuning")
    
    execute()
