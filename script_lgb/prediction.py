import os,sys,time,gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
import logging
from tqdm import tqdm
import compe_data
import features
import validation
from utils import utils
import settings

def predict(types_clfs, test_df, use_feats_dict, val_score=0, log_dir="./"):
    test_preds = np.zeros(len(test_df))
    types = test_df["type"].unique()
    progress_types = tqdm(types)
    for type_name in progress_types:
        progress_types.set_description("Prediction type: {}".format(type_name))
        clfs = types_clfs[type_name]
        type_idx = test_df.type == type_name
        type_use_feats = use_feats_dict[type_name]
        for clf in clfs:
            test_preds[type_idx] += clf.predict(test_df.loc[type_idx, type_use_feats], 
                                        num_iteration=clf.best_iteration) / len(clfs)
    submit = compe_data.read_sample_submission()
    submit["scalar_coupling_constant"] = test_preds
    submit.to_csv(os.path.join(log_dir,'predict_{0:%Y%m%d%H%M%S}_{1}.csv'
                .format(datetime.now(), val_score, float_format='%.5f')), index=False)
    submit.to_csv('../predict/predict_{0:%Y%m%d%H%M%S}_{1}.csv'
                .format(datetime.now(), val_score, float_format='%.5f'), index=False)

