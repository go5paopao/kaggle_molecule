lgb_param_target = {
            'objective':'regression',
            "metric":"mae",
            "verbosity": -1,
            "boosting": "gbdt",
            'learning_rate': 0.2,
            'num_leaves': 128,
            'min_data_in_leaf': 79, 
            'max_depth': 9,
            "bagging_freq": 1,
            "bagging_fraction": 0.9,
            "bagging_seed": 11,
            "lambda_l1": 0.1,
            "lambda_l2": 0.3,
            "feature_fraction": 1.0,
            "seed": 11,
            "num_threads": 4
}
lgb_param_meta = {
            'num_leaves': 40,
            'min_data_in_leaf': 30, 
            'objective':'regression',
            'max_depth': 4,
            "metric":"mae",
            'learning_rate': 0.3,
            "boosting": "gbdt",
            "feature_fraction": 0.8,
            "bagging_freq": 1,
            "bagging_fraction": 0.8 ,
            "bagging_seed": 11,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "seed": 201,
            "verbosity": -1,
            "num_threads": 4,
            #"max_bin": 128,
}

lgb_param_atom = {
            'num_leaves': 60,
            'min_data_in_leaf': 30, 
            'objective':'regression',
            'max_depth': 4,
            "metric":"mae",
            'learning_rate': 0.5,
            "boosting": "gbdt",
            "feature_fraction": 0.8,
            "bagging_freq": 1,
            "bagging_fraction": 0.8 ,
            "bagging_seed": 11,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "seed": 201,
            "verbosity": -1,
            "num_threads": 4
}


