import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
import datetime as dt
from collections import defaultdict

class FoldValidation():
    def __init__(self):
        self.fold_num = 4
        self.random_state = 1000
        self.shuffle_flg = True
        return

    def set_fold_num(self, fold_num):
        assert(isinstance(fold_num, int))
        self.fold_num = fold_num 

    def get_fold_num(self):
        return self.fold_num

    def set_shuffle_flg(self, shuffle_flg):
        assert(isinstance(shuffle_flg, bool))
        self.shuffle_flg = shuffle_flg

    def set_random_state(self, random_state):
        assert(isinstance(random_state, int))
        self.random_state = random_state

    def set_valid_type(self, valid_type):
        assert(isinstance(valid_type, str))
        self.valid_type = valid_type

    def make_splits_per_type(self, x_arr, y_arr, 
            unique_id_col=None,time_col=None,group_col=None):
        """
        valid_typeは以下の中から指定
         - GroupKFold (groupsを指定)
        """
        if self.valid_type == "GroupKFold": 
            """
            self.folds = {}
            self.split_index_list = {}
            for type_name in x_arr["type"].unique():
                type_idx = x_arr.type == type_name
                groups = x_arr.loc[type_idx,group_col]
                self.folds[type_name] = GroupKFold(
                    n_splits = self.fold_num, 
                )
                self.split_index_list[type_name] = \
                    [(trn_, val_) for trn_, val_ 
                        in self.folds[type_name].split(
                            x_arr.loc[type_idx], y_arr.loc[type_idx], groups = groups
                        )]
            """
            self.folds = GroupKFold(
                    n_splits = self.fold_num, 
                )
            groups = x_arr[group_col]
            self.all_split_index_list = \
                [(trn_, val_) for trn_, val_ 
                    in self.folds.split(
                        x_arr, y_arr, groups = groups
                    )]

            self.split_index_list = defaultdict(list)
            for type_name in x_arr["type"].unique():
                for split_idx in range(self.fold_num):
                    trn_, val_ = self.all_split_index_list[split_idx]
                    all_trn_x = x_arr.iloc[trn_]
                    all_val_x = x_arr.iloc[val_]
                    trn_idx = all_trn_x[all_trn_x.type == type_name].index.values
                    val_idx = all_val_x[all_val_x.type == type_name].index.values
                    self.split_index_list[type_name].append((trn_idx, val_idx))
        else:
            raise("Not Implemented Error")

    def make_atom_splits(self, train_atoms, train_pair):
        """
        atom単位でのsplit
        split自体はペア単位のものに合わせる
        ただしtype別にはできないので、複数タイプのものをまとめる
        """
        if self.valid_type == "GroupKFold": 
            self.atom_split_index_list = []
            for fold_idx in range(self.fold_num):
                trn_idx = np.array([],dtype=np.int64)
                val_idx = np.array([],dtype=np.int64)
                for type_name in self.split_index_list.keys():
                    pair_trn_, pair_val_ = self.split_index_list[type_name][fold_idx]
                    #train_type_pair = train_pair[train_pair.type == type_name]
                    trn_molecules = train_pair.iloc[pair_trn_]["molecule_name"].unique()
                    val_molecules = train_pair.iloc[pair_val_]["molecule_name"].unique()
                    trn_ = train_atoms[train_atoms.molecule_name.isin(trn_molecules)].index.values 
                    val_ = train_atoms[train_atoms.molecule_name.isin(val_molecules)].index.values 
                    trn_idx = np.concatenate([trn_idx, trn_])
                    val_idx = np.concatenate([val_idx, val_])
                self.atom_split_index_list.append((trn_idx,val_idx))
        else:
            raise("Not Implemented Error")




    def make_splits(self, x_arr, y_arr, 
            unique_id_col=None,time_col=None,groups=None):
        """
        valid_typeは以下の中から指定
         - StratifiedKFold
         - KFold
         - TimeSplit
         - GroupKFold (groupsを指定)
        """
        if self.valid_type == "TimeSplit":
            assert(time_col is not None)
        if self.valid_type == "GroupKFold":
            assert(groups is not None)
        if self.valid_type == "StratifiedKFold":
            self.folds = StratifiedKFold(
                    n_splits = self.fold_num,
                    shuffle = self.shuffle_flg,
                    random_state = self.random_state
            )
            self.split_index_list = \
                    [(trn_, val_) for trn_, val_ in self.folds.split(x_arr, y_arr)]
        elif self.valid_type == "KFold":
            self.folds = StratifiedKFold(
                    n_splits = self.fold_num,
                    shuffle = self.shuffle_flg,
                    random_state = self.random_state
            )
            self.split_index_list = \
                    [(trn_, val_) for trn_, val_ in self.folds.split(x_arr, y_arr)]
        elif self.valid_type == "GroupKFold":
            self.folds = GroupKFold(
                    n_splits = self.fold_num, 
            )
            self.split_index_list = \
                    [(trn_, val_) for trn_, val_ 
                            in self.folds.split(x_arr, y_arr, groups = groups)]
        #elif self.valid_type == "TimeSplit":
        #    self.split_index_list = []
        #    for split_time in self.split_time_list:
        #        trn_idx = x_arr[x_arr[time_col].dt.date < split_time].index
        #        val_idx = x_arr[x_arr[time_col].dt.date >= split_time].index
        #        self.split_index_list.append((trn_idx, val_idx))       
        else:
            raise("Not Implemented Error")

    def get_split_index(self):
        return self.split_index_list

folding = FoldValidation()

def set_fold(fold_num, fold_type="KFold", random_state=2019, shuffle_flg=True, split_time_list = None):
    folding.set_fold_num(fold_num)
    folding.set_shuffle_flg(shuffle_flg)
    folding.set_random_state(random_state)
    folding.set_valid_type(fold_type)


def make_splits(x_arr, y_arr, unique_id_col=None, time_col=None, group_col=None, mode=None):
    folding.make_splits_per_type(x_arr, y_arr,
            unique_id_col=unique_id_col,time_col=time_col, group_col=group_col)

def make_atom_splits(train_atoms, train_pair):
    folding.make_atom_splits(train_atoms, train_pair)

def get_folding():
    return folding

def get_split_list():
    return folding.split_index_list

def get_fold_num():
    return folding.get_fold_num()

def get_atom_split_list():
    return folding.atom_split_index_list

def __init__():
    return
