import pandas as pd
import numpy as np
import os,sys,gc,pickle
import logging
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from utils import utils,feature_util
import compe_data



def calc_rotate_matrix(p):
    """
    p:atom_index_0-1の２点間を結ぶベクトル
    """
    # 回転行列を計算する
    ## X軸
    cos_x = p[:,2] / np.linalg.norm(p[:,[1,2]],axis=1)
    sin_x = p[:,1] / np.linalg.norm(p[:,[1,2]],axis=1)
    ## Y軸
    cos_y = p[:,0] / np.linalg.norm(p[:,[0,2]],axis=1)
    sin_y = p[:,2] / np.linalg.norm(p[:,[0,2]],axis=1)
    ## Z軸
    cos_z = p[:,1] / np.linalg.norm(p[:,[0,1]],axis=1)
    sin_z = p[:,0] / np.linalg.norm(p[:,[0,1]],axis=1)
    n_rows = len(p)
    # 物体座標系の 1->2->3 軸で回転させる
    Rx = np.array([[np.ones(n_rows), np.zeros(n_rows), np.zeros(n_rows)],
                   [np.zeros(n_rows), cos_x, sin_x],
                   [np.zeros(n_rows), -sin_x, cos_x]]).transpose([2,0,1])
    Ry = np.array([[cos_y, np.zeros(n_rows), -sin_y],
                   [np.zeros(n_rows), np.ones(n_rows), np.zeros(n_rows)],
                   [sin_y, np.zeros(n_rows), cos_y]]).transpose([2,0,1])
    Rz = np.array([[cos_z, sin_z, np.zeros(n_rows)],
                   [-sin_z, cos_z, np.zeros(n_rows)],
                   [np.zeros(n_rows), np.zeros(n_rows), np.ones(n_rows)]]).transpose([2,0,1])
    R = np.zeros([p.shape[0],p.shape[1],p.shape[1]])
    # 各行ごとに内積で回転行列を求める
    for ix in tqdm(range(n_rows)):
        R[ix,:,:] = Rz[ix].dot(Ry[ix]).dot(Rx[ix])
    return R

def each_make(df, st_df):
    molecules = df["molecule_name"].unique()
    atoms_vec = df[["x_1","y_1","z_1"]].values - df[["x_0","y_0","z_0"]].values
    r_mat_all = calc_rotate_matrix(atoms_vec)
    for molecule in molecules[:10]:
        st = st_df.loc[st_df.molecule_name == molecule, ["x","y","z"]]
        mol_idx = df.molecule_name == molecule
        mol_df = df[mol_idx]
        mol_r_mat = r_mat_all[mol_idx]
        for ix in range(len(mol_r_mat)):
            r_mat = mol_r_mat[ix]
            rot_arr = np.dot(st,r_mat)
            print(rot_arr)
    return df

def make_st_dict(st_df):
    st_dict = {}
    molecules = st_df["molecule_name"].unique()
    for molecule in tqdm(molecules):
        st_dict[molecule] = st_df[st_df.molecule_name == molecule]
    return st_dict

def map_atom_info(df, st, atom_idx):
    df = pd.merge(df, st, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df


def make_relative_position():
    with utils.timer("read data"):
        train_df = compe_data.read_train()
        test_df = compe_data.read_test()
        st_df = compe_data.read_structures()
    # merge structure to train and test
    with utils.timer("merge atom_st"):
        train_df = map_atom_info(train_df, st_df, 0)
        train_df = map_atom_info(train_df, st_df, 1)
        test_df = map_atom_info(test_df, st_df, 0)
        test_df = map_atom_info(test_df, st_df, 1)
    # make structure dict
    #with utils.timer("make st_dict"):
    #    st_dict = make_st_dict(st_df)
    #    del st_df
    #    gc.collect()
    # train test each make relative position matrix
    train_df = each_make(train_df, st_df)
    test_df = each_make(test_df, st_df)

if __name__ == "__main__":
    make_relative_position()
