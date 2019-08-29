import os
import sys
import gc
import pickle
import itertools
import numpy as np
import pandas as pd
import logging
import compe_data
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
from numba import jit
from math import sqrt
from utils import utils,feature_util
from utils.feature_util import CreateFeature
import validation

TARGET_COL = "scalar_coupling_constant"
GROUP_COL = "molecule_name"
EXCEPT_FEATURES = [
        "id",
        "type",
        "type_int", # type別学習時は不要
        "molecule_name",
        "atom_index_0","atom_index_1",
        "atom_0","atom_1",
        "scalar_coupling_constant",
        "m_x", "m_y",
        "x_0","y_0","z_0",
        "x_1","y_1","z_1",
        "c_x","c_y","c_z",
        "dist_x","dist_y","dist_z",
        #"rad_0","rad_1",
        "atom_index_2","atom_index_30","atom_index_31","atom_index_32"
]
ALL_TYPES = ['1JHC','2JHC','3JHC','1JHN','2JHN','3JHN','2JHH','3JHH']

def select_type_feats(use_feats,select_type_name):
    """
    指定したtypeのfeatureを抽出する
    以下の３パターンを想定
     - 全部一致
     - 数字が一致
     - 結合原子が一致
    """
    types_dict = {
            '1JHC':['1JHC','1Jxx','xJHC'],
            '2JHC':['2JHC','2Jxx','xJHC'],
            '3JHC':['3JHC','3Jxx','xJHC'],
            '1JHN':['1JHN','1Jxx','xJHN'],
            '2JHN':['2JHN','2Jxx','xJHN'],
            '3JHN':['3JHN','3Jxx','xJHN'],
            '2JHH':['2JHH','2Jxx','xJHH'],
            '3JHH':['3JHH','3Jxx','xJHH'],
    }
    num_types = ['1Jxx','2Jxx', '3Jxx']
    atoms_types = ['xJHC','xJHN','xJHH']
    except_feats = []
    # type別に処理
    for i,types in enumerate([ALL_TYPES,num_types,atoms_types]):
        for type_name in types:
            if type_name == types_dict[select_type_name][i]:
                continue
            except_feats += [c for c in use_feats if c.endswith(type_name)]
    #print("Not use: ",except_feats)
    return [c for c in use_feats if c not in except_feats] 

def replace_concat(r_df):
    # index入れ替えたものとconcatする
    r_df = pd.concat([r_df, r_df.rename(
        columns={"atom_index_0":"atom_index_1","atom_index_1":"atom_index_0"})],
        axis=0,
        sort=True
    ).reset_index(drop=True)
    return r_df



def make_factorize(train_df, test_df, fact_cols):
    for col in fact_cols:
        train_df[col], uniques = pd.factorize(train_df[col], sort=True)
        test_df[col] = uniques.get_indexer(test_df[col])
    return train_df, test_df

class MergeAtomInfo(CreateFeature):
    def structure_preprocess(self, structures):
        # https://www.kaggle.com/adrianoavelar/bond-calculaltion-lb-0-82
        atomic_radius = {'H':0.38, 'C':0.77, 'N':0.75, 'O':0.73, 'F':0.71} # Without fudge factor
        fudge_factor = 0.05
        atomic_radius = {k:v + fudge_factor for k,v in atomic_radius.items()}
        electronegativity = {'H':2.2, 'C':2.55, 'N':3.04, 'O':3.44, 'F':3.98}
        atoms = structures['atom'].values
        atoms_en = [electronegativity[x] for x in atoms]
        atoms_rad = [atomic_radius[x] for x in atoms]
        structures['EN'] = atoms_en
        structures['rad'] = atoms_rad
        i_atom = structures['atom_index'].values
        p = structures[['x', 'y', 'z']].values
        p_compare = p
        m = structures['molecule_name'].values
        m_compare = m
        r = structures['rad'].values
        r_compare = r
        source_row = np.arange(len(structures))
        max_atoms = 28
        bonds = np.zeros((len(structures)+1, max_atoms+1), dtype=np.int8)
        bond_dists = np.zeros((len(structures)+1, max_atoms+1), dtype=np.float32)
        print('Calculating bonds')
        for i in tqdm(range(max_atoms-1)):
            p_compare = np.roll(p_compare, -1, axis=0)
            m_compare = np.roll(m_compare, -1, axis=0)
            r_compare = np.roll(r_compare, -1, axis=0)
            mask = np.where(m == m_compare, 1, 0) #Are we still comparing atoms in the same molecule?
            dists = np.linalg.norm(p - p_compare, axis=1) * mask
            r_bond = r + r_compare
            bond = np.where(np.logical_and(dists > 0.0001, dists < r_bond), 1, 0)
            source_row = source_row
            target_row = source_row + i + 1 #Note: Will be out of bounds of bonds array for some values of i
            target_row = np.where(np.logical_or(target_row > len(structures), mask==0), len(structures), target_row) #If invalid target, write to dummy row
            source_atom = i_atom
            target_atom = i_atom + i + 1 #Note: Will be out of bounds of bonds array for some values of i
            target_atom = np.where(np.logical_or(target_atom > max_atoms, mask==0), max_atoms, target_atom) #If invalid target, write to dummy col
            bonds[(source_row, target_atom)] = bond
            bonds[(target_row, source_atom)] = bond
            bond_dists[(source_row, target_atom)] = dists
            bond_dists[(target_row, source_atom)] = dists

        bonds = np.delete(bonds, axis=0, obj=-1) #Delete dummy row
        bonds = np.delete(bonds, axis=1, obj=-1) #Delete dummy col
        bond_dists = np.delete(bond_dists, axis=0, obj=-1) #Delete dummy row
        bond_dists = np.delete(bond_dists, axis=1, obj=-1) #Delete dummy col
        print('Counting and condensing bonds')
        bonds_numeric = [[i for i,x in enumerate(row) if x] for row in tqdm(bonds)]
        bond_lengths = [[dist for i,dist in enumerate(row) if i in bonds_numeric[j]] for j,row in enumerate(tqdm(bond_dists))]
        bond_lengths_mean = [ np.mean(x) for x in bond_lengths]
        #bond_lengths_min = [ np.min(x) for x in bond_lengths]
        bond_lengths_max = [ np.max(x) for x in bond_lengths]
        bond_lengths_std = [ np.std(x) for x in bond_lengths]
        bond_lengths_1 = [ sorted(x)[0] for x in bond_lengths]
        bond_lengths_2 = [ sorted(x)[1] if len(x)>=2 else np.nan for x in bond_lengths]
        bond_lengths_3 = [ sorted(x)[2] if len(x)>=3 else np.nan for x in bond_lengths]
        bond_lengths_4 = [ sorted(x)[3] if len(x)>=4 else np.nan for x in bond_lengths]
        n_bonds = [len(x) for x in bonds_numeric]
        #bond_data = {'bond_' + str(i):col for i, col in enumerate(np.transpose(bonds))}
        #bond_data.update({'bonds_numeric':bonds_numeric, 'n_bonds':n_bonds})
        bond_data = {
                'n_bonds':n_bonds, 
                'bond_lengths_mean': bond_lengths_mean,
                #'bond_lengths_min': bond_lengths_min,
                'bond_lengths_max': bond_lengths_max,
                'bond_lengths_std': bond_lengths_std,
                'bond_lengths_1': bond_lengths_1,
                'bond_lengths_2': bond_lengths_2,
                'bond_lengths_3': bond_lengths_3,
                'bond_lengths_4': bond_lengths_4,
        }
        bond_df = pd.DataFrame(bond_data)
        structures = structures.join(bond_df)       
        return structures

    def __call__(self, train_df, test_df, st_df):
        st_df = self.structure_preprocess(st_df)
        def map_atom_info(df, atom_idx):
            df = pd.merge(df, st_df, how = 'left',
                          left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                          right_on = ['molecule_name',  'atom_index'])
            
            df = df.drop('atom_index', axis=1)
            rename_dict = {c:c+"_{}".format(atom_idx) 
                    for c in st_df.columns if c not in ["molecule_name", "atom_index"]}
            df.rename(columns=rename_dict, inplace=True)
            return df
        train_df = map_atom_info(train_df, 0)
        train_df = map_atom_info(train_df, 1)
        test_df = map_atom_info(test_df, 0)
        test_df = map_atom_info(test_df, 1)
        # center
        st_df_center = st_df.groupby("molecule_name").agg({"x":["mean"],"y":["mean"],"z":["mean"]})
        st_df_center.columns = pd.Index(["c_x","c_y","c_z"])
        st_df.reset_index(inplace=True)
        train_df = utils.fast_merge(train_df, st_df_center, on="molecule_name")
        test_df = utils.fast_merge(test_df, st_df_center, on="molecule_name")
        return train_df, test_df


def read_graph(molecule_name):
    with open("../data/graph/{}.pickle".format(molecule_name),"rb") as f:
        data = pickle.load(f)[2]
        data = data[:,5:-1]
        return (molecule_name,data)

class AtomInfoFromGraphData(CreateFeature):
    def each_make(self,df):
        df_gr = df.groupby("molecule_name")
        molecules = df_gr.groups.keys()
        with Pool(4) as p:
            res = p.map(read_graph, molecules)
        atom_df = pd.concat([pd.DataFrame(d,index=[key]*len(d)) for key,d in res])
        atom_df.columns = ["is_accepter","is_donor","is_aromatic",
                "hybridization1","hybridization2","hybridization3","num_h"]
        atom_df.reset_index(inplace=True)
        atom_df.rename(columns={"index":"molecule_name"},inplace=True)
        atom_df.reset_index(inplace=True)
        atom_df.rename(columns={"index":"atom_index"},inplace=True)
        #print(df.columns)
        df = df.merge(atom_df, 
                left_on=["molecule_name","atom_index_0"],
                right_on=["molecule_name","atom_index"], 
                how="left", suffixes=("","_0"))
        del df["atom_index"]
        df = df.merge(atom_df, 
                left_on=["molecule_name","atom_index_1"], 
                right_on=["molecule_name","atom_index"],
                how="left", suffixes=("","_1"))
        del df["atom_index"]
        gc.collect()
        #df = utils.fast_merge(df, atom_df, on=["molecule_name","atom_index"])
        return df
        
    def __call__(self, train_df, test_df):
        train_df = self.each_make(train_df)
        test_df = self.each_make(test_df)
        return train_df, test_df


class MakeDistance(CreateFeature):
    def each_make(self, df):
        p_0 = df[['x_0', 'y_0', 'z_0']].values
        p_1 = df[['x_1', 'y_1', 'z_1']].values
        df['dist'] = np.linalg.norm(p_0 - p_1, axis=1)
        # inv dist per atom
        radius = {"H":0.38,"C":0.77,"N":0.75,"O":0.73,"F":0.71}
        #electron = {"H":2.2,"C":2.55,"N":3.04,"O":3.44,"F":3.98}
        rad0 = df["atom_0"].map(lambda x: radius[x])
        rad1 = df["atom_1"].map(lambda x: radius[x])
        #ele0 = df["atom_0"].map(lambda x: electron[x])
        #ele1 = df["atom_1"].map(lambda x: electron[x])
        df["inv_dist"] = 1 / ((df["dist"]-rad0-rad1)**3)
        df["inv_sum_dist_0"] = df.groupby(["molecule_name","atom_index_0"])["inv_dist"].transform("sum")
        df["inv_sum_dist_1"] = df.groupby(["molecule_name","atom_index_0"])["inv_dist"].transform("sum")
        df["inv_sum_dist_PR"] = (df["inv_sum_dist_0"]*df["inv_sum_dist_1"]) / (df["inv_sum_dist_0"]+df["inv_sum_dist_1"])
        del df["inv_dist"],
        return df

    def __call__(self, train_df, test_df):
        train_df = self.each_make(train_df)
        test_df = self.each_make(test_df)
        return train_df, test_df


class TypeMapping(CreateFeature):
    def __call__(self, train_df, test_df):
        train_df["type_0"] = train_df["type"].apply(lambda x: x[0]).astype(np.int8)
        test_df["type_0"] = test_df["type"].apply(lambda x: x[0]).astype(np.int8)
        type_dict = {"1JHC":0, "1JHN":1, "2JHH":2, "2JHN":3, "2JHC":4, "3JHH":5,"3JHC":6, "3JHN":7}
        train_df["type_int"] = train_df["type"].map(type_dict).astype(np.int8)
        test_df["type_int"] = test_df["type"].map(type_dict).astype(np.int8)
        type1_dict = {"JHC":0, "JHN":1, "JHH":2}
        train_df["type_1"] = train_df["type"].map(lambda x: type1_dict[x[1:]]).astype(np.int8)
        test_df["type_1"] = test_df["type"].map(lambda x: type1_dict[x[1:]]).astype(np.int8)
        return train_df,test_df

class MoleculeFeat(CreateFeature):
    def each_make(self,df,st_df):
        # num atom
        df["count_in_molecule"] = df.groupby("molecule_name")["id"].transform("count")
        for a in ["H","C","N","O","F"]:
            # st_df
            df_gr = st_df[st_df["atom"] == a].groupby("molecule_name")["atom_index"].count().reset_index()
            df_gr.columns = ["molecule_name", "count_{}".format(a)]
            df = df.merge(df_gr, on="molecule_name", how="left")
            # count
            df_gr = df[df["atom_1"] == a].groupby("molecule_name")["id"].count().reset_index()
            df_gr.columns = ["molecule_name", "bond_count_{}".format(a)]
            df = df.merge(df_gr, on="molecule_name", how="left")
            # dist
            #df_gr = df[df["atom_1"] == a].groupby("molecule_name")["dist"].mean().reset_index()
            #df_gr.columns = ["molecule_name", "dist_mean_{}".format(a)]
            #df = df.merge(df_gr, on="molecule_name", how="left")
        # num bond
        df['count_in_id_atom_0'] = df.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')
        df['count_in_id_atom_1'] = df.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')
        #df["sum_count_in_atom01"] = df["count_in_atom_0"] + df["count_in_atom_1"]
        tmp = df.groupby(["molecule_name","atom_index_0"]).size().reset_index()
        tmp.columns = ["m", "a", "size_0"]
        tmp2 = df.groupby(["molecule_name","atom_index_1"]).size().reset_index()
        tmp2.columns = ["m", "a", "size_1"]
        all_tmp = pd.merge(tmp,tmp2, on=["m","a"], how="outer")
        all_tmp["n_bond_atomic"] = all_tmp["size_0"].fillna(0) + all_tmp["size_1"].fillna(0)
        df = df.merge(all_tmp.rename(
            columns={"n_bond_atomic":"count_in_atom_0","size_0":"size_0_0","size_1":"size_1_0"}), 
                      left_on=["molecule_name","atom_index_0"], right_on=["m","a"],how="left")
        del df["m"], df["a"]
        df = df.merge(all_tmp.rename(
            columns={"n_bond_atomic":"count_in_atom_1","size_0":"size_0_1","size_1":"size_1_1"}), 
                      left_on=["molecule_name","atom_index_1"], right_on=["m","a"], how="left")
        del df["m"], df["a"]
        gc.collect()
        #df["sum_count_atomic"] = df["count_in_atom_0"] + df["count_in_atom_1"]
        df["diff_count_atomic"] = np.abs(df["count_in_atom_0"] - df["count_in_atom_1"])
        df["min_count_atomic"] = df[["count_in_atom_0","count_in_atom_1"]].min(axis=1)
        df["max_count_atomic"] = df[["count_in_atom_0","count_in_atom_1"]].max(axis=1)
        # distance in molecule
        df["mean_dist_in_molecule"] = df.groupby("molecule_name")["dist"].transform("mean")
        df["rel_molecule_dist"] = df["dist"] / df["mean_dist_in_molecule"]
        df["diff_molecule_dist"] = df["dist"] - df["mean_dist_in_molecule"]
        df["max_dist_in_molecule"] = df.groupby("molecule_name")["dist"].transform("max")
        df["rel_max_dist_in_molecule"] = df["dist"] / df["max_dist_in_molecule"]
        df["diff_max_dist_in_molecule"] = df["dist"] - df["max_dist_in_molecule"]
        df["min_dist_in_molecule"] = df.groupby("molecule_name")["dist"].transform("min")
        df["rel_min_dist_in_molecule"] = df["dist"] / df["min_dist_in_molecule"]
        df["diff_min_dist_in_molecule"] = df["dist"] - df["min_dist_in_molecule"]
        # groupby atom_index
        ## mean
        df["mean_dist_in_index_atom_0"] = df.groupby(["molecule_name","atom_index_0"])["dist"].transform("mean")
        df["mean_dist_in_index_atom_1"] = df.groupby(["molecule_name","atom_index_1"])["dist"].transform("mean")
        df["rel_mean_dist_in_index_atom_0"] = df["dist"] / df["mean_dist_in_index_atom_0"]
        df["rel_mean_dist_in_index_atom_1"] = df["dist"] / df["mean_dist_in_index_atom_1"]
        df["diff_mean_dist_in_index_atom_0"] = df["dist"] - df["mean_dist_in_index_atom_0"]
        df["diff_mean_dist_in_index_atom_1"] = df["dist"] - df["mean_dist_in_index_atom_1"]
        df["mean_dist_in_index_atom01"] = df[["mean_dist_in_index_atom_0","mean_dist_in_index_atom_1"]].mean(axis=1)
        df["diff_mean_dist_in_index_atom"] = np.abs(df["mean_dist_in_index_atom_0"]-df["mean_dist_in_index_atom_1"])
        ## min
        df["min_dist_in_index_atom_0"] = df.groupby(["molecule_name","atom_index_0"])["dist"].transform("min")
        df["min_dist_in_index_atom_1"] = df.groupby(["molecule_name","atom_index_1"])["dist"].transform("min")
        df["rel_min_dist_in_index_atom_0"] = df["dist"] / df["min_dist_in_index_atom_0"]
        df["rel_min_dist_in_index_atom_1"] = df["dist"] / df["min_dist_in_index_atom_1"]
        df["diff_min_dist_in_index_atom_0"] = df["dist"] - df["min_dist_in_index_atom_0"]
        df["diff_min_dist_in_index_atom_1"] = df["dist"] - df["min_dist_in_index_atom_1"]
        df["min_dist_in_index_atom_01"] = df[["min_dist_in_index_atom_0","min_dist_in_index_atom_1"]].min(axis=1)
        df["diff_min_dist_in_index_atom"] = np.abs(df["min_dist_in_index_atom_0"]-df["min_dist_in_index_atom_1"])
        ## max
        df["max_dist_in_index_atom_0"] = df.groupby(["molecule_name","atom_index_0"])["dist"].transform("max")
        df["max_dist_in_index_atom_1"] = df.groupby(["molecule_name","atom_index_1"])["dist"].transform("max")
        df["rel_max_dist_in_index_atom_0"] = df["dist"] / df["max_dist_in_index_atom_0"]
        df["rel_max_dist_in_index_atom_1"] = df["dist"] / df["max_dist_in_index_atom_1"]
        df["max_dist_in_index_atom_01"] = df[["max_dist_in_index_atom_0","max_dist_in_index_atom_1"]].mean(axis=1)
        df["diff_max_dist_in_index_atom"] = np.abs(df["max_dist_in_index_atom_0"]-df["max_dist_in_index_atom_1"])
        ## std
        df["std_dist_in_index_atom_0"] = df.groupby(["molecule_name","atom_index_0"])["dist"].transform("std")
        df["std_dist_in_index_atom_1"] = df.groupby(["molecule_name","atom_index_1"])["dist"].transform("std")
        df["rel_std_dist_in_index_atom_0"] = df["dist"] / df["std_dist_in_index_atom_0"]
        df["rel_std_dist_in_index_atom_1"] = df["dist"] / df["std_dist_in_index_atom_1"]
        df["std_dist_in_index_atom_01"] = df[["std_dist_in_index_atom_0","std_dist_in_index_atom_1"]].mean(axis=1)
        df["diff_std_dist_in_index_atom"] = np.abs(df["std_dist_in_index_atom_0"]-df["std_dist_in_index_atom_1"])
        return df
    def __call__(self, train_df, test_df, st_df):
        train_df = self.each_make(train_df, st_df)
        test_df = self.each_make(test_df, st_df)
        return train_df,test_df



class NBondFeat(CreateFeature):
    def each_make(self, df):
        df["diff_n_bonds"] = np.abs(df["n_bonds_0"] - df["n_bonds_1"]) 
        df["sum_n_bonds"] = df["n_bonds_0"] + df["n_bonds_1"]
        df["min_n_bonds"] = df[["n_bonds_0", "n_bonds_1"]].min(axis=1)
        df["max_n_bonds"] = df[["n_bonds_0", "n_bonds_1"]].max(axis=1)
        df["diff_bond_lengths_mean"] = np.abs(df["bond_lengths_mean_0"] - df["bond_lengths_mean_1"]) 
        df["sum_bond_lengths_mean"] = df["bond_lengths_mean_0"] + df["bond_lengths_mean_1"]
        df["min_bond_lengths_mean"] = df[["bond_lengths_mean_0", "bond_lengths_mean_1"]].min(axis=1)
        df["max_bond_lengths_mean"] = df[["bond_lengths_mean_0", "bond_lengths_mean_1"]].max(axis=1)
        df["rel_lengths_mean"] = df["sum_bond_lengths_mean"] \
            - df.groupby("molecule_name")["sum_bond_lengths_mean"].transform("mean")
        return df
    def __call__(self, train_df, test_df):
        train_df = self.each_make(train_df)
        test_df = self.each_make(test_df)
        return train_df, test_df


class AtomMetaFeat(CreateFeature):
    def each_make(self, df):
        df["max_mulliken_in_molecule"] = df.groupby("molecule_name")["mulliken_charge_atom_0"].transform("max")
        df["min_mulliken_in_molecule"] = df.groupby("molecule_name")["mulliken_charge_atom_0"].transform("min")
        df["std_mulliken_in_molecule"] = df.groupby("molecule_name")["mulliken_charge_atom_0"].transform("std")
        return df
    def __call__(self,train_df,test_df):
        train_meta = pd.read_pickle("../pickle/atomic_meta_train.pkl")
        test_meta = pd.read_pickle("../pickle/atomic_meta_test.pkl")
        # atom_0
        train_df = pd.merge(train_df, train_meta, 
                left_on=["molecule_name","atom_index_0"], 
                right_on=["molecule_name","atom_index"], 
                how="left"
        )
        test_df = pd.merge(test_df, test_meta, 
                left_on=["molecule_name","atom_index_0"], 
                right_on=["molecule_name","atom_index"], 
                how="left"
        )
        train_df.rename(columns={"mulliken_charge":"mulliken_charge_atom_0"},inplace=True)
        test_df.rename(columns={"mulliken_charge":"mulliken_charge_atom_0"},inplace=True)
        # atom_1
        train_df = pd.merge(train_df, train_meta, 
                left_on=["molecule_name","atom_index_1"], 
                right_on=["molecule_name","atom_index"], 
                how="left"
        )
        test_df = pd.merge(test_df, test_meta, 
                left_on=["molecule_name","atom_index_1"], 
                right_on=["molecule_name","atom_index"], 
                how="left"
        )
        train_df.rename(columns={"mulliken_charge":"mulliken_charge_atom_1"},inplace=True)
        test_df.rename(columns={"mulliken_charge":"mulliken_charge_atom_1"},inplace=True)
        # diff
        train_df["abs_diff_mulliken_charges"] = \
                np.abs(train_df["mulliken_charge_atom_0"] - train_df["mulliken_charge_atom_1"])
        test_df["abs_diff_mulliken_charges"] = \
                np.abs(test_df["mulliken_charge_atom_0"] - test_df["mulliken_charge_atom_1"])
        # sum
        train_df["sum_mulliken_charges"] = \
                train_df["mulliken_charge_atom_0"] + train_df["mulliken_charge_atom_1"]
        test_df["sum_mulliken_charges"] = \
                test_df["mulliken_charge_atom_0"] + test_df["mulliken_charge_atom_1"]
        #train_df = self.each_make(train_df)
        #test_df = self.each_make(test_df)
        return train_df, test_df

class AngleFeat(CreateFeature):
    def each_make(self, df):
        df["dist_center0"]=((df['x_0']-df['c_x'])**2+(df['y_0']-df['c_y'])**2+(df['z_0']-df['c_z'])**2)**(1/2)
        df["dist_center1"]=((df['x_0']-df['c_x'])**2+(df['y_0']-df['c_y'])**2+(df['z_0']-df['c_z'])**2)**(1/2)
        df["dist_c0"]=((df['x_0']-df['x_closest_0'])**2+(df['y_0']-df['y_closest_0'])**2+(df['z_0']-df['z_closest_0'])**2)**(1/2)
        df["dist_c1"]=((df['x_1']-df['x_closest_1'])**2+(df['y_1']-df['y_closest_1'])**2+(df['z_1']-df['z_closest_1'])**2)**(1/2)
        df["dist_f0"]=((df['x_0']-df['x_farthest_0'])**2+(df['y_0']-df['y_farthest_0'])**2+(df['z_0']-df['z_farthest_0'])**2)**(1/2)
        df["dist_f1"]=((df['x_1']-df['x_farthest_1'])**2+(df['y_1']-df['y_farthest_1'])**2+(df['z_1']-df['z_farthest_1'])**2)**(1/2)
        df["vec_center0_x"]=(df['x_0']-df['c_x'])/(df["dist_center0"]+1e-10)
        df["vec_center0_y"]=(df['y_0']-df['c_y'])/(df["dist_center0"]+1e-10)
        df["vec_center0_z"]=(df['z_0']-df['c_z'])/(df["dist_center0"]+1e-10)
        df["vec_center1_x"]=(df['x_1']-df['c_x'])/(df["dist_center1"]+1e-10)
        df["vec_center1_y"]=(df['y_1']-df['c_y'])/(df["dist_center1"]+1e-10)
        df["vec_center1_z"]=(df['z_1']-df['c_z'])/(df["dist_center1"]+1e-10)
        df["vec_c0_x"]=(df['x_0']-df['x_closest_0'])/(df["dist_c0"]+1e-10)
        df["vec_c0_y"]=(df['y_0']-df['y_closest_0'])/(df["dist_c0"]+1e-10)
        df["vec_c0_z"]=(df['z_0']-df['z_closest_0'])/(df["dist_c0"]+1e-10)
        df["vec_c1_x"]=(df['x_1']-df['x_closest_1'])/(df["dist_c1"]+1e-10)
        df["vec_c1_y"]=(df['y_1']-df['y_closest_1'])/(df["dist_c1"]+1e-10)
        df["vec_c1_z"]=(df['z_1']-df['z_closest_1'])/(df["dist_c1"]+1e-10)
        df["vec_f0_x"]=(df['x_0']-df['x_farthest_0'])/(df["dist_f0"]+1e-10)
        df["vec_f0_y"]=(df['y_0']-df['y_farthest_0'])/(df["dist_f0"]+1e-10)
        df["vec_f0_z"]=(df['z_0']-df['z_farthest_0'])/(df["dist_f0"]+1e-10)
        df["vec_f1_x"]=(df['x_1']-df['x_farthest_1'])/(df["dist_f1"]+1e-10)
        df["vec_f1_y"]=(df['y_1']-df['y_farthest_1'])/(df["dist_f1"]+1e-10)
        df["vec_f1_z"]=(df['z_1']-df['z_farthest_1'])/(df["dist_f1"]+1e-10)
        df["vec_x"]=(df['x_1']-df['x_0'])/df["dist"]
        df["vec_y"]=(df['y_1']-df['y_0'])/df["dist"]
        df["vec_z"]=(df['z_1']-df['z_0'])/df["dist"]
        df["cos_c0_c1"]=df["vec_c0_x"]*df["vec_c1_x"]+df["vec_c0_y"]*df["vec_c1_y"]+df["vec_c0_z"]*df["vec_c1_z"]
        df["cos_f0_f1"]=df["vec_f0_x"]*df["vec_f1_x"]+df["vec_f0_y"]*df["vec_f1_y"]+df["vec_f0_z"]*df["vec_f1_z"]
        df["cos_center0_center1"]=df["vec_center0_x"]*df["vec_center1_x"]+df["vec_center0_y"]*df["vec_center1_y"]+df["vec_center0_z"]*df["vec_center1_z"]
        df["cos_c0"]=df["vec_c0_x"]*df["vec_x"]+df["vec_c0_y"]*df["vec_y"]+df["vec_c0_z"]*df["vec_z"]
        df["cos_c1"]=df["vec_c1_x"]*df["vec_x"]+df["vec_c1_y"]*df["vec_y"]+df["vec_c1_z"]*df["vec_z"]
        df["cos_f0"]=df["vec_f0_x"]*df["vec_x"]+df["vec_f0_y"]*df["vec_y"]+df["vec_f0_z"]*df["vec_z"]
        df["cos_f1"]=df["vec_f1_x"]*df["vec_x"]+df["vec_f1_y"]*df["vec_y"]+df["vec_f1_z"]*df["vec_z"]
        df["cos_center0"]=df["vec_center0_x"]*df["vec_x"]+df["vec_center0_y"]*df["vec_y"]+df["vec_center0_z"]*df["vec_z"]
        df["cos_center1"]=df["vec_center1_x"]*df["vec_x"]+df["vec_center1_y"]*df["vec_y"]+df["vec_center1_z"]*df["vec_z"]
        df=df.drop(['vec_c0_x','vec_c0_y','vec_c0_z','vec_c1_x','vec_c1_y','vec_c1_z',
                    'vec_f0_x','vec_f0_y','vec_f0_z','vec_f1_x','vec_f1_y','vec_f1_z',
                    'vec_center0_x','vec_center0_y','vec_center0_z','vec_center1_x','vec_center1_y','vec_center1_z',
                    'vec_x','vec_y','vec_z'], axis=1)
        return df

    def __call__(self, train_df, test_df):
        train_df = self.each_make(train_df)
        test_df = self.each_make(test_df)
        return train_df, test_df

class CompareOtherAtoms(CreateFeature):
    def each_make(self, df, st_df):
        # distance per atoms
        #atom_index_0 (atom_index_0はHしかない)
        for a in ["H"]:
            target_atoms = st_df[st_df.atom == a]
            df_atoms = target_atoms.merge(df[["molecule_name","atom_index_0","x_0","y_0","z_0"]],
                    left_on=["molecule_name","atom_index"],
                    right_on = ["molecule_name", "atom_index_0"], how="inner") 
            df_atoms.reset_index(inplace=True)
            p_0 = df_atoms[["x","y","z"]].values
            p_1 = df_atoms[["x_0","y_0","z_0"]].values
            df_atoms["dist"] = np.linalg.norm(p_0 - p_1, axis=1)
            df_atoms_gr = df_atoms.groupby(["molecule_name","atom_index"]).agg(
                    {"dist":["max","min","mean"]})
            df_atoms_gr.columns = pd.Index([a+"0_"+e[0] + "_" + e[1] for e in df_atoms_gr.columns.tolist()])
            df_atoms_gr.reset_index(inplace=True)
            df_atoms_gr.rename(columns={"atom_index":"atom_index_0"},inplace=True)
            df = df.merge(df_atoms_gr, on=["molecule_name","atom_index_0"], how="left") 
        #atom_index_1
        for a in ["H","C","N","F","O"]:
            target_atoms = st_df[st_df.atom == a]
            df_atoms = target_atoms.merge(df[["molecule_name","atom_index_1","x_0","y_0","z_0"]],
                    left_on=["molecule_name","atom_index"],
                    right_on = ["molecule_name", "atom_index_1"], how="inner") 
            df_atoms.reset_index(inplace=True)
            p_0 = df_atoms[["x","y","z"]].values
            p_1 = df_atoms[["x_0","y_0","z_0"]].values
            df_atoms["dist"] = np.linalg.norm(p_0 - p_1, axis=1)
            df_atoms_gr = df_atoms.groupby(["molecule_name","atom_index"]).agg(
                    {"dist":["max","min","mean"]})
            df_atoms_gr.columns = pd.Index([a+"_"+e[0] + "_" + e[1] for e in df_atoms_gr.columns.tolist()])
            df_atoms_gr.reset_index(inplace=True)
            df_atoms_gr.rename(columns={"atom_index":"atom_index_1"},inplace=True)
            df = df.merge(df_atoms_gr, on=["molecule_name","atom_index_1"], how="left") 
        return df
            
    def __call__(self, train_df, test_df, st_df):
        train_df = self.each_make(train_df, st_df)
        test_df = self.each_make(test_df,st_df)
        return train_df, test_df

class GroupAtomIndex1(CreateFeature):
    def each_make(self, df, st_df):
        df["std_dist_same_atom_type"] = df.groupby(["molecule_name","atom_index_1","type"])\
                                        ["dist"].transform("std")
        df["mean_dist_same_atom_type"] = df.groupby(["molecule_name","atom_index_1","type"])\
                                        ["dist"].transform("mean")
        df["min_dist_same_atom_type"] = df.groupby(["molecule_name","atom_index_1","type"])\
                                        ["dist"].transform("min")
        df["max_dist_same_atom_type"] = df.groupby(["molecule_name","atom_index_1","type"])\
                                        ["dist"].transform("max")
        return df
            
    def __call__(self, train_df, test_df, st_df):
        train_df = self.each_make(train_df, st_df)
        test_df = self.each_make(test_df,st_df)
        return train_df, test_df

class SortedDist(CreateFeature):

    def each_make(self, df):
        # all type
        tmp_df = df[["molecule_name","atom_index_0","dist"]].copy()
        tmp_df.sort_values(["molecule_name","atom_index_0","dist"], inplace=True)
        dist1 = tmp_df.groupby(["molecule_name","atom_index_0"])["dist"].nth(0).reset_index()
        dist2 = tmp_df.groupby(["molecule_name","atom_index_0"])["dist"].nth(1).reset_index()
        dist3 = tmp_df.groupby(["molecule_name","atom_index_0"])["dist"].nth(2).reset_index()
        dist4 = tmp_df.groupby(["molecule_name","atom_index_0"])["dist"].nth(3).reset_index()
        df = utils.fast_merge(df, dist1.rename(columns={"dist":"all_dist1"}), on=["molecule_name","atom_index_0"])
        df = utils.fast_merge(df, dist2.rename(columns={"dist":"all_dist2"}), on=["molecule_name","atom_index_0"])
        df = utils.fast_merge(df, dist3.rename(columns={"dist":"all_dist3"}), on=["molecule_name","atom_index_0"])
        df = utils.fast_merge(df, dist4.rename(columns={"dist":"all_dist4"}), on=["molecule_name","atom_index_0"])
        # atom H
        tmp_df = df[df.atom_1 == "H"][["molecule_name","atom_index_0","dist"]].copy()
        tmp_df.sort_values(["molecule_name","atom_index_0","dist"], inplace=True)
        dist1 = tmp_df.groupby(["molecule_name","atom_index_0"])["dist"].nth(0).reset_index()
        dist2 = tmp_df.groupby(["molecule_name","atom_index_0"])["dist"].nth(1).reset_index()
        dist3 = tmp_df.groupby(["molecule_name","atom_index_0"])["dist"].nth(2).reset_index()
        dist4 = tmp_df.groupby(["molecule_name","atom_index_0"])["dist"].nth(3).reset_index()
        df = utils.fast_merge(df, dist1.rename(columns={"dist":"H_dist1"}), on=["molecule_name","atom_index_0"])
        df = utils.fast_merge(df, dist2.rename(columns={"dist":"H_dist2"}), on=["molecule_name","atom_index_0"])
        df = utils.fast_merge(df, dist3.rename(columns={"dist":"H_dist3"}), on=["molecule_name","atom_index_0"])
        df = utils.fast_merge(df, dist4.rename(columns={"dist":"H_dist4"}), on=["molecule_name","atom_index_0"])
        # atom C
        tmp_df = df[df.atom_1 == "C"][["molecule_name","atom_index_0","dist"]].copy()
        tmp_df.sort_values(["molecule_name","atom_index_0","dist"], inplace=True)
        dist1 = tmp_df.groupby(["molecule_name","atom_index_0"])["dist"].nth(0).reset_index()
        dist2 = tmp_df.groupby(["molecule_name","atom_index_0"])["dist"].nth(1).reset_index()
        dist3 = tmp_df.groupby(["molecule_name","atom_index_0"])["dist"].nth(2).reset_index()
        dist4 = tmp_df.groupby(["molecule_name","atom_index_0"])["dist"].nth(3).reset_index()
        df = utils.fast_merge(df, dist1.rename(columns={"dist":"C_dist1"}), on=["molecule_name","atom_index_0"])
        df = utils.fast_merge(df, dist2.rename(columns={"dist":"C_dist2"}), on=["molecule_name","atom_index_0"])
        df = utils.fast_merge(df, dist3.rename(columns={"dist":"C_dist3"}), on=["molecule_name","atom_index_0"])
        df = utils.fast_merge(df, dist4.rename(columns={"dist":"C_dist4"}), on=["molecule_name","atom_index_0"])
        # atom O
        """
        tmp_df = df[df.atom_1 == "O"][["molecule_name","atom_index_0","dist"]].copy()
        tmp_df.sort_values(["molecule_name","atom_index_0","dist"], inplace=True)
        dist1 = tmp_df.groupby(["molecule_name","atom_index_0"])["dist"].nth(0).reset_index()
        dist2 = tmp_df.groupby(["molecule_name","atom_index_0"])["dist"].nth(1).reset_index()
        dist3 = tmp_df.groupby(["molecule_name","atom_index_0"])["dist"].nth(2).reset_index()
        dist4 = tmp_df.groupby(["molecule_name","atom_index_0"])["dist"].nth(3).reset_index()
        df = utils.fast_merge(df, dist1.rename({"dist":"O_dist1"}), on=["molecule_name","atom_index_0"])
        df = utils.fast_merge(df, dist2.rename({"dist":"O_dist2"}), on=["molecule_name","atom_index_0"])
        df = utils.fast_merge(df, dist3.rename({"dist":"O_dist3"}), on=["molecule_name","atom_index_0"])
        df = utils.fast_merge(df, dist4.rename({"dist":"O_dist4"}), on=["molecule_name","atom_index_0"])
        """
        # atom N
        tmp_df = df[df.atom_1 == "N"][["molecule_name","atom_index_0","dist"]].copy()
        tmp_df.sort_values(["molecule_name","atom_index_0","dist"], inplace=True)
        dist1 = tmp_df.groupby(["molecule_name","atom_index_0"])["dist"].nth(0).reset_index()
        dist2 = tmp_df.groupby(["molecule_name","atom_index_0"])["dist"].nth(1).reset_index()
        dist3 = tmp_df.groupby(["molecule_name","atom_index_0"])["dist"].nth(2).reset_index()
        dist4 = tmp_df.groupby(["molecule_name","atom_index_0"])["dist"].nth(3).reset_index()
        df = utils.fast_merge(df, dist1.rename(columns={"dist":"N_dist1"}), on=["molecule_name","atom_index_0"])
        df = utils.fast_merge(df, dist2.rename(columns={"dist":"N_dist2"}), on=["molecule_name","atom_index_0"])
        df = utils.fast_merge(df, dist3.rename(columns={"dist":"N_dist3"}), on=["molecule_name","atom_index_0"])
        df = utils.fast_merge(df, dist4.rename(columns={"dist":"N_dist4"}), on=["molecule_name","atom_index_0"])
        return df
    def __call__(self, train_df, test_df):
        train_df = self.each_make(train_df)
        test_df = self.each_make(test_df)
        return train_df, test_df

def read_bond_from_graph(molecule_name):
    bond_types = ["single","double","triple","aromatic"]
    with open("../data/graph/{}.pickle".format(molecule_name),"rb") as f:
        data = pickle.load(f)
    arr = np.concatenate([data[4],data[3][:,:-1]],axis=1)
    df = pd.DataFrame(arr,columns=["atom_index_0","atom_index_1"] + bond_types)
    
    df["molecule_name"] = molecule_name
    return df

class NextBondPerType(CreateFeature):
    def each_type_make(self, df, bond_df, select_bond_type):
        print("make type {} ... ".format(select_bond_type))
        if select_bond_type != "all":
            select_type_bond_df = bond_df[bond_df.bond_atom.str.contains(select_bond_type)]
        else:
            select_type_bond_df = bond_df
        #いったんこの段階でマージ
        agg_recipe = {  **{c:["sum"] for c in ["single","double","triple","aromatic"]},
                        **{"L2dist":["min","max","std","mean"]},
                    }
        next_bond_gr = select_type_bond_df.rename(
                columns={
                    "atom_index_0":"idx_0",
                    "atom_index_1":"idx_1"
                }).groupby(["molecule_name","idx_0"]).agg(agg_recipe)
        next_bond_gr.columns = pd.Index(
                [select_bond_type+"_next_"+e[0]+"_"+e[1] for e in next_bond_gr.columns.tolist()])
        next_bond_gr.reset_index(inplace=True)
        df = df.merge(next_bond_gr, how="left", 
                left_on=["molecule_name","atom_index_1"],
                right_on=["molecule_name", "idx_0"])
        del df["idx_0"] 
        return df

    def each_make(self, df, bond_df):
        df = self.each_type_make(df,bond_df,"all")
        df = self.each_type_make(df,bond_df,"CC")
        df = self.each_type_make(df,bond_df,"CO")
        df = self.each_type_make(df,bond_df,"CN")
        return df

    def __call__(self, train_df, test_df):
        #train_bonds_df = pd.read_csv("../external/train_bonds.csv")
        #test_bonds_df = pd.read_csv("../external/test_bonds.csv")
        train_bonds_df = pd.read_pickle("../pickle/train_bond_v2.pkl")
        train_df = self.each_make(train_df, train_bonds_df)
        test_bonds_df = pd.read_pickle("../pickle/test_bond_v2.pkl")
        test_df = self.each_make(test_df, test_bonds_df)
        return train_df, test_df


class BondTypesFeat(CreateFeature):
    def each_type_make(self, df, bond_df, select_bond_type):
        print("make type {} ... ".format(select_bond_type))
        if select_bond_type != "all":
            select_type_bond_df = bond_df[bond_df.bond_atom.str.contains(select_bond_type)]
        else:
            select_type_bond_df = bond_df
        #いったんこの段階でマージ
        agg_recipe = {  **{c:["sum"] for c in ["single","double","triple","aromatic"]},
                        **{"L2dist":["min","max","std","mean"]},
                    }
        next_bond_gr = select_type_bond_df.rename(
                columns={
                    "atom_index_0":"idx_0",
                    "atom_index_1":"idx_1"
                }).groupby(["molecule_name","idx_0"]).agg(agg_recipe)
        next_bond_gr.columns = pd.Index(
                [select_bond_type+"_next_"+e[0]+"_"+e[1] for e in next_bond_gr.columns.tolist()])
        next_bond_gr.reset_index(inplace=True)
        df = df.merge(next_bond_gr, how="left", 
                left_on=["molecule_name","atom_index_1"],
                right_on=["molecule_name", "idx_0"])
        del df["idx_0"] 
        #さらにその先の結合との関係性
        next_bond2 = bond_df.merge(
                             select_type_bond_df, 
                             left_on=["molecule_name","atom_index_1"], 
                             right_on=["molecule_name","atom_index_0"],
                             how="inner",
                             suffixes=("_l","_r"))
        del next_bond2["atom_index_0_r"]
        next_bond2.rename(
                columns={
                        "atom_index_0_l":"idx_0",
                        "atom_index_1_l":"idx_1",
                        "atom_index_1_r":"idx_2",
                        "L2dist_r":"L2dist",
                },
                inplace=True
        )
        #行って変えるルート（元の原子と次の次の原子が同じ）を除く
        next_bond2 = next_bond2[next_bond2.idx_0 != next_bond2.idx_2]
        #groupbyでまとめるときのagg作成
        agg_recipe = {**{"nbond":["min","max","nunique","count"]}, 
                        #**{"1_"+c:["sum"] for c in next_bond2["bond_type1"].unique()},
                        **{"2_"+c:["sum"] for c in next_bond2["bond_type2"].unique()},
                        **{"L2dist":["min","max","std","mean"]},
                    }
        #結合元であるidx_0でgroupby
        next_bond2_gr = next_bond2.groupby(["molecule_name","idx_0"]).agg(agg_recipe)
        next_bond2_gr.columns = pd.Index(
                [select_bond_type+"_next2_"+e[0] + "_" +e[1] for e in next_bond2_gr.columns.tolist()])
        next_bond2_gr.reset_index(inplace=True)
        #元のdfにマージ
        df = df.merge(next_bond2_gr, how="left", 
                left_on=["molecule_name","atom_index_1"],
                right_on=["molecule_name", "idx_0"])
        del df["idx_0"]
        return df

    def each_make(self, df, bond_df):
        df = self.each_type_make(df,bond_df,"all")
        df = self.each_type_make(df,bond_df,"CC")
        df = self.each_type_make(df,bond_df,"CO")
        df = self.each_type_make(df,bond_df,"CN")
        return df
    def __call__(self, train_df, test_df):
        #train_bonds_df = pd.read_csv("../external/train_bonds.csv")
        #test_bonds_df = pd.read_csv("../external/test_bonds.csv")
        train_bonds_df = pd.read_pickle("../pickle/train_bond_v2.pkl")
        train_df = self.each_make(train_df, train_bonds_df)
        test_bonds_df = pd.read_pickle("../pickle/test_bond_v2.pkl")
        test_df = self.each_make(test_df, test_bonds_df)
        return train_df, test_df



class BondAngleFeat012(CreateFeature):
    """
    A:AtomIndex0(H)、B:それに結合した原子、C:更にそれに結合した原子の角ABCを出す
    """
    def each_make(self, df, mode):
        bond_df = pd.read_pickle(f"../pickle/{mode}_bond_v2.pkl")
        edge_df = pd.read_pickle(f"../pickle/{mode}_edge_angle.pkl")
        ## 1Jxx以外、dfのatom_index_1 != Bでないことに注意
        ##この時点でatom_index_0に対する1のユニーク数は1
        bond_AB = df[["molecule_name","atom_index_0"]].drop_duplicates()\
                .merge(bond_df, how="inner", on=["molecule_name","atom_index_0"])
        del bond_AB["atom_1"]
        #edge情報をマージ
        edge_use_cols = ["molecule_name","atom_index_0","atom_index_1","atom_index_2",
                        "atom_1","atom_2","benzen_1","single_1","cos_012"]
        edge_012 = bond_AB.merge(edge_df[edge_use_cols], 
                        on=["molecule_name","atom_index_0","atom_index_1"],
                        how="inner")
        ## cos_012を小さい順にソート
        edge_012.sort_values(["molecule_name","atom_index_0","atom_index_1","cos_012"],
                ascending=False,inplace=True)
        #いろんなatomCに対して、小さい順3つ+stdをもとめる
        def second(x):
            return x.iloc[1] if len(x)>1 else np.nan
        def third(x):
            return x.iloc[2] if len(x)>2 else np.nan
        #まずは全結合
        edge_012_gr = edge_012.groupby(["molecule_name","atom_index_0"]).agg({
                "cos_012":["first",second,third,"std"]
                })
        edge_012_gr.columns = pd.Index(["all_"+e[0]+"_"+e[1] for e in edge_012_gr.columns.tolist()])
        edge_012_gr.reset_index(inplace=True)
        df = utils.fast_merge(df, edge_012_gr, on=["molecule_name","atom_index_0"])
        print(edge_012.columns)
        #atom12がCC結合
        """
        edge_012_gr = edge_012[(edge_012["atom_1"]=="C")&(edge_012["atom_2"]=="C")]\
                .groupby(["molecule_name","atom_index_0"]).agg({
                    "cos_012":["first",second,third,"std"]
                    })
        edge_012_gr.columns = pd.Index(["CC_"+e[0]+"_"+e[1] for e in edge_012_gr.columns.tolist()])
        edge_012_gr.reset_index(inplace=True)
        df = utils.fast_merge(df, edge_012_gr, on=["molecule_name","atom_index_0"])
        #atom12がCO結合
        edge_012_gr = edge_012[(edge_012["atom_1"]=="C")&(edge_012["atom_2"]=="O")]\
                .groupby(["molecule_name","atom_index_0"]).agg({
                    "cos_012":["first",second,third,"std"]
                    })
        edge_012_gr.columns = pd.Index(["CO_"+e[0]+"_"+e[1] for e in edge_012_gr.columns.tolist()])
        edge_012_gr.reset_index(inplace=True)
        df = utils.fast_merge(df, edge_012_gr, on=["molecule_name","atom_index_0"])
        #edge12がベンゼン環
        edge_012_gr = edge_012[(edge_012["benzen_1"]=="C")]\
                .groupby(["molecule_name","atom_index_0"]).agg({
                    "cos_012":["first",second,third,"std"]
                    })
        edge_012_gr.columns = pd.Index(["benzen_"+e[0]+"_"+e[1] for e in edge_012_gr.columns.tolist()])
        edge_012_gr.reset_index(inplace=True)
        df = utils.fast_merge(df, edge_012_gr, on=["molecule_name","atom_index_0"])
        """
        return df
    def __call__(self, train_df, test_df):
        train_df = self.each_make(train_df, mode="train")
        test_df = self.each_make(test_df, mode="test")
        return train_df, test_df


class BondAngleFeat123(CreateFeature):
    def each_make(self, df, mode):
        bond_df = pd.read_pickle(f"../pickle/{mode}_bond_v2.pkl")
        edge_df = pd.read_pickle(f"../pickle/{mode}_edge_angle.pkl")
        #A:AtomIndex0(H)、B:それに結合した原子、C:更にそれに結合した原子(C1,C2,...)
        #角Cx-B-Cyを出す
        ## その後、小さい順に並べたり、いくつか特徴量化する
        ##この時点でatom_index_0に対する1のユニーク数は1
        bond_AB = df[["molecule_name","atom_index_0"]].drop_duplicates()\
                .merge(bond_df, how="inner", on=["molecule_name","atom_index_0"])
        bond_AB.drop(["atom_0","atom_1"],axis=1,inplace=True)
        #edge情報をマージ
        edge_use_cols = ["molecule_name","atom_index_0","atom_index_1","atom_index_2",
                        "atom_1","atom_2","benzen_1","single_1","cos_012"]
        edge_012 = bond_AB.merge(edge_df[edge_use_cols], 
                        left_on=["molecule_name","atom_index_1"],
                        right_on=["molecule_name", "atom_index_1"],
                        how="inner")
        #元のAtomIndex0とedge_dfのAtomIndex0が異なるものに絞る
        edge_012 = edge_012[edge_012["atom_index_0_x"]!=edge_012["atom_index_0_y"]]
        ## cos_012を小さい順にソート
        edge_012.sort_values(["molecule_name","atom_index_1","cos_012"],
                ascending=False,inplace=True)
        #いろんなatomCに対して、小さい順3つ+stdをもとめる
        #def second(x):
        #    return x.iloc[1] if len(x)>1 else np.nan
        #def third(x):
        #    return x.iloc[2] if len(x)>2 else np.nan
        edge_012_gr = edge_012.groupby(["molecule_name","atom_index_1"]).agg({
                "cos_012":["min","max"]
                #"cos_012":["first",second,third,"std"]
                })
        edge_012_gr.columns = pd.Index(["all123_"+e[0]+"_"+e[1] for e in edge_012_gr.columns.tolist()])
        edge_012_gr["all123_diff_minmax_cos_012"] = edge_012_gr["all123_cos_012_max"]\
                                                 -edge_012_gr["all123_cos_012_min"]
        edge_012_gr.reset_index(inplace=True)
        # dfにマージ
        df = utils.fast_merge(df, edge_012_gr, on=["molecule_name","atom_index_1"])
        return df
    def __call__(self, train_df, test_df):
        train_df = self.each_make(train_df, mode="train")
        test_df = self.each_make(test_df, mode="test")
        return train_df, test_df



class AtomBondFeat(CreateFeature):
    def each_make(self, df):
        molecules = df["molecule_name"].unique().tolist()
        with Pool(4) as p:
            res = p.map(read_bond_from_graph, molecules)
        bond_info_df = pd.concat(res, axis=0)
        df = utils.fast_merge(df, bond_info_df, on=["molecule_name","atom_index_0","atom_index_1"])
        df["is_bond"] = df[["single","double","triple","aromatic"]].max(axis=1)
        #c_bond_df = df.loc[(df["is_bond"]>0)&(df["atom_1"]=="C"),["single","double","triple","aromatic"]]
        return df

    def __call__(self, train_df, test_df):
        train_df = self.each_make(train_df)
        test_df = self.each_make(test_df)
        return train_df, test_df


class ACSFDescriptor(CreateFeature):
    def each_make(self, df, acsf_df):
        acsf_cols = [c for c in acsf_df.columns 
                        if c not in ["molecule_name","atom_index"]]
        #merge to atom_index_0
        df = df.merge(acsf_df.rename(columns={c:"atom0_"+c for c in acsf_cols}), 
                left_on=["molecule_name","atom_index_0"],
                right_on=["molecule_name","atom_index"],
                how="left"
        )
        del df["atom_index"]
        #merge to atom_index_1
        df = df.merge(acsf_df.rename(columns={c:"atom1_"+c for c in acsf_cols}), 
                left_on=["molecule_name","atom_index_1"],
                right_on=["molecule_name","atom_index"],
                how="left"
        )
        del df["atom_index"]
        #差分
        acsf_cols_0 = ["atom0_"+c for c in acsf_cols]
        acsf_cols_1 = ["atom1_"+c for c in acsf_cols]
        diff_acsf_df = pd.DataFrame(df[acsf_cols_1].values - df[acsf_cols_0].values,
                        columns=["atom0to1_"+c for c in acsf_cols])
        df = utils.fast_concat(df, diff_acsf_df)
        return df
             
    def __call__(self, train_df, test_df):
        acsf_df = pd.read_pickle("../pickle/acsf_array.pkl")
        train_df = self.each_make(train_df,acsf_df)
        test_df = self.each_make(test_df,acsf_df)
        return train_df, test_df

class NextACSFDescriptor(CreateFeature):
    """
    接続しているatomのACSFを求める
    """
    def each_make(self, df, acsf_df, bond_df):
        acsf_cols = [c for c in acsf_df.columns 
                        if c not in ["molecule_name","atom_index"]]

        use_cols = ["molecule_name","atom_index_0","atom_index_1"]
        next_bond = bond_df.loc[bond_df.bond_type!=-1, use_cols]
        #いったんこの段階でマージ
        #next_bond = replace_concat(select_type_bond_df)[use_cols]
        next_bond.rename(
                columns={
                    "atom_index_0":"idx_0",
                    "atom_index_1":"idx_1"
                },
                inplace=True
        )
        #merge to atom_index_0
        next_bond = next_bond.merge(acsf_df.rename(columns={c:"atom1_"+c for c in acsf_cols}), 
                left_on=["molecule_name","idx_1"],
                right_on=["molecule_name","atom_index"],
                how="left"
        )
        del next_bond["atom_index"]
        agg_recipe = {"atom1_"+c:["mean"] for c in acsf_cols}
        next_bond_gr = next_bond.groupby(["molecule_name","idx_0"]).agg(agg_recipe)
        next_bond_gr.columns = pd.Index(
                ["next_"+e[0]+"_"+e[1] for e in next_bond_gr.columns.tolist()])
        next_bond_gr.reset_index(inplace=True)
        df = df.merge(next_bond_gr, how="left", 
                left_on=["molecule_name","atom_index_1"],
                right_on=["molecule_name", "idx_0"])
        del df["idx_0"] 
        gc.collect()
        return df
             
    def __call__(self, train_df, test_df):
        acsf_df = pd.read_pickle("../pickle/acsf_array.pkl")
        #train_bonds_df = pd.read_csv("../external/train_bonds.csv")
        #test_bonds_df = pd.read_csv("../external/test_bonds.csv")
        train_bonds_df = pd.read_pickle("../pickle/train_bond_v2.pkl")
        test_bonds_df = pd.read_pickle("../pickle/test_bond_v2.pkl")
        train_df = self.each_make(train_df,acsf_df, train_bonds_df)
        test_df = self.each_make(test_df,acsf_df, test_bonds_df)
        return train_df, test_df


class OpenBabelMC(CreateFeature):
    """
    This kernel output
    https://www.kaggle.com/asauve/v7-estimation-of-mulliken-charges-with-open-babel/output
    predicting some types of mulliken charge by openbabel
    """
    def each_make(self, df, mc_df):
        del mc_df["Unnamed: 0"]
        rename_dict = {c:c+"_0" for c in mc_df.columns 
                        if c not in ["molecule_name","atom_index"]}
        df = df.merge(mc_df.rename(columns=rename_dict), 
                                    left_on=["molecule_name","atom_index_0"], 
                                    right_on=["molecule_name","atom_index"],
                                    how="left"
                                )
        del df["atom_index"]
        rename_dict = {c:c+"_1" for c in mc_df.columns 
                        if c not in ["molecule_name","atom_index"]}
        df = df.merge(mc_df.rename(columns=rename_dict), 
                                    left_on=["molecule_name","atom_index_1"], 
                                    right_on=["molecule_name","atom_index"],
                                    how="left"
                                )
        del df["atom_index"]
        gc.collect()
        mc_cols = [c for c in mc_df.columns if c not in ["molecule_name","atom_index"]]
        for mc_col in mc_cols:
            df[mc_col+"_sum"] = df[mc_col+"_0"] + df[mc_col+"_1"]
            df[mc_col+"_diff"] = df[mc_col+"_0"] - df[mc_col+"_1"]
        return df

    def __call__(self, train_df, test_df):
        train_mc_df = pd.read_csv("../data/train_ob_charges.csv")
        test_mc_df = pd.read_csv("../data/test_ob_charges.csv")
        train_df = self.each_make(train_df, train_mc_df)
        test_df = self.each_make(test_df, test_mc_df)
        return train_df, test_df


class HinokkiMeta(CreateFeature):
    def each_make(self, df,mode):
        hinokki_df = pd.read_csv(f"../data/{mode}_hinokki.csv")
        use_cols = ["new_type",
                "atom_index_2","atom_index_30","atom_index_31","atom_index_32"]
        assert(len(df) == len(hinokki_df))
        df = utils.fast_concat(df, hinokki_df[use_cols])
        return df

    def __call__(self, train_df, test_df):
        train_df = self.each_make(train_df, mode="train")
        test_df = self.each_make(test_df, mode="test")
        train_df["new_type"],uniques = pd.factorize(train_df["new_type"])
        test_df["new_type"] = uniques.get_indexer(test_df["new_type"])
        return train_df, test_df



@jit
def numba_dist_matrix(xyz, ssx, molecule_id):
    start_molecule, end_molecule = ssx[molecule_id], ssx[molecule_id+1]
    locs = xyz[start_molecule:end_molecule]     
   # return locs
    num_atoms = end_molecule - start_molecule
    dmat = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            d = sqrt((locs[i,0] - locs[j,0])**2 + (locs[i,1] - locs[j,1])**2 + (locs[i,2] - locs[j,2])**2)
            dmat[i,j] = d
            dmat[j,i] = d
    return dmat
def numba_dist_matrices(xyz, ssx, molecules):
    dist_arr_dict = {}
    for idx,molecule_id in enumerate(molecules):
        dist_arr_dict[molecule_id] = numba_dist_matrix(xyz, ssx, idx)
    return dist_arr_dict

class NearDistFeat(CreateFeature):
    def make_atom_dict(self, atom_arr, ssx, molecules):
        atom_arr_dict = {}
        for idx,molecule_id in enumerate(molecules):
            atom_arr_dict[molecule_id] = atom_arr[ssx[idx]:ssx[idx+1]]
        return atom_arr_dict

    def make_dict(self, st_df, ssx):
        xyz = st_df[["x","y","z"]].values
        molecules = st_df["molecule_name"].unique() 
        dist_arr_dict = numba_dist_matrices(xyz, ssx, molecules)
        atom_dict = self.make_atom_dict(st_df["atom"].values, ssx, molecules)
        return dist_arr_dict,atom_dict
    
    def make_ssx(self, st_df):
        ss = st_df.groupby('molecule_name').size()
        ss = ss.cumsum()
        ssx = np.zeros(len(ss) + 1, 'int')
        ssx[1:] = ss
        return ssx

    def __call__(self, train_df, test_df, st_df):
        ssx = self.make_ssx(st_df)
        dist_arr_dict, atom_dict = self.make_dict(st_df, ssx)
        c_near_num = np.zeros(len(st_df))
        h_near_num = np.zeros(len(st_df))
        o_near_num = np.zeros(len(st_df))
        n_near_num = np.zeros(len(st_df))
        for m_ix,molecule_name in enumerate(tqdm(st_df["molecule_name"].unique())):
            dist_arr = dist_arr_dict[molecule_name]
            atom_arr = atom_dict[molecule_name]
            start,end = ssx[m_ix], ssx[m_ix+1] 
            c_near_num[start:end] = (dist_arr[:,atom_arr=="C"]<2.0).sum(axis=1)
            h_near_num[start:end] = (dist_arr[:,atom_arr=="H"]<2.0).sum(axis=1)
            o_near_num[start:end] = (dist_arr[:,atom_arr=="O"]<2.0).sum(axis=1)
            n_near_num[start:end] = (dist_arr[:,atom_arr=="N"]<2.0).sum(axis=1)
        st_df["near_num_C"] = c_near_num
        st_df["near_num_H"] = h_near_num
        st_df["near_num_O"] = o_near_num
        st_df["near_num_N"] = n_near_num
        train_df = pd.merge(train_df, st_df[["molecule_name","atom_index",
            "near_num_C","near_num_H","near_num_O","near_num_N"]]\
                    .rename(columns={"atom_index":"atom_index_0"}),
                    on=["molecule_name","atom_index_0"],how="left")
        train_df = pd.merge(train_df, st_df[["molecule_name","atom_index",
            "near_num_C","near_num_H","near_num_O","near_num_N"]]\
                    .rename(columns={"atom_index":"atom_index_1"}),
                    on=["molecule_name","atom_index_1"],how="left")
        test_df = pd.merge(test_df, st_df[["molecule_name","atom_index",
            "near_num_C","near_num_H","near_num_O","near_num_N"]]\
                    .rename(columns={"atom_index":"atom_index_0"}),
                    on=["molecule_name","atom_index_0"],how="left")
        test_df = pd.merge(test_df, st_df[["molecule_name","atom_index",
            "near_num_C","near_num_H","near_num_O","near_num_N"]]\
                    .rename(columns={"atom_index":"atom_index_1"}),
                    on=["molecule_name","atom_index_1"],how="left")
        train_df["diff_nearnum_C"] = train_df["near_num_C_x"] - train_df["near_num_C_y"]
        test_df["diff_nearnum_C"] = test_df["near_num_C_x"] - test_df["near_num_C_y"]
        train_df["diff_nearnum_O"] = train_df["near_num_O_x"] - train_df["near_num_O_y"]
        test_df["diff_nearnum_O"] = test_df["near_num_O_x"] - test_df["near_num_O_y"]
        train_df["diff_nearnum_H"] = train_df["near_num_H_x"] - train_df["near_num_H_y"]
        test_df["diff_nearnum_H"] = test_df["near_num_H_x"] - test_df["near_num_H_y"]
        train_df["diff_nearnum_N"] = train_df["near_num_N_x"] - train_df["near_num_N_y"]
        test_df["diff_nearnum_N"] = test_df["near_num_N_x"] - test_df["near_num_N_y"]
        drop_cols =["near_num_C_x","near_num_H_x","near_num_O_x","near_num_N_x",
                "near_num_C_y","near_num_H_y","near_num_O_y","near_num_N_y"]
        train_df.drop(drop_cols,axis=1,inplace=True)
        test_df.drop(drop_cols,axis=1,inplace=True)
        print(train_df["diff_nearnum_N"].value_counts())
        return train_df, test_df 


class NearON(CreateFeature):
    """
    最も近いO,Nがどれだけ近いか
    """
    def make_atom_dict(self, atom_arr, ssx, molecules):
        atom_arr_dict = {}
        for idx,molecule_id in enumerate(molecules):
            atom_arr_dict[molecule_id] = atom_arr[ssx[idx]:ssx[idx+1]]
        return atom_arr_dict

    def make_dict(self, st_df, ssx):
        xyz = st_df[["x","y","z"]].values
        molecules = st_df["molecule_name"].unique() 
        dist_arr_dict = numba_dist_matrices(xyz, ssx, molecules)
        atom_dict = self.make_atom_dict(st_df["atom"].values, ssx, molecules)
        return dist_arr_dict,atom_dict
    
    def make_ssx(self, st_df):
        ss = st_df.groupby('molecule_name').size()
        ss = ss.cumsum()
        ssx = np.zeros(len(ss) + 1, 'int')
        ssx[1:] = ss
        return ssx

    def __call__(self, train_df, test_df, st_df):
        ssx = self.make_ssx(st_df)
        dist_arr_dict, atom_dict = self.make_dict(st_df, ssx)
        o_near_dist = np.zeros(len(st_df))
        n_near_dist = np.zeros(len(st_df))
        for m_ix,molecule_name in enumerate(tqdm(st_df["molecule_name"].unique())):
            dist_arr = dist_arr_dict[molecule_name]
            atom_arr = atom_dict[molecule_name]
            start,end = ssx[m_ix], ssx[m_ix+1] 
            # 一つもなかった場合は距離を10とするようにconcatしておく
            o_near_dist[start:end] = np.concatenate([
                                            dist_arr[:,atom_arr=="O"], 
                                            np.ones(len(dist_arr)).reshape([-1,1])*10,
                                        ],
                                        axis=1
                                    ).min(axis=1)
            n_near_dist[start:end] = np.concatenate([
                                            dist_arr[:,atom_arr=="N"], 
                                            np.ones(len(dist_arr)).reshape([-1,1])*10
                                        ],
                                        axis=1
                                    ).min(axis=1)
        st_df["near_O_dist"] = o_near_dist
        st_df["near_N_dist"] = n_near_dist
        train_df = pd.merge(train_df, st_df[["molecule_name","atom_index",
                            "near_O_dist","near_N_dist"]]\
                    .rename(columns={"atom_index":"atom_index_0"}),
                    on=["molecule_name","atom_index_0"],how="left")
        test_df = pd.merge(test_df, st_df[["molecule_name","atom_index",
                            "near_O_dist","near_N_dist"]]\
                    .rename(columns={"atom_index":"atom_index_0"}),
                    on=["molecule_name","atom_index_0"],how="left")
        train_df = pd.merge(train_df, st_df[["molecule_name","atom_index",
                            "near_O_dist","near_N_dist"]]\
                    .rename(columns={"atom_index":"atom_index_1"}),
                    on=["molecule_name","atom_index_1"],how="left")
        test_df = pd.merge(test_df, st_df[["molecule_name","atom_index",
                            "near_O_dist","near_N_dist"]]\
                    .rename(columns={"atom_index":"atom_index_1"}),
                    on=["molecule_name","atom_index_1"],how="left")
        drop_cols =["near_O_dist_x","near_N_dist_x"]
        train_df.drop(drop_cols,axis=1,inplace=True)
        test_df.drop(drop_cols,axis=1,inplace=True)
        print(train_df["near_O_dist_y"].round(3).value_counts().iloc[:10])
        return train_df, test_df 


class MullikenTotal(CreateFeature):
    """
    MullikenCharge/distの合計を計算
    """
    def make_mc_dict(self, mc_arr, ssx, molecules):
        mc_arr_dict = {}
        for idx,molecule_id in enumerate(molecules):
            mc_arr_dict[molecule_id] = mc_arr[ssx[idx]:ssx[idx+1]]
        return mc_arr_dict

    def make_dict(self, st_df, ssx, mc_arr):
        xyz = st_df[["x","y","z"]].values
        molecules = st_df["molecule_name"].unique() 
        dist_arr_dict = numba_dist_matrices(xyz, ssx, molecules)
        mc_dict = self.make_mc_dict(mc_arr, ssx, molecules)
        return dist_arr_dict,mc_dict
    
    def make_ssx(self, st_df):
        ss = st_df.groupby('molecule_name').size()
        ss = ss.cumsum()
        ssx = np.zeros(len(ss) + 1, 'int')
        ssx[1:] = ss
        return ssx

    def __call__(self, train_df, test_df, st_df):
        ssx = self.make_ssx(st_df)
        train_mc = pd.read_pickle("../pickle/atomic_meta_train.pkl")["mulliken_charge"].values
        test_mc = pd.read_pickle("../pickle/atomic_meta_test.pkl")["mulliken_charge"].values
        mc_arr = np.concatenate([train_mc, test_mc])
        dist_arr_dict, mc_dict = self.make_dict(st_df, ssx, mc_arr)
        #分子単位に計算
        total_mc = np.zeros(len(st_df))
        for m_ix,molecule_name in enumerate(tqdm(st_df["molecule_name"].unique())):
            dist_arr = dist_arr_dict[molecule_name]
            mc_arr = mc_dict[molecule_name]
            start,end = ssx[m_ix], ssx[m_ix+1] 
            total_mc[start:end] = (mc_arr/(1+dist_arr)).sum(axis=1)
        st_df["mc"] = total_mc
        train_df = pd.merge(train_df, st_df[["molecule_name","atom_index","mc"]]\
                    .rename(columns={"atom_index":"atom_index_0"}),
                    on=["molecule_name","atom_index_0"],how="left")
        train_df = pd.merge(train_df, st_df[["molecule_name","atom_index","mc"]]\
                    .rename(columns={"atom_index":"atom_index_1"}),
                    on=["molecule_name","atom_index_1"],how="left")
        test_df = pd.merge(test_df, st_df[["molecule_name","atom_index","mc"]]\
                    .rename(columns={"atom_index":"atom_index_0"}),
                    on=["molecule_name","atom_index_0"],how="left")
        test_df = pd.merge(test_df, st_df[["molecule_name","atom_index","mc"]]\
                    .rename(columns={"atom_index":"atom_index_1"}),
                    on=["molecule_name","atom_index_1"],how="left")

        return train_df, test_df 


class Feature_1JHC(CreateFeature):
    def each_make(self, df, bond_df, st_df):
        suffix = "_1JHC"
        bond_df = replace_concat(bond_df)
        #couplingのCとHに関する特徴量
        bond_CH = bond_df[bond_df.bond_type=="1CH"]
        bond_CH_agg = {
                "atom_index_0":["count"], #C側に何個のHがついていたか
                "L2dist":["mean","max","min"]
        }
        bond_CH_gr = bond_CH.groupby(["molecule_name","atom_index_1"]).agg(bond_CH_agg)
        bond_CH_gr_cols = ["CH_"+e[0]+"_"+e[1]+suffix
                                for e in bond_CH_gr.columns.tolist()]
        bond_CH_gr.columns = pd.Index(bond_CH_gr_cols)
        bond_CH_gr.reset_index(inplace=True)
        df = utils.fast_merge(df, bond_CH_gr, on=["molecule_name","atom_index_1"])
        ## max minのdiff
        df["CH_L2dist_diff_maxmin"+suffix] = df["CH_L2dist_max"+suffix]-df["CH_L2dist_min"+suffix]
        del bond_CH_gr, bond_CH
        del df["CH_L2dist_max"+suffix], df["CH_L2dist_min"+suffix]
        gc.collect()
        #couplingのCに対してH以外の結合パターン
        ## bondにatomをマージ
        bond_df = bond_df.merge(st_df[["molecule_name","atom_index","atom"]], 
                left_on=["molecule_name","atom_index_0"],
                right_on=["molecule_name","atom_index"], 
                how="left").rename(columns={"atom":"atom_0"})
        del bond_df["atom_index"]
        bond_df = bond_df.merge(st_df[["molecule_name","atom_index","atom"]], 
                left_on=["molecule_name","atom_index_1"],
                right_on=["molecule_name","atom_index"], 
                how="left").rename(columns={"atom":"atom_1"})
        del bond_df["atom_index"]
        tmp = bond_df[bond_df.bond_type=="1CH"][["molecule_name","atom_index_1"]]\
                .drop_duplicates().merge(
                        bond_df, 
                        on=["molecule_name","atom_index_1"], how="left"
                    )
        tmp.sort_values(["molecule_name","atom_index_1","bond_type"], inplace=True)
        bond_pattern = tmp[tmp.atom_1 == "C"].groupby(["molecule_name","atom_index_1"])\
                ["bond_type"].apply(lambda x: "_".join(x)).reset_index()
        df = utils.fast_merge(df, bond_pattern.rename(
                                columns={"bond_type":"bond_atom1_C"+suffix}
                            ),
                    on=["molecule_name","atom_index_1"])
        del bond_pattern
        # CH以外の結合におけるCの距離
        agg_recipe = {"L2dist":["min","max","mean"]}
        bond_df_gr = bond_df[(bond_df.atom_0!="H")&(bond_df.atom_1=="C")]\
                .groupby(["molecule_name","atom_index_1"])["L2dist"].agg(agg_recipe)
        bond_df_gr.columns = pd.Index(
                ["notCH_"+e[0]+"_"+e[1]+suffix for e in bond_df_gr.columns.tolist()])
        bond_df_gr.reset_index(inplace=True)
        df = utils.fast_merge(df, bond_df_gr, on=["molecule_name","atom_index_1"])
        df["notCH_L2dist_diff_maxmin"+suffix] = \
                df["notCH_L2dist_max"+suffix]-df["notCH_L2dist_min"+suffix]
        del df["notCH_L2dist_max"+suffix],df["notCH_L2dist_min"+suffix]
        # CHとCH以外で比較
        df["diff_CH_notCH_L2dist_mean"+suffix] = \
                df["notCH_L2dist_mean"+suffix] - df["CH_L2dist_mean"+suffix]
        #df["rel_CH_notCH_L2dist_mean"+suffix] = \
        #        df["notCH_L2dist_mean"+suffix] / df["CH_L2dist_mean"+suffix]
        # CCの結合
        agg_recipe = {"L2dist":["min","max","mean"]}
        bond_df_gr = bond_df[(bond_df.atom_0=="C")&(bond_df.atom_1=="C")]\
                .groupby(["molecule_name","atom_index_1"])["L2dist"].agg(agg_recipe)
        bond_df_gr.columns = pd.Index(
                ["CC_"+e[0]+"_"+e[1]+suffix for e in bond_df_gr.columns.tolist()])
        bond_df_gr.reset_index(inplace=True)
        df = utils.fast_merge(df, bond_df_gr, on=["molecule_name","atom_index_1"])
        df["CC_L2dist_diff_maxmin"+suffix] = \
                df["CC_L2dist_max"+suffix]-df["CC_L2dist_min"+suffix]
        del df["CC_L2dist_max"+suffix],df["CC_L2dist_min"+suffix]
        df["diff_CH_CC_L2dist_mean"+suffix] = \
                df["CH_L2dist_mean"+suffix] - df["CC_L2dist_mean"+suffix]
        #df["rel_CH_CC_L2dist_mean"+suffix] = \
        #        df["CH_L2dist_mean"+suffix] / df["CC_L2dist_mean"+suffix]
        # COの結合
        agg_recipe = {"L2dist":["min","max","mean"]}
        bond_df_gr = bond_df[(bond_df.atom_0=="O")&(bond_df.atom_1=="C")]\
                .groupby(["molecule_name","atom_index_1"])["L2dist"].agg(agg_recipe)
        bond_df_gr.columns = pd.Index(
                ["CO_"+e[0]+"_"+e[1]+suffix for e in bond_df_gr.columns.tolist()])
        bond_df_gr.reset_index(inplace=True)
        df = utils.fast_merge(df, bond_df_gr, on=["molecule_name","atom_index_1"])
        df["CO_L2dist_diff_maxmin"+suffix] = \
                df["CO_L2dist_max"+suffix]-df["CO_L2dist_min"+suffix]
        del df["CO_L2dist_max"+suffix],df["CO_L2dist_min"+suffix]
        df["diff_CH_CO_L2dist_mean"+suffix] = \
                df["CH_L2dist_mean"+suffix] - df["CO_L2dist_mean"+suffix]
        #df["rel_CH_CO_L2dist_mean"+suffix] = \
        #        df["CH_L2dist_mean"+suffix] / df["CO_L2dist_mean"+suffix]
        df["diff_CC_CO_L2dist_mean"+suffix] = \
                df["CC_L2dist_mean"+suffix] - df["CO_L2dist_mean"+suffix]
        #df["rel_CC_CO_L2dist_mean"+suffix] = \
        #        df["CC_L2dist_mean"+suffix] / df["CO_L2dist_mean"+suffix]
        return df
    def __call__(self, train_df, test_df, st_df):
        train_bonds_df = pd.read_csv("../external/train_bonds.csv")
        test_bonds_df = pd.read_csv("../external/test_bonds.csv")
        train_df = self.each_make(train_df, train_bonds_df, st_df)
        test_df = self.each_make(test_df, test_bonds_df, st_df)
        return train_df, test_df 

class Feature_3Jxx(CreateFeature):
    def each_make(self, df, mc_df):
        suffix = "_3Jxx"
        # atom_2, atom_30,atom_31, atom_32
        for ix in ["2","30","31","32"]:
            df = pd.merge(df, mc_df, 
                    left_on=["molecule_name",f"atom_index_{ix}"], 
                    right_on=["molecule_name","atom_index"], 
                    how="left"
            )
            df.rename(columns={"mulliken_charge":f"mc_atom_{ix}"+suffix},inplace=True)
        return df
    def __call__(self, train_df, test_df, st_df):
        train_mc = pd.read_pickle("../pickle/atomic_meta_train.pkl")
        test_mc = pd.read_pickle("../pickle/atomic_meta_test.pkl")
        train_df = self.each_make(train_df, train_mc)
        test_df = self.each_make(test_df, test_mc)
        return train_df, test_df 

class Feature_2Jxx(CreateFeature):
    def each_make(self, df, mc_df):
        suffix = "_2Jxx"
        # atom_2, atom_30,atom_31, atom_32
        for ix in ["2"]:
            df = pd.merge(df, mc_df, 
                    left_on=["molecule_name",f"atom_index_{ix}"], 
                    right_on=["molecule_name","atom_index"], 
                    how="left"
            )
            df.rename(columns={"mulliken_charge":f"mc_atom_{ix}"+suffix},inplace=True)
        return df
    def __call__(self, train_df, test_df, st_df):
        train_mc = pd.read_pickle("../pickle/atomic_meta_train.pkl")
        test_mc = pd.read_pickle("../pickle/atomic_meta_test.pkl")
        train_df = self.each_make(train_df, train_mc)
        test_df = self.each_make(test_df, test_mc)
        return train_df, test_df 

class SimpleDistancFeature(CreateFeature):
    def each_make(self, df, mode):
        for type_name in tqdm(ALL_TYPES):
            filepath = f"../data/simple_distance/{mode}_{type_name}.csv"
            feature_df = pd.read_csv(filepath)
            feature_df.rename(
                columns={ c:c+f"_{type_name}" for c in 
                    [c for c in feature_df.columns if c not in 
                        ["molecule_name","atom_index_0","atom_index_1","id"]
                    ]
                },
                inplace=True
            )
            feature_df.drop(["molecule_name","atom_index_0","atom_index_1"],axis=1,inplace=True)
            df = utils.fast_merge(df, feature_df, on=["id"])
        return df
    def __call__(self, train_df, test_df):
        train_df = self.each_make(train_df, mode="train")
        test_df = self.each_make(test_df, mode="test")
        return train_df, test_df

def test_use_truth_mulliken_charges(train_df, test_df):
    #mulliken charge test
    ## mulliken chargeの生データ使ったらどのくらいの精度になるのか試す
    mc = compe_data.read_mulliken_charges()
    train_df = train_df.merge(mc.rename(columns={"mulliken_charge":"mc0"}), 
                                left_on=["molecule_name","atom_index_0"], 
                                right_on=["molecule_name","atom_index"],
                                how="left"
                            )
    del train_df["atom_index"]
    train_df = train_df.merge(mc.rename(columns={"mulliken_charge":"mc1"}), 
                                left_on=["molecule_name","atom_index_1"], 
                                right_on=["molecule_name","atom_index"],
                                how="left"
                            )
    del train_df["atom_index"]
    test_df["mc0"] = np.nan
    test_df["mc1"] = np.nan
    return train_df, test_df

def test_use_truth_magnetic_shield_tensors(train_df, test_df):
    # magnetic shield test
    ## mstの生データ使ったらどのくらいの精度になるのか試す
    mst = compe_data.read_magnetic_shielding_tensors()
    mst_cols = [c for c in mst.columns if c not in ["molecule_name","atom_index"]]
    train_df = train_df.merge(mst.rename(columns={c:c+"_0" for c in mst_cols}), 
                                left_on=["molecule_name","atom_index_0"], 
                                right_on=["molecule_name","atom_index"],
                                how="left"
                            )
    del train_df["atom_index"]
    train_df = train_df.merge(mst.rename(columns={c:c+"_1" for c in mst_cols}), 
                                left_on=["molecule_name","atom_index_1"], 
                                right_on=["molecule_name","atom_index"],
                                how="left"
                            )
    del train_df["atom_index"]
    #内積を利用
    mst_cols0 = [c+"_0" for c in mst_cols]
    mst_cols1 = [c+"_1" for c in mst_cols]

    dot_arr = np.einsum(
            "ijk,ikj->ij",
            train_df[mst_cols0].values.reshape([len(train_df),3,3]),
            train_df[mst_cols1].values.reshape([len(train_df),3,3]),
    )
    print(dot_arr.shape)
    dot_arr = dot_arr.reshape([len(train_df),3])
    train_df = pd.concat([train_df, pd.DataFrame(dot_arr,columns=[f"dot{i}" for i in range(3)])], axis=1)
    train_df.drop(mst_cols0, axis=1, inplace=True)
    train_df.drop(mst_cols1, axis=1, inplace=True)
    for col in mst_cols:
        test_df[col+"_diff"] = np.nan
    
    return train_df, test_df

def test_add_giba_feature(train_df, test_df):
    # kernelにあるgiba_featureがどれだけ影響あるか確認
    ## かぶっている特徴もたくさんあるので実際に使うときには精査（というか作り直し）が必要
    giba_train = pd.read_csv("../data/train_giba_feat.csv")
    giba_test = pd.read_csv("../data/test_giba_feat.csv")
    use_cols = [c for c in giba_train.columns if c[:4] in ["yuka","vand","coul","link"]]
    giba_train = giba_train[use_cols].reset_index(drop=True)
    giba_test = giba_test[use_cols].reset_index(drop=True)
    print(giba_train.columns)
    train_df = utils.fast_concat(train_df, giba_train)
    # dummy
    test_df = utils.fast_concat(test_df, giba_test.sample(len(test_df),replace=True))
    return train_df, test_df


def get_all_no_use_features():
    no_use_feats = []
    for type_name in ALL_TYPES:
        no_use_feat = feature_util.get_no_use_feature(df, importance_df)
        no_use_feats.append(no_use_feat)
    return no_use_feats

def make(train_df,test_df, st_df):
    #最初に全く使わない特徴量を用意しておく
    #no_use_cols = get_all_no_use_feature()
    train_df, test_df = MergeAtomInfo(reset=False).make(train_df, test_df, st_df)
    train_df, test_df = MakeDistance(reset=False).make(train_df, test_df)
    # 実験
    #train_df, test_df = test_use_truth_mulliken_charges(train_df, test_df)
    #train_df, test_df = test_use_truth_magnetic_shield_tensors(train_df, test_df)
    train_df, test_df = TypeMapping(reset=False).make(train_df, test_df)

    #train_df, test_df = NearDistFeat(reset=True).make(train_df, test_df, st_df)
    #train_df, test_df = NearON(reset=True).make(train_df, test_df, st_df)
    #new
    train_df, test_df = HinokkiMeta(reset=False).make(train_df, test_df)
    train_df, test_df = NextBondPerType(reset=False).make(train_df, test_df)
    #train_df, test_df = MullikenTotal(reset=False).make(train_df, test_df, st_df)

    #train_df, test_df = BondAngleFeat012(reset=True).make(train_df, test_df)
    #train_df, test_df = BondAngleFeat123(reset=True).make(train_df, test_df)
    train_df, test_df = OpenBabelMC(reset=False).make(train_df, test_df)
    train_df, test_df = ACSFDescriptor(reset=False).make(train_df, test_df)

    train_df, test_df = MoleculeFeat(reset=False).make(train_df, test_df, st_df)
    train_df, test_df = NBondFeat(reset=False).make(train_df, test_df)
    train_df, test_df = AtomMetaFeat(reset=False).make(train_df, test_df)
    #train_df, test_df = AngleFeat(reset=False).make(train_df, test_df)
    train_df, test_df = CompareOtherAtoms(reset=False).make(train_df, test_df, st_df)
    train_df, test_df = GroupAtomIndex1(reset=False).make(train_df, test_df, st_df)
    train_df, test_df = SortedDist(reset=False).make(train_df, test_df)
    #train_df, test_df = AtomBondFeat(reset=False).make(train_df, test_df)
    train_df, test_df = NextACSFDescriptor(reset=False).make(train_df, test_df)
    with utils.timer("reduce memory"):
        train_df = utils.reduce_memory(train_df)
        test_df = utils.reduce_memory(test_df)
    train_df, test_df = SimpleDistancFeature(reset=False).make(train_df, test_df)
    # typeごとに専用の特徴量を作る
    train_df, test_df = Feature_1JHC(reset=False).make(train_df, test_df, st_df)
    with utils.timer("reduce memory"):
        train_df = utils.reduce_memory(train_df)
        test_df = utils.reduce_memory(test_df)
    ##train_df, test_df = Feature_3Jxx(reset=False).make(train_df, test_df, st_df)
   
    train_df["bond_atom1_C_1JHC"],uniques = pd.factorize(train_df["bond_atom1_C_1JHC"])
    test_df["bond_atom1_C_1JHC"] = uniques.get_indexer(test_df["bond_atom1_C_1JHC"])
    
    # add giba feature test
    #train_df, test_df = test_add_giba_feature(train_df, test_df)


    cat_cols = [
        #"bond_atom1_C_1JHC",
        #"new_type"
    ]
    # drop low importance
    drop_low_importance = False
    drop_threshold = 1000
    if drop_low_importance:
        train_df, test_df = \
                feature_util.drop_low_importance(train_df, test_df, 
                    "./importance/f_importance.csv", threshold=drop_threshold)
    #指定したcategorical列が全てあるかわからないためここでフィルタリング
    cat_cols = [c for c in cat_cols if c in train_df.columns]
    gc.collect()
    return train_df, test_df, cat_cols

