import os,sys,gc,pickle,itertools
import numpy as np
import pandas as pd
import logging
from functools import partial
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
#original module
import compe_data
from utils import utils,feature_util
from utils.feature_util import CreateFeature
import validation

TARGET_COL = "mulliken_charge"
GROUP_COL = "molecule_name"
EXCEPT_FEATURES = [
        "id",
        #"x","y","z",
        "type",
        "atom",
        "molecule_name",
        "atom_index",
        "scalar_coupling_constant",
        "mulliken_charge"
]


def make_factorize(train_df, test_df, fact_cols):
    for col in fact_cols:
        train_df[col], uniques = pd.factorize(train_df[col], sort=True)
        test_df[col] = uniques.get_indexer(test_df[col])
    return train_df, test_df

class A_BondInfo(CreateFeature):
    def each_make(self, df, bond_df):
        def replace_concat(df):
            df = pd.concat([
                df,
                df.rename(columns={"atom_index_0":"atom_index_1","atom_index_1":"atom_index_0"})
            ],axis=0)
            return df
        bond_df = replace_concat(bond_df)
        bond_df_gr = bond_df.groupby(["molecule_name","atom_index_0"]).agg({
                    "L2dist":["mean","std","min","max"],
                    "nbond":["max"],
                    "atom_index_1":["count"],
                    "bond_type":["nunique"],
                    })
        bond_df_gr.columns = pd.Index([e[0]+"_"+e[1] for e in bond_df_gr.columns.tolist()])
        bond_df_gr.reset_index(inplace=True)
        df = df.merge(bond_df_gr, left_on=["molecule_name","atom_index"],
                right_on=["molecule_name","atom_index_0"], how="left")
        return df
    
    def __call__(self, train_df, test_df):
        train_bonds_df = pd.read_csv("../external/train_bonds.csv")
        test_bonds_df = pd.read_csv("../external/test_bonds.csv")
        train_df = self.each_make(train_df, train_bonds_df)
        test_df = self.each_make(test_df, test_bonds_df)
        return train_df, test_df

#p_0 = df[['x_0', 'y_0', 'z_0']].values
#p_1 = df[['x_1', 'y_1', 'z_1']].values
#df['dist'] = np.linalg.norm(p_0 - p_1, axis=1)

class A_AtomBasicFeat(CreateFeature):
    def each_make(self, df):
        # atom to int
        atom_map = {'H':0, 'C':1, 'N':2, 'O':3, 'F':4}
        df["atom_int"] = df["atom"].map(atom_map)
        # atom number in same molecule
        df["n_atoms_in_molecule"] = df.groupby("molecule_name")["atom_index"].transform("count")
        # atom number of same atom type in same molecule
        df["n_same_atoms_in_molecule"] = df.groupby(["molecule_name","atom"])["atom_index"].transform("count")
        # ratio of same atom in molecule
        df["ratio_same_atom_in_molecule"] = df["n_same_atoms_in_molecule"] / df["n_atoms_in_molecule"]
        # atom number of per atom type in same molecule
        for atom_type in atom_map.keys():
            df["n_{}_atoms_in_molecule".format(atom_type)] = \
                    df[df.atom == atom_type].groupby("molecule_name")["atom_index"].transform("count")
        return df

    def __call__(self, train_df, test_df):
        train_df = self.each_make(train_df)
        test_df = self.each_make(test_df)
        return train_df,test_df


def read_graph(molecule_name):
    with open("../data/graph/{}.pickle".format(molecule_name),"rb") as f:
        data = pickle.load(f)[2]
        data = data[:,5:-1]
        return (molecule_name,data)


class A_AtomInfoFromGraphData(CreateFeature):
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
        #print(atom_df.head())
        #print(df.columns)
        df = df.merge(atom_df, on=["molecule_name","atom_index"], how="left")
        #df = utils.fast_merge(df, atom_df, on=["molecule_name","atom_index"])
        return df
        
    def __call__(self, train_df, test_df):
        train_df = self.each_make(train_df)
        test_df = self.each_make(test_df)
        return train_df, test_df

class A_ACSFDescriptor(CreateFeature):
    def each_make(self, df, acsf_df):
        #merge to atom_index
        df = df.merge(acsf_df, 
                on=["molecule_name","atom_index"],
                how="left"
        )
        return df
             
    def __call__(self, train_df, test_df):
        acsf_df = pd.read_pickle("../pickle/acsf_array.pkl")
        train_df = self.each_make(train_df,acsf_df)
        test_df = self.each_make(test_df,acsf_df)
        return train_df, test_df



def make(train_df,test_df, train_pair, test_pair):
    train_df, test_df = A_BondInfo(reset=True).make(train_df, test_df)
    train_df, test_df = A_AtomBasicFeat(reset=False).make(train_df, test_df)
    train_df, test_df = A_AtomInfoFromGraphData(reset=False).make(train_df, test_df)
    train_df, test_df = A_ACSFDescriptor(reset=True).make(train_df, test_df)
    #train_df, test_df = MakeDistance(reset=False).make(train_df, test_df)
    #train_df, test_df = TypeMapping(reset=True).make(train_df, test_df)
    cat_cols = [
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
    return train_df, test_df, cat_cols

