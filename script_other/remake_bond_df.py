import pandas as pd
import numpy as np
import os,sys,pickle
from multiprocessing import Pool
sys.path.append("../script")
pd.set_option("max_columns",100)
from utils import utils

"""
bond情報をカーネルのものを使っていたが、
信頼性が低そうなため、RDkitベースのものに作り直す
"""

def map_func(x):
    if x == "HC":
        return "CH"
    elif x == "OC":
        return "CO"
    elif x == "NC":
        return "CN"
    elif x == "NH":
        return "HN"
    elif x =="OH":
        return "HO"
    elif x =="FC":
        return "CF"
    elif x == "ON":
        return "NO"
    else:
        return x

def make_bond(molecule_name):
    with open(f"../data/graph_v2/{molecule_name}.pickle", "rb") as f:
        graph = pickle.load(f)
    bond_arr = graph[4][0]
    dist_arr = graph[4][1]
    edge_arr = graph[5]
    df = pd.DataFrame(
            np.concatenate([edge_arr, bond_arr],axis=-1),
            columns = ["atom_index_0","atom_index_1",
                "single","double","triple","aromatic"],
            dtype=np.int8
    )
    df["L2dist"] = dist_arr
    bond_type_arr =np.argmax(df[["single","double","triple","aromatic"]].values,axis=1)
    df["bond_type"] = bond_type_arr
    df.loc[(df["single"]==0)&(df["bond_type"]==0),"bond_type"] = -1
    atom_df = pd.DataFrame(
            np.argwhere(graph[3][0]),
            columns = ["atom_index","atom"]
    )
    df = df.merge(atom_df, left_on="atom_index_0", right_on="atom_index", how="left")
    del df["atom_index"]
    df = df.merge(atom_df, left_on="atom_index_1", right_on="atom_index", how="left")
    del df["atom_index"]
    df.rename(columns={"atom_x":"atom_0","atom_y":"atom_1"},inplace=True)
    df["bond_atom"] = df["atom_0"].map(atom_dict) + df["atom_1"].map(atom_dict)
    df["bond_atom"] = df["bond_atom"].map(map_func)
    df["molecule_name"] = molecule_name
    return df

atom_dict = {0:"H",1:"C",2:"N",3:"O",4:"F"}
#mode = "train"
mode = "test"
meta_df = pd.read_pickle(f"../pickle/{mode}.pkl")
molecules = meta_df["molecule_name"].unique()

with Pool(4) as p:
    res = p.map(make_bond, molecules)

all_df = pd.concat(res,axis=0).reset_index(drop=True)
utils.save_pickle(all_df,f"../pickle/{mode}_bond_v2.pkl")
# save only connect info
only_bond_df = all_df.loc[all_df.bond_type!=-1,["molecule_name","atom_index_0","atom_index_1"]]
utils.save_pickle(only_bond_df,f"../pickle/{mode}_bond_v2_only_bond.pkl")

