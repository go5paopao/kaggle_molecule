import pickle, os, gc, random
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from utils import utils
pd.set_option("max_columns",300)

"""
node: oofのmulliken_charge, openbabelのmulliken_charge
edge: 3ノード間の角度
"""

def make_per_molecule(molecule):
    #graphデータを読み込む
    with open("../data/graph_v2/{}.pickle".format(molecule),"rb") as f:
        #tupleだとデータ追加出来ないのでlistにする
        graph = pickle.load(f)
    mol_st = st_dict[molecule]
    #以下cosineを計算
    #edge_idx_df = pd.DataFrame(graph[5],columns=["atom_index_0","atom_index_1"])
    edge_df = pd.concat(
        [pd.DataFrame(graph[5], columns=["atom_index_0","atom_index_1"]), 
            pd.DataFrame(np.concatenate(graph[4][:-1], -1), 
             columns=["single","double","triple","benzen","dist"])],
        axis=1
    )
    edge_df = edge_df.merge(
                    edge_df.rename(columns={"atom_index_0":"atom_index_2"}),
                    on = "atom_index_1",
                    suffixes=("_0","_1")
    )
    ##同じ原子に戻ってくるのを除く
    edge_df = edge_df[edge_df.atom_index_0 != edge_df.atom_index_2]
    ##結合しているものに絞る
    edge_df["is_bond_0"] = edge_df[["single_0","double_0","triple_0","benzen_0"]].max(axis=1)
    edge_df["is_bond_1"] = edge_df[["single_1","double_1","triple_1","benzen_1"]].max(axis=1)
    edge_df = edge_df[
            (
                (edge_df["single_0"] > 0) |
                (edge_df["double_0"] > 0) |
                (edge_df["triple_0"] > 0) |
                (edge_df["benzen_0"] > 0)
            )
            &
            (
                (edge_df["single_1"] > 0) |
                (edge_df["double_1"] > 0) |
                (edge_df["triple_1"] > 0) |
                (edge_df["benzen_1"] > 0)
            )]
    ##xyz情報を追加
    for i in range(3):
        edge_df = edge_df.merge(
            mol_st.rename(
                columns={"atom_index":f"atom_index_{i}",
                    "x":f"x_{i}","y":f"y_{i}","z":f"z_{i}","atom":f"atom_{i}"}
            ),
            on = f"atom_index_{i}",
            how="left"
        )
    ##cosineを求める
    vec_0 = edge_df[["x_0","y_0","z_0"]].values
    vec_1 = edge_df[["x_1","y_1","z_1"]].values
    vec_2 = edge_df[["x_2","y_2","z_2"]].values
    edge_df["cos_012"] = np.einsum("ij,ij->i",(vec_0-vec_1),(vec_2-vec_1)) \
                        / (edge_df["dist_0"]*edge_df["dist_1"])
    edge_df.drop(["x_0","x_1","x_2","y_0","y_1","y_2","z_0","z_1","z_2"],axis=1,inplace=True)
    edge_df.to_pickle(f"../data/edge_angle/{molecule}.pkl")


if __name__ == "__main__":
    with utils.timer("make_feature_per_molecule"):
        for mode in ["train","test"]:
            meta_df = pd.read_pickle(f"../pickle/{mode}.pkl").set_index("id")
            molecules = meta_df["molecule_name"].unique().tolist()
            st_df = pd.read_pickle("../pickle/structures.pkl")
            ## train or validのstructureに絞る
            st_df = st_df[st_df.molecule_name.isin(molecules)]\
                    [["molecule_name","atom_index","atom","x","y","z"]]
            # 分子単位に処理
            st_gr = st_df.groupby("molecule_name")
            st_dict = {}
            for molecule in tqdm(molecules):
                st_dict[molecule] = st_gr.get_group(molecule)
            all_file_num = len(molecules)
            with Pool(4) as p:
                 res = p.map(make_per_molecule, molecules)
    with utils.timer("concatenate_molecules_feature"):
        for mode in ["train","test"]:
            meta_df = pd.read_pickle(f"../pickle/{mode}.pkl").set_index("id")
            molecules = meta_df["molecule_name"].unique().tolist()
            df_list = []
            for molecule in tqdm(molecules):
                df_list.append(utils.load_pickle(f"../data/edge_angle/{molecule}.pkl"))
            all_df = pd.concat(df_list).reset_index(drop=True)
            utils.save_pickle(all_df, f"../pickle/{mode}_edge_angle.pkl")
