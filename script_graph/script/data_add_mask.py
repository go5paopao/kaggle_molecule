import pickle, os, gc, random
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool

save_dir = Path("../../data/graph_v3")
save_dir.mkdir(exist_ok=True)

"""
def make_per_molecule_mask_atom(molecule):
    #graphデータを読み込む
    with open("../../data/graph_v2/{}.pickle".format(molecule),"rb") as f:
        #tupleだとデータ追加出来ないのでlistにする
        graph = list(pickle.load(f))
    #dictで保存
    edge_mask_dict = {}
    node_mask_dict = {}
    mol_st = st_gr.get_group(molecule)
    mol_meta = meta_gr.get_group(molecule)
    ## 相互の距離を求めるためにst同士merge
    mol_st = mol_st.merge(mol_st, on="molecule_name", how="left", suffixes=("_0","_1"))
    ## 同じ原子を除く
    mol_st = mol_st[mol_st.atom_index_0 != mol_st.atom_index_1]
    ## 距離を計算
    mol_st["dist"] = np.linalg.norm(mol_st[["x_0","y_0","z_0"]].values 
                                    - mol_st[["x_1","y_1","z_1"]].values, axis=1)
    ## couplingのindexペア
    coupling_pair_indices = mol_meta[["atom_index_0","atom_index_1"]].values
    pair_indices = mol_st[["atom_index_0","atom_index_1"]].values
    atom_indices = mol_st["atom_index_0"].unique()
    molecule_ids = mol_meta.index.values
    for coupling_idxs,df_id in zip(coupling_pair_indices, molecule_ids):
        # coupling毎に近傍のedge_index, node_indexをidをキーにしたdictで保存
        near_atom_index = np.unique(
            np.concatenate([
                mol_st[
                    (mol_st["dist"]<r_cutoff)&(mol_st.atom_index_0.isin(coupling_idxs))
                    ]["atom_index_1"].values,
                coupling_idxs,
            ])
        )
        edge_mask_dict[df_id] = np.isin(pair_indices, near_atom_index).any(axis=1)
        node_mask_dict[df_id] = np.isin(atom_indices, near_atom_index)
    #graphに追加
    graph.append(edge_mask_dict)
    graph.append(node_mask_dict)
    if random.random() < 0.01:
        print(len(os.listdir("../../data/graph_v3/")) / all_file_num, end="")
    with open(save_dir / f"{molecule}.pickle", "wb") as f:
        pickle.dump(graph, f)
"""

def make_per_molecule_mask_atom(molecule):
    #graphデータを読み込む
    with open("../../data/graph_v2/{}.pickle".format(molecule),"rb") as f:
        #tupleだとデータ追加出来ないのでlistにする
        graph = list(pickle.load(f))
    #dictで保存
    mol_st = st_gr.get_group(molecule)
    mol_meta = meta_gr.get_group(molecule)
    ## 相互の距離を求めるためにst同士merge
    mol_st = mol_st.merge(mol_st, on="molecule_name", how="left", suffixes=("_0","_1"))
    ## 同じ原子を除く
    mol_st = mol_st[mol_st.atom_index_0 != mol_st.atom_index_1]
    ## 距離を計算
    mol_st["dist"] = np.linalg.norm(mol_st[["x_0","y_0","z_0"]].values 
                                    - mol_st[["x_1","y_1","z_1"]].values, axis=1)
    #graphに追加
    with open(save_dir / f"{molecule}.pickle", "wb") as f:
        pickle.dump(graph, f)



if __name__ == "__main__":
    r_cutoff = 2.5
    #mode = "train"
    mode = "test"
    meta_df = pd.read_pickle(f"../../pickle/{mode}.pkl").set_index("id")
    molecules = meta_df["molecule_name"].unique().tolist()
    mc_df = pd.read_pickle(f"../../pickle/atomic_meta_{mode}.pkl")
    ob_mc_df = pd.read_pickle(f"../../data/{mode}_ob_charges.csv")
    del ob_mc_df["Unnamed: 0"]
    st_df = pd.read_pickle("../../pickle/structures.pkl")
    ## train or validのstructureに絞る
    st_df = st_df[st_df.molecule_name.isin(molecules)]
    # 分子単位に処理
    meta_gr = meta_df.groupby("molecule_name")
    st_gr = st_df.groupby("molecule_name")
    mc_gr = mc_df.groupby("molecule_name")
    ob_mc_gr = ob_mc_df.groupby("molecule_name")
    all_file_num = len(molecules)
    with Pool(4) as p:
        p.map(make_per_molecule, molecules)


