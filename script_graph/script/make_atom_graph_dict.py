import pandas as pd
import numpy as np
import os,pickle,gc
from tqdm import tqdm

"""
今まで分子単位だったデータをcoupling単位にする
"""
def make_idx_dict(mode, r_cutoff=3.0):
    meta_df = pd.read_pickle(f"../../pickle/{mode}.pkl").set_index("id")
    molecules = meta_df["molecule_name"].unique()
    st_df = pd.read_pickle("../../pickle/structures.pkl")
    ## train or validのstructureに絞る
    st_df = st_df[st_df.molecule_name.isin(molecules)]
    # 分子単位に処理
    edge_mask_dict = {}
    node_mask_dict = {}
    meta_gr = meta_df.groupby("molecule_name")
    st_gr = st_df.groupby("molecule_name")
    for molecule in tqdm(molecules):
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
    return edge_mask_dict, node_mask_dict 
        
if __name__ == "__main__":
    #train
    #edge_dict, node_dict = make_idx_dict("train", r_cutoff=3.0)
    #with open("../../pickle/train_edge_idx_dict.pkl","wb") as f:
    #    pickle.dump(edge_dict,f, protocol=3)
    #with open("../../pickle/train_node_idx_dict.pkl","wb") as f:
    #    pickle.dump(node_dict,f, protocol=3)
    #test
    edge_dict, node_dict = make_idx_dict("test", r_cutoff=3.0)
    with open("../../pickle/test_edge_idx_dict.pkl","wb") as f:
        pickle.dump(edge_dict,f, protocol=3)
    with open("../../pickle/test_node_idx_dict.pkl","wb") as f:
        pickle.dump(node_dict,f, protocol=3)

