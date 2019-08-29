import pickle, os, gc, random
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool

"""
node: oofのmulliken_charge, openbabelのmulliken_charge
edge: 3ノード間の角度
"""


save_dir = Path("../../data/graph_v4")
save_dir.mkdir(exist_ok=True)

def make_per_molecule(molecule):
    #graphデータを読み込む
    with open("../../data/graph_v2/{}.pickle".format(molecule),"rb") as f:
        #tupleだとデータ追加出来ないのでlistにする
        graph = list(pickle.load(f))
    mol_st = st_gr.get_group(molecule)
    mol_meta = meta_gr.get_group(molecule)
    mol_mc = mc_gr.get_group(molecule)
    mol_ob_mc = ob_mc_gr.get_group(molecule)
    #以下cosineを計算
    ##edgeのdf作成
    edge_arr = np.concatenate(graph[4], -1)
    edge_idx_df = pd.DataFrame(graph[5],columns=["atom_index_0","atom_index_1"])
    edge_df = pd.concat(
        [pd.DataFrame(edge_idx_df,columns=["atom_index_0","atom_index_1"]), 
         pd.DataFrame(edge_arr, columns=["single","double","triple","benzen","dist","g_angle"])],
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
    edge_df = edge_df[(edge_df.is_bond_0 == 1)&(edge_df.is_bond_1 == 1)]
    ##xyz情報を追加
    for i in range(3):
        edge_df = edge_df.merge(
            mol_st[["atom_index","atom","x","y","z"]].rename(
                columns={"atom_index":f"atom_index_{i}","x":f"x_{i}","y":f"y_{i}","z":f"z_{i}"}
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
    #edge_df = edge_df[["atom_index_0","atom_index_2","cos_012"]]
    #同じatom_index_0と1がある場合、最大角(cosが小さい)のものを選ぶようにする
    ##最大のものが一番距離が近い
    ##nanの場合はcos=1(無限遠を想定)としておく
    edge_df = edge_df.groupby(["atom_index_0","atom_index_2"])["cos_012"].min().reset_index()
    cos012 = edge_idx_df.merge(edge_df.rename(columns={"atom_index_2":"atom_index_1"}),
                on=["atom_index_0","atom_index_1"], 
                how="left")["cos_012"].fillna(1).values.reshape([-1,1])
    if (graph[4][0].shape[0] != cos012.shape[0]):
        print(graph[4][0].shape, cos012.shape)
    ## edgeにcos情報を追加
    graph[4].append(cos012)
    
    #mulliken_chargeをnodeに追加
    graph[3].append(mol_mc["mulliken_charge"].values.reshape([-1,1]))
    graph[3].append(
        mol_ob_mc[[c for c in mol_ob_mc.columns if c not in ["molecule_name","atom_index"]]].values
    )
    #graphを保存
    with open(save_dir / f"{molecule}.pickle", "wb") as f:
        pickle.dump(graph, f)

if __name__ == "__main__":
    mode = "train"
    #mode = "test"
    meta_df = pd.read_pickle(f"../../pickle/{mode}.pkl").set_index("id")
    molecules = meta_df["molecule_name"].unique().tolist()
    mc_df = pd.read_pickle(f"../../pickle/atomic_meta_{mode}.pkl")
    mc_df["mulliken_charge"].fillna(0,inplace=True)
    ob_mc_df = pd.read_csv(f"../../data/{mode}_ob_charges.csv")
    ob_mc_df.fillna(0, inplace=True)
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
    #for i in tqdm(range(100)):
    #    make_per_molecule(molecules[i])


