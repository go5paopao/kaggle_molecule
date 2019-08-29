import pandas as pd
import numpy as np
from tqdm import tqdm
from dscribe.descriptors import ACSF
from dscribe.core.system import System
from multiprocessing import Pool
import compe_data

g2_params=[[1,1],[1,2],[1,3]]
g4_params=[[1,1,1],[1,2,1],[1,1,-1],[1,2,-1]]

gen = ACSF(
        species = ["H","C","N","O","F"],
        rcut = 6.0,
        g2_params = g2_params,
        g4_params = g4_params,
)
st_df = compe_data.read_structures()
molecules = st_df["molecule_name"].unique()
st_gr = st_df.groupby("molecule_name")
# make st_dict for pararells
st_dict = {}
for molecule in tqdm(molecules):
    st_dict[molecule] = st_gr.get_group(molecule)


def func_acsf(params):
    i, molecule = params
    #if i%1000 == 0:
    #    print(f"{i}th finish")
    st = st_dict[molecule]
    atoms = System(symbols=st["atom"].values, positions=st[["x","y","z"]].values)
    return gen.create(atoms)

def make_acsf():
    # pararell process
    params = [(i,molecule) for i,molecule in enumerate(molecules)]
    with Pool(4) as p:
        res = p.map(func_acsf, params)
    acsf_cols = [f"acsf_{i}" for i in range(res[0].shape[1])]
    acsf_df = pd.DataFrame(np.concatenate(res),columns=acsf_cols)
    acsf_df["molecule_name"] = st_df["molecule_name"]
    acsf_df["atom_index"] = st_df["atom_index"]
    acsf_df.to_pickle("../pickle/acsf_array.pkl")

if __name__ == "__main__":
    make_acsf()
