import os
import pandas as pd

def read_data(csv_path, pkl_path):
    if os.path.exists(pkl_path):
        df = pd.read_pickle(pkl_path)
    else:
        df = pd.read_csv(csv_path)
        df.to_pickle(pkl_path)
    return df

def read_train():
    pkl_path = "../pickle/train.pkl"
    csv_path = "../input/train.csv"
    df = read_data(csv_path, pkl_path)
    return df

def read_test():
    pkl_path = "../pickle/test.pkl"
    csv_path = "../input/test.csv"
    df = read_data(csv_path, pkl_path)
    return df

def read_structures():
    pkl_path = "../pickle/structures.pkl"
    csv_path = "../input/structures.csv"
    df = read_data(csv_path, pkl_path)
    return df

def read_potential_energy():
    pkl_path = "../pickle/potential_energy.pkl"
    csv_path = "../input/potential_energy.csv"
    df = read_data(csv_path, pkl_path)
    return df

def read_mulliken_charges():
    pkl_path = "../pickle/mulliken_charges.pkl"
    csv_path = "../input/mulliken_charges.csv"
    df = read_data(csv_path, pkl_path)
    return df

def read_magnetic_shielding_tensors():
    pkl_path = "../pickle/magnetic_shielding_tensors.pkl"
    csv_path = "../input/magnetic_shielding_tensors.csv"
    df = read_data(csv_path, pkl_path)
    return df

def read_dipole_moments():
    pkl_path = "../pickle/dipole_moments.pkl"
    csv_path = "../input/dipole_moments.csv"
    df = read_data(csv_path, pkl_path)
    return df

def read_scalar_coupling_contributions():
    pkl_path = "../pickle/scalar_coupling_contributions.pkl"
    csv_path = "../input/scalar_coupling_contributions.csv"
    df = read_data(csv_path, pkl_path)
    return df

def read_sample_submission():
    pkl_path = "../pickle/sample_submission.pkl"
    csv_path = "../input/sample_submission.csv"
    df = read_data(csv_path, pkl_path)
    return df

   
