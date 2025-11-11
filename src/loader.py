import pandas as pd
import os

def load_training_data(path="data/d00_tr.csv", drop_columns=None):
    df = pd.read_csv(path)
    if drop_columns:
        df = df.drop(columns=drop_columns, errors='ignore')
    return df

def load_all_cases(case_dir="data/cases", drop_columns=None):
    cases = {}
    for i in range(1, 22):
        fname = f"d{i:02d}_te.csv"
        fpath = os.path.join(case_dir, fname)
        df = pd.read_csv(fpath)
        if drop_columns:
            df = df.drop(columns=drop_columns, errors='ignore')
        cases[f"case_{i:02d}"] = df
    return cases