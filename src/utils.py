import random
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import yaml

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

#def prepare_data(df, sample_size, seed):
    # Randomly sample `sample_size` rows with the given seed
#    df_sample = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    # Split into train/val
#    X_train, X_val = train_test_split(df_sample, test_size=0.2, random_state=seed)

    # Normalize
#    scaler = StandardScaler()
#    X_train_scaled = scaler.fit_transform(X_train)
#    X_val_scaled = scaler.transform(X_val)

#    return X_train_scaled, X_val_scaled, scaler

def prepare_data(df, n_simulations, seed):
    
    # 1. Get all unique simulation IDs
    sim_ids = df['sim_id'].unique()

    # 2. Randomly choose n_simulations
    selected_sims = np.random.choice(sim_ids, size=n_simulations, replace=False)

    # 3. Subset the DataFrame to the selected simulations
    df_subset = df[df['sim_id'].isin(selected_sims)].copy()

    # 4. Simulation-aware split: 75% train, 25% val
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(gss.split(df_subset, groups=df_subset['sim_id']))

    df_train = df_subset.iloc[train_idx]
    df_val = df_subset.iloc[val_idx]

    # 5. Drop non-feature columns (keep only input variables)
    feature_cols = df.columns.drop('sim_id')  # Extend this list if needed
    X_train = df_train[feature_cols].values
    X_val = df_val[feature_cols].values

    # 6. Normalize using training set statistics only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    return X_train_scaled, X_val_scaled, scaler

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_data_from_train_points(df, train_point_target, seed, sim_len=500):
    """
    Prepares training and validation data for a target number of training points,
    maintaining an 80/20 split, using full simulations where possible and slicing
    the last one if needed. Prints and returns sim IDs used.

    Returns:
        X_train_scaled, X_val_scaled, scaler, train_sim_ids, val_sim_ids
    """
    np.random.seed(seed)

    # === Step 1: Get simulation IDs and compute sizes ===
    sim_ids = df['sim_id'].unique()
    required_train_sims = int(np.ceil(train_point_target / sim_len))
    actual_train_points = required_train_sims * sim_len

    total_points = int(actual_train_points / 0.8)
    val_points_needed = total_points - actual_train_points
    val_sims_needed = int(np.ceil(val_points_needed / sim_len))

    total_sims_needed = required_train_sims + val_sims_needed

    # === Step 2: Sample simulations ===
    selected_sims = np.random.choice(sim_ids, size=total_sims_needed, replace=False)
    train_sim_ids = selected_sims[:required_train_sims]
    val_sim_ids = selected_sims[required_train_sims:]

    # === Step 3: Extract Data ===
    df_train = df[df['sim_id'].isin(train_sim_ids)].copy()
    df_val_pool = df[df['sim_id'].isin(val_sim_ids)].copy()
    df_val = df_val_pool.iloc[:val_points_needed]  # slice to exact val size

    # === Step 4: Feature extraction ===
    feature_cols = df.columns.drop('sim_id')
    X_train = df_train[feature_cols].values
    X_val = df_val[feature_cols].values

    # === Step 5: Normalize ===
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # === Step 6: Print simulation IDs ===
    print(f"🧪 Train sim_ids: {train_sim_ids}")
    print(f"🧪 Val sim_ids: {val_sim_ids}")
    print(f"✅ Train samples: {X_train.shape[0]} | Val samples: {X_val.shape[0]}")

    return X_train_scaled, X_val_scaled, scaler



def load_model_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_name = config.pop("model_name")
    return model_name, config