# src/hypertune/config.py

# === File Paths ===
TRAIN_PATH = "data/cases/d00.csv"
VAL_PATH = "data/cases/d00_te.csv"
MODEL_SAVE_ROOT = "hypertune/results"

# === Experiment Parameters ===
SEED = 43
NUM_EPOCHS = 500
PATIENCE = 15
BATCH_SIZE = 256

# === Columns to Drop ===
DROP_COLUMNS = [
    'Conc_A_ReactorFeedStream6_molPct', 'Conc_B_ReactorFeedStream6_molPct',
    'Conc_C_ReactorFeedStream6_molPct', 'Conc_D_ReactorFeedStream6_molPct',
    'Conc_E_ReactorFeedStream6_molPct', 'Conc_F_ReactorFeedStream6_molPct',
    'Conc_A_PurgeStream9_molPct', 'Conc_B_PurgeStream9_molPct',
    'Conc_C_PurgeStream9_molPct', 'Conc_D_PurgeStream9_molPct',
    'Conc_E_PurgeStream9_molPct', 'Conc_F_PurgeStream9_molPct',
    'Conc_G_PurgeStream9_molPct', 'Conc_H_PurgeStream9_molPct',
    'Conc_D_StrprUFlowStream11_molPct', 'Conc_E_StrprUFlowStream11_molPct',
    'Conc_F_StrprUFlowStream11_molPct', 'Conc_G_StrprUFlowStream11_molPct',
    'Conc_H_StrprUFlowStream11_molPct'
]