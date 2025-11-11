# src/config.py

# === File Paths ===
TRAIN_PATH = "data/d00_tr.csv"
CASE_DIR = "data/cases"
MODEL_SAVE_ROOT = "results"

# === Experiment Parameters ===
#SAMPLE_SIZES = [5,10,20,40,80,160,320,500]   # You can add more later
NUM_SIMULATIONS = [500,1000,1500, 2500,4000,6500,11000,18500,30500,51500,86000,144000,200000]
#NUM_SIMULATIONS = [4000]
SEEDS = [0, 1, 3, 7, 8]#10, 11, 13, 17, 21, 23, 27, 32, 42, 43,
    #55, 77, 101, 123, 256, 512, 999, 1111, 1337, 1995, 2024, 2025, 2048, 3141, 9001] # For error bars


NUM_EPOCHS = 500
LEARNING_RATE = 1.99e-3
PATIENCE = 15
BATCH_SIZE = 256
WEIGHT_DECAY = 0.017
SCHEDULER_FACTOR = 0.973

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
#DROP_COLUMNS = None