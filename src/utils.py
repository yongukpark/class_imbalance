# utils.py
import numpy as np
import random
import warnings

warnings.filterwarnings("ignore")

# ===== Seed 고정 =====
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)