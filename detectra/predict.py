import numpy as np
import pandas as pd
import joblib
import os # Make sure os is imported for path handling

# === Define ===
COMMON_AXIS = np.linspace(4000, 400, 900)
# Ensure paths are correct relative to where app.py runs
compound_files = {
    "cocaine": os.path.join("Training Data", "cocaine.csv"),
    "morphine": os.path.join("Training Data", "morphine.csv"),
    "heroin": os.path.join("Training Data", "heroin.csv"),
    "methadone": os.path.join("Training Data", "methadone.csv"),
    "meth": os.path.join("Training Data", "meth.csv"),
    "sucrose": os.path.join("Training Data", "sucrose.csv"),
    "lactic": os.path.join("Training Data", "lactic.csv"),
    "glucose": os.path.join("Training Data", "glucose.csv"),
    "ethanol": os.path.join("Training Data", "ethanol.csv"),
    "citric": os.path.join("Training Data", "citric.csv")
}
valid_compounds = list(compound_files.keys())
drug_compounds = ["cocaine", "morphine", "heroin", "methadone", "meth"]

# === Load Ensemble Chains & Label Binarizer ===
# These will be loaded when the module is imported by app.py
# Ensure model files exist from running multi.py standalone.
chains = None
mlb = None
try:
    chains = joblib.load("model/ensemble_classifier_chains.pkl")
    mlb = joblib.load("model/multidrug_label_binarizer.pkl")
except FileNotFoundError:
    print("Error: Model files for multi-compound prediction not found. Please run multi.py to train models first.")
    # Allow app.py to handle this by checking if chains/mlb are None

# === Load & Interpolate ===
def load_and_interpolate_spectrum(path):
    df = pd.read_csv(path, usecols=["wavenumber", "absorbance"])
    x = df["wavenumber"].values
    y = df["absorbance"].values
    if x[0] > x[-1]:
        x = x[::-1]
        y = y[::-1]
    return np.interp(COMMON_AXIS, x, y)

# No if __name__ == "__main__": block here.
# This script is now purely a module for app.py to import and use its functions.
