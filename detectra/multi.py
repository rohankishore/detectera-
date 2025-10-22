import numpy as np
import pandas as pd
import joblib
import os
import warnings
from itertools import combinations
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(__file__)
os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)

# -------------------------------------------------------------------
# 1. CONSTANTS
# -------------------------------------------------------------------
COMMON_AXIS  = np.linspace(4000, 400, 900)

DRUGS        = ["cocaine", "morphine", "heroin", "methadone", "meth"]
NON_DRUGS    = ["sucrose", "lactic", "glucose", "ethanol", "citric"]
ALL_COMPOUNDS = DRUGS + NON_DRUGS
# Use absolute paths to training data relative to this module
TRAINING_DIR = os.path.join(BASE_DIR, "Training Data")
compound_files = {n: os.path.join(TRAINING_DIR, f"{n}.csv") for n in ALL_COMPOUNDS}

# -------------------------------------------------------------------
# 2. HELPERS
# -------------------------------------------------------------------
def load_interp(path: str) -> np.ndarray:
    """Load CSV and interpolate onto COMMON_AXIS."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training file not found: {path}")
    df = pd.read_csv(path, usecols=["wavenumber", "absorbance"])
    x = df["wavenumber"].values
    y = df["absorbance"].values
    if x[0] > x[-1]:
        x = x[::-1]
        y = y[::-1]
    return np.interp(COMMON_AXIS, x, y)

def create_mixtures_and_labels(all_compounds, drugs, common_axis):
    """Generates synthetic mixtures and their multi-labels."""
    X = []
    y_labels = []

    # Load all individual spectra
    spectra = {c: load_interp(compound_files[c]) for c in all_compounds}

    # Single compounds
    for compound in all_compounds:
        X.append(spectra[compound])
        y_labels.append([compound] if compound in drugs else [])

    # Two-compound mixtures
    for c1, c2 in combinations(all_compounds, 2):
        mixture = (spectra[c1] + spectra[c2]) / 2
        X.append(mixture)
        current_labels = []
        if c1 in drugs:
            current_labels.append(c1)
        if c2 in drugs:
            current_labels.append(c2)
        y_labels.append(current_labels)

    # Three-compound mixtures (drug + 2 non-drugs or 2 drugs + 1 non-drug)
    # Adjusted to generate more diverse mixtures up to 5 compounds
    for i in range(3, min(len(all_compounds) + 1, 6)): # Up to 5 compounds in the mix for multi.html
        for combo in combinations(all_compounds, i):
            mixture = np.zeros_like(common_axis)
            current_labels = []
            for compound in combo:
                mixture += spectra[compound]
                if compound in drugs:
                    current_labels.append(compound)
            
            # Only add mixtures that contain at least one drug or are pure non-drug mixtures
            # This helps balance the dataset for multi-label prediction
            if current_labels or (not current_labels and i == 1): # Include pure non-drugs
                mixture /= len(combo)
                X.append(mixture)
                y_labels.append(current_labels)

    return np.array(X), y_labels

def train_multi_models():
    """Trains and saves the multi-label classification models."""
    # print("🚀 Starting multi-label model training...")

    X, y_labels = create_mixtures_and_labels(ALL_COMPOUNDS, DRUGS, COMMON_AXIS)
    
    # Use MultiLabelBinarizer to convert labels to binary format
    mlb = MultiLabelBinarizer(classes=DRUGS)
    y = mlb.fit_transform(y_labels)

    # print(f"📊 Dataset built → X: {X.shape}, y: {y.shape}")

    # Shuffle and split data
    X, y = shuffle(X, y, random_state=42)
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42) # Using 80% for training

    models = {
        "xgb": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss", verbosity=0
        ),
        "extratrees": ExtraTreesClassifier(
            n_estimators=200, max_features="sqrt",
            min_samples_split=5, random_state=42
        ),
        "ridge": RidgeClassifier(alpha=1.0, solver="sag"),
        "catboost": CatBoostClassifier(
            iterations=300, learning_rate=0.1, depth=6,
            l2_leaf_reg=3, thread_count=-1, verbose=0
        ),
        "svc": SVC(C=1.0, kernel="rbf", probability=True, gamma="scale"),
        "adaboost": AdaBoostClassifier(
            n_estimators=150, learning_rate=0.8, random_state=42
        )
    }

    trained_chains = []
    for name, base in models.items():
        # print(f"🔧 Training chain: {name}")
        classifier = ClassifierChain(base, order='random', random_state=42)
        classifier.fit(X_tr, y_tr)
        trained_chains.append(classifier)

    joblib.dump(trained_chains, "model/ensemble_classifier_chains.pkl")
    joblib.dump(mlb, "model/multidrug_label_binarizer.pkl")
    # print("✅ Multi-label Models saved to model/*.pkl")

# No if __name__ == "__main__": block here.
# Call train_multi_models() explicitly when you want to train.
