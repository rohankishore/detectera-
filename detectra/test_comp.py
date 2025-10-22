import numpy as np
import pandas as pd
import joblib
from itertools import combinations
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import warnings
import csv
import os

os.environ['LIGHTGBM_VERBOSE'] = '0'
warnings.filterwarnings("ignore")  # Suppress sklearn and LightGBM warnings

# === Constants
COMMON_AXIS = np.linspace(4000, 400, 900)

compound_files = {
    "cocaine": "Training Data/cocaine.csv",
    "morphine": "Training Data/morphine.csv",
    "heroin": "Training Data/heroin.csv",
    "methadone": "Training Data/methadone.csv",
    "meth": "Training Data/meth.csv",
    "sucrose": "Training Data/sucrose.csv",
    "lactic": "Training Data/lactic.csv",
    "glucose": "Training Data/glucose.csv",
    "ethanol": "Training Data/ethanol.csv",
    "citric": "Training Data/citric.csv"
}
drug_compounds = ["cocaine", "morphine", "heroin", "methadone", "meth"]
non_drug_compounds = [c for c in compound_files if c not in drug_compounds]

# === Load classifier chains and label binarizer
chains = joblib.load("model/ensemble_classifier_chains.pkl")
mlb = joblib.load("model/multidrug_label_binarizer.pkl")

# === Interpolation function
def load_and_interpolate_spectrum(path):
    df = pd.read_csv(path, usecols=["wavenumber", "absorbance"])
    x = df["wavenumber"].values
    y = df["absorbance"].values
    if x[0] > x[-1]:
        x = x[::-1]
        y = y[::-1]
    return np.interp(COMMON_AXIS, x, y)

# === Voting function
def vote_predict(chains, X_input):
    if X_input.ndim == 1:
        X_input = X_input.reshape(1, -1)

    predictions = []

    for i, chain in enumerate(chains):
        try:
            check_is_fitted(chain)
            for est in chain.estimators_:
                check_is_fitted(est)

            pred = chain.predict(X_input)  # shape: (1, n_labels)
            predictions.append(pred[0])    # convert to 1D list
        except Exception as e:
            print(f"❌ Skipping chain {i}: {e}")
            continue

    if not predictions:
        raise ValueError("No valid predictions from any model.")

    # Stack predictions and apply hard voting (majority)
    pred_array = np.array(predictions)
    vote_sum = np.sum(pred_array, axis=0)
    majority = (vote_sum >= (len(predictions) // 2 + 1)).astype(int)
    return majority.reshape(1, -1)

# === Preload all spectra
compound_spectra = {
    name: load_and_interpolate_spectrum(path)
    for name, path in compound_files.items()
}

# === Evaluation
y_true_bin, y_pred_bin = [], []
output_file = "test_results.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Input_Compounds", "Expected_Drugs", "Predicted_Drugs", "Correct"])

    for num_drugs in [1, 2]:
        for drug_combo in combinations(drug_compounds, num_drugs):
            for num_nondrugs in range(1, 6):
                for nondrug_combo in combinations(non_drug_compounds, num_nondrugs):
                    combo = list(drug_combo) + list(nondrug_combo)
                    mixture = np.zeros_like(COMMON_AXIS)
                    expected = []

                    for compound in combo:
                        mixture += compound_spectra[compound]
                        if compound in drug_compounds:
                            expected.append(compound)
                    mixture /= len(combo)

                    mixture_input = mixture.reshape(1, -1).astype(np.float32)

                    try:
                        pred_bin = vote_predict(chains, mixture_input)
                    except Exception as e:
                        print(f"⚠ Skipped combo {combo} due to voting failure: {e}")
                        continue
                    predicted = mlb.inverse_transform(pred_bin)[0]

                    expected_bin = mlb.transform([expected])[0]
                    y_true_bin.append(expected_bin)
                    y_pred_bin.append(pred_bin[0])
                    correct = np.array_equal(expected_bin, pred_bin[0])

                    writer.writerow([
                        ", ".join(combo),
                        ", ".join(expected) if expected else "None",
                        ", ".join(predicted) if predicted else "None",
                        "TRUE" if correct else "FALSE"
                    ])

# === Metrics
if len(y_true_bin) == 0 or len(y_pred_bin) == 0:
    print("⚠ No predictions made. Evaluation skipped.")
else:
    y_true_bin = np.array(y_true_bin)
    y_pred_bin = np.array(y_pred_bin)

    exact_matches = np.all(y_true_bin == y_pred_bin, axis=1)
    exact_accuracy = np.mean(exact_matches)
    precision = precision_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)

    print("\n✅ All valid combinations tested (voting-based).")
    print(f"📊 Exact Match Accuracy: {exact_accuracy * 100:.2f}%")
    print(f"🎯 Micro Precision      : {precision:.3f}")
    print(f"🎯 Micro Recall         : {recall:.3f}")
    print(f"🎯 Micro F1 Score       : {f1:.3f}")
    print("📁 Detailed results saved to test_results.csv")
