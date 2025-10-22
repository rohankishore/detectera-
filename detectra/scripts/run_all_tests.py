"""
Run automated tests for the detectra app (pure, mixture, multiple compounds).
Saves a JSON report to reports/test_run_report.json and prints a short summary.

Usage (from detectra/):
    python scripts/run_all_tests.py

This script is resilient to missing model files: it will attempt to load models and record missing artifacts.
"""
import os
import json
import tempfile
import traceback
import numpy as np
import pandas as pd
import joblib

ROOT = os.path.dirname(os.path.dirname(__file__))  # detectra/
REPORTS_DIR = os.path.join(ROOT, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

RESULT_PATH = os.path.join(REPORTS_DIR, 'test_run_report.json')

# Helpers to load models safely
def safe_load(path):
    try:
        obj = joblib.load(path)
        return obj, None
    except Exception as e:
        return None, str(e)

# Import internal modules (use absolute paths)
import sys
sys.path.insert(0, ROOT)

from model_1 import predict_sample, DRUG_INFO
import run_1
import predict as multi_predict

report = {
    'environment': {},
    'pure_analysis': [],
    'mixture_analysis': [],
    'multiple_compounds': [],
    'errors': []
}

# Check that training data exists
training_dir = os.path.join(ROOT, 'Training Data')
report['environment']['training_dir'] = training_dir
report['environment']['training_files'] = []
for f in os.listdir(training_dir):
    if f.endswith('.csv'):
        report['environment']['training_files'].append(f)

# Try to load models for pure analysis
binary_model_path = os.path.join(ROOT, 'drug_binary_xgb.pkl')
multiclass_model_path = os.path.join(ROOT, 'drug_multiclass_xgb.pkl')
le_path = os.path.join(ROOT, 'drug_label_encoder.pkl')

binary_model, err = safe_load(binary_model_path)
if err:
    report['environment']['binary_model'] = f'missing or error: {err}'
else:
    report['environment']['binary_model'] = 'loaded'

multiclass_model, err = safe_load(multiclass_model_path)
if err:
    report['environment']['multiclass_model'] = f'missing or error: {err}'
else:
    report['environment']['multiclass_model'] = 'loaded'

le, err = safe_load(le_path)
if err:
    report['environment']['label_encoder'] = f'missing or error: {err}'
else:
    report['environment']['label_encoder'] = 'loaded'

# Pure Analysis: run predict_sample on each drug file if models are loaded
drug_files = {d: os.path.join(training_dir, f"{d}.csv") for d in DRUG_INFO.keys()}

for drug, path in drug_files.items():
    entry = {'drug': drug, 'file': path}
    try:
        if not os.path.exists(path):
            entry['status'] = 'missing_file'
        elif binary_model is None or multiclass_model is None or le is None:
            entry['status'] = 'skipped_missing_models'
        else:
            res = predict_sample(path, binary_model, multiclass_model, le)
            entry['status'] = 'ok'
            entry['result'] = res
    except Exception as e:
        entry['status'] = 'error'
        entry['error'] = str(e)
        entry['traceback'] = traceback.format_exc()
    report['pure_analysis'].append(entry)

# Mixture Analysis: create several mixtures and run peak matching + optionally prediction
mixture_tests = [
    # (drug, drug_pct, cutting_agent, cutting_pct)
    ('cocaine', 70, 'sucrose', 30),
    ('heroin', 50, 'glucose', 50),
    ('meth', 60, 'ethanol', 40),
]

for drug, dp, cut, cp in mixture_tests:
    entry = {'drug': drug, 'drug_pct': dp, 'cutting_agent': cut, 'cutting_pct': cp}
    try:
        # Generate mixture spectrum
        spectrum = run_1.generate_mixture(drug, dp, cut, cp)
        wavenumbers = run_1.wavenumbers
        # Save temporary CSV
        tmp_csv = None
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as tf:
            tmp_csv = tf.name
            df = pd.DataFrame({'wavenumber': wavenumbers, 'absorbance': spectrum})
            df.to_csv(tf, index=False)
        entry['tmp_csv'] = tmp_csv
        # Find characteristic peaks
        found = run_1.find_characteristic_peaks(spectrum, drug)
        entry['found_peaks'] = found
        # If models available, run predict_sample
        if binary_model is not None and multiclass_model is not None and le is not None:
            res = predict_sample(tmp_csv, binary_model, multiclass_model, le)
            entry['prediction'] = res
            entry['status'] = 'ok'
        else:
            entry['status'] = 'ok_no_models'
    except Exception as e:
        entry['status'] = 'error'
        entry['error'] = str(e)
        entry['traceback'] = traceback.format_exc()
    report['mixture_analysis'].append(entry)

# Multiple Compounds: combine spectra and use ensemble chains voting (if present)
chains_path = os.path.join(ROOT, 'model', 'ensemble_classifier_chains.pkl')
mlb_path = os.path.join(ROOT, 'model', 'multidrug_label_binarizer.pkl')

chains, err = safe_load(chains_path)
if err:
    report['environment']['chains'] = f'missing or error: {err}'
else:
    report['environment']['chains'] = 'loaded'

mlb, err = safe_load(mlb_path)
if err:
    report['environment']['mlb'] = f'missing or error: {err}'
else:
    report['environment']['mlb'] = 'loaded'

# define some combos to test
combos = [
    ['cocaine', 'sucrose'],
    ['cocaine', 'heroin', 'sucrose'],
    ['heroin', 'glucose'],
    ['meth', 'citric', 'sucrose'],
]

# helper vote_predict implementation (from test_comp.py logic)
def vote_predict(chains, X_input):
    if X_input.ndim == 1:
        X_input = X_input.reshape(1, -1)
    predictions = []
    for i, chain in enumerate(chains):
        try:
            pred = chain.predict(X_input)
            predictions.append(pred[0])
        except Exception as e:
            # skip failing chain
            continue
    if not predictions:
        raise ValueError('No valid predictions from any model.')
    pred_array = np.array(predictions)
    vote_sum = np.sum(pred_array, axis=0)
    majority = (vote_sum >= (len(predictions) // 2 + 1)).astype(int)
    return majority.reshape(1, -1)

for combo in combos:
    entry = {'combo': combo}
    try:
        # Build mixture by loading and summing spectra
        spectra = []
        for comp in combo:
            path = os.path.join(ROOT, 'Training Data', f"{comp}.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(f'Missing training file for {comp}: {path}')
            arr = multi_predict.load_and_interpolate_spectrum(path)
            spectra.append(arr)
        mixture = np.sum(spectra, axis=0) / len(spectra)
        entry['mixture_mean_abs'] = float(np.mean(mixture))
        # If chains available, run vote predict
        if chains is not None and mlb is not None:
            try:
                pred_bin = vote_predict(chains, mixture)
                predicted = mlb.inverse_transform(pred_bin)
                entry['predicted'] = predicted[0] if len(predicted)>0 else []
                entry['status'] = 'ok'
            except Exception as e:
                entry['status'] = 'error_predict'
                entry['error'] = str(e)
        else:
            entry['status'] = 'skipped_missing_chains'
    except Exception as e:
        entry['status'] = 'error'
        entry['error'] = str(e)
        entry['traceback'] = traceback.format_exc()
    report['multiple_compounds'].append(entry)

# Save report
with open(RESULT_PATH, 'w') as f:
    json.dump(report, f, indent=2, default=str)

# Print concise summary
print('Test run completed. Summary:')
print('Pure analysis tests:', len(report['pure_analysis']))
print('Mixture analysis tests:', len(report['mixture_analysis']))
print('Multiple compound tests:', len(report['multiple_compounds']))
print('\nDetailed JSON written to:', RESULT_PATH)
