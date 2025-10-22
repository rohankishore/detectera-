import os
import joblib
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.dirname(__file__))

# Candidate pickle files (relative to repo root)
CANDIDATES = [
    os.path.join(ROOT, 'drug_binary_xgb.pkl'),
    os.path.join(ROOT, 'drug_multiclass_xgb.pkl'),
]

def try_extract_and_resave(fn):
    if not os.path.exists(fn):
        print(f"Not found: {fn}")
        return
    print(f"Loading {fn}")
    obj = joblib.load(fn)

    xgb_obj = None
    if hasattr(obj, 'named_steps'):
        for name, step in obj.named_steps.items():
            if hasattr(step, 'get_booster'):
                xgb_obj = step
                break
    elif hasattr(obj, 'get_booster'):
        xgb_obj = obj

    if xgb_obj is None:
        print(f"No XGBClassifier found in {fn}; skipping")
        return

    booster_path = fn + '.booster.json'
    try:
        booster = xgb_obj.get_booster()
        booster.save_model(booster_path)
        print(f"Saved booster to {booster_path}")
    except Exception as e:
        print(f"Failed to save booster from {fn}: {e}")
        return

    # load booster into fresh XGBClassifier and repack
    new_clf = XGBClassifier()
    try:
        new_clf.load_model(booster_path)
        print(f"Loaded booster into new XGBClassifier for {fn}")
    except Exception as e:
        print(f"Failed to load booster into XGBClassifier: {e}")
        return

    out_fn = fn + '.resaved.pkl'
    if hasattr(obj, 'named_steps'):
        for name, step in obj.named_steps.items():
            if step is xgb_obj:
                obj.named_steps[name] = new_clf
                break
        joblib.dump(obj, out_fn)
    else:
        joblib.dump(new_clf, out_fn)

    print(f"Wrote resaved pickle: {out_fn}")

def main():
    for fn in CANDIDATES:
        try_extract_and_resave(fn)

if __name__ == '__main__':
    main()
