import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from scipy.signal import savgol_filter, find_peaks
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from scipy.stats import uniform, randint
import traceback
import os

BASE_DIR = os.path.dirname(__file__)

# Enhanced drug information database with verified peak ranges
DRUG_INFO = {
    'cocaine': {
        'class': 'Stimulant',
        'key_peaks': [1715, 1695, 1600, 1275, 1105, 1005, 705], # Current peaks
        'min_peaks_required': 3,
        'effects': 'Euphoria, increased energy, mental alertness',
        'risks': 'Heart attack, stroke, seizures, addiction'
    },
    'heroin': {
        'class': 'Opioid',
        'key_peaks': [1745, 1715, 1650, 1245, 1170, 1030, 750], # Current peaks
        'min_peaks_required': 3,
        'effects': 'Euphoria, pain relief, drowsiness',
        'risks': 'Respiratory depression, overdose, addiction'
    },
    'methadone': {
        'class': 'Synthetic opioid',
        'key_peaks': [1715, 1600, 1450, 1240, 1100, 750, 700], # Current peaks
        'min_peaks_required': 3,
        'effects': 'Pain relief, prevention of opioid withdrawal symptoms',
        'risks': 'Respiratory depression, heart problems, addiction'
    },
    'morphine': {
        'class': 'Opioid',
        'key_peaks': [3400, 3000, 1620, 1505, 1250, 1100, 810], # Current peaks
        'min_peaks_required': 3,
        'effects': 'Pain relief, euphoria, sedation',
        'risks': 'Respiratory depression, addiction, constipation'
    },
    'meth': {
        'class': 'Stimulant',
        'key_peaks': [3300, 3000, 1600, 1450, 1250, 1050, 750], # Current peaks
        'min_peaks_required': 3,
        'effects': 'Increased energy, alertness, euphoria',
        'risks': 'Addiction, paranoia, cardiovascular problems'
    }
}

def analyze_spectrum(df, drug_type):
    """Enhanced spectrum analysis without visualization"""
    # print(f"\n🔍 ANALYZING {drug_type.upper()} SPECTRUM:")
    # print(f"  - Data points: {len(df)}")
    # print(f"  - wavenumber range: {df['wavenumber'].min():.1f}-{df['wavenumber'].max():.1f} nm")
    # print(f"  - Absorbance range: {df['absorbance'].min():.4f}-{df['absorbance'].max():.4f}")
    
    # Find peaks - CORRECTED: Using a fraction of max absorbance for height, and smaller prominence
    peaks, _ = find_peaks(df['absorbance'], height=df['absorbance'].max() * 0.1, 
                         prominence=0.01, distance=10) # Adjusted prominence to 0.01
    peak_wavenumbers = df['wavenumber'].iloc[peaks].values
    peak_absorptions = df['absorbance'].iloc[peaks].values
    
    top_peaks = sorted(zip(peak_wavenumbers, peak_absorptions), key=lambda x: x[1], reverse=True)[:5]
    # print("  - Top 5 peaks:", [f"{w:.1f}nm ({a:.3f})" for w, a in top_peaks])
    return top_peaks

def augment_spectrum(df, n_augments=5):
    """Create realistic spectral variations"""
    augmented_dfs = []
    for _ in range(n_augments):
        new_df = df.copy()
        
        # Add different types of noise
        noise = np.random.normal(0, 0.005, len(df))  # Gaussian noise
        noise += np.random.uniform(-0.002, 0.002, len(df))  # Baseline drift
        
        # Random wavenumber shift (small)
        shift = np.random.uniform(-1, 1)
        
        # Random intensity variation
        intensity = np.random.uniform(0.9, 1.1)
        
        new_df['wavenumber'] = new_df['wavenumber'] + shift
        new_df['absorbance'] = new_df['absorbance'] * intensity + noise
        
        augmented_dfs.append(new_df)
    return augmented_dfs

def extract_features(df):
    """Completely revised feature extraction focused on drug signatures"""
    features = {}
    
    # 1. Basic statistics
    features['abs_mean'] = df['absorbance'].mean()
    features['abs_std'] = df['absorbance'].std()
    features['abs_max'] = df['absorbance'].max()
    features['abs_min'] = df['absorbance'].min()
    
    # 2. Peak characteristics
    # CORRECTED: Using a fraction of max absorbance for height, and smaller prominence
    peaks, properties = find_peaks(df['absorbance'], 
                                 height=df['absorbance'].max() * 0.1, 
                                 prominence=0.01, # Adjusted prominence to 0.01
                                 width=5)
    
    features['num_peaks'] = len(peaks)
    if len(peaks) > 0:
        features['mean_peak_height'] = np.mean(properties['peak_heights'])
        features['max_peak_height'] = np.max(properties['peak_heights'])
        features['mean_peak_width'] = np.mean(properties['widths'])
    else:
        features['mean_peak_height'] = 0
        features['max_peak_height'] = 0
        features['mean_peak_width'] = 0
    
    # 3. Drug-specific peak matches
    for drug, info in DRUG_INFO.items():
        matched_peaks = 0
        total_intensity = 0
        
        for target_peak in info['key_peaks']:
            # Find closest peak in actual data
            idx = (df['wavenumber'] - target_peak).abs().idxmin()
            actual_peak = df['wavenumber'].iloc[idx]
            
            # Check if there's a significant peak nearby
            # This condition also needs to be robust for low absorbance. 
            # Check against a small absolute value or a fraction of max absorbance.
            # Keeping it consistent with peak finding: use a fraction of max absorbance
            # Changed tolerance from 10 to 20 for matching
            if (df['absorbance'].iloc[idx] > df['absorbance'].max() * 0.1) and \
               (abs(actual_peak - target_peak) < 30):
                matched_peaks += 1
                total_intensity += df['absorbance'].iloc[idx]
        
        features[f'{drug}_peaks_matched'] = matched_peaks
        features[f'{drug}_peak_intensity'] = total_intensity
        features[f'{drug}_is_present'] = int(matched_peaks >= info['min_peaks_required'])
    
    # 4. Spectral derivatives
    window_length = min(21, len(df)//2)
    if window_length % 2 == 0:
        window_length -= 1
    
    try:
        smoothed = savgol_filter(df['absorbance'], window_length=window_length, polyorder=2)
        first_deriv = savgol_filter(df['absorbance'], window_length=window_length, polyorder=2, deriv=1)
        
        features['deriv_mean'] = np.mean(first_deriv)
        features['deriv_std'] = np.std(first_deriv)
        features['deriv_max'] = np.max(first_deriv)
    except:
        features['deriv_mean'] = 0
        features['deriv_std'] = 0
        features['deriv_max'] = 0

    features['total_drug_peaks'] = sum(features.get(f'{d}_peaks_matched', 0) for d in DRUG_INFO)
    features['total_drug_intensity'] = sum(features.get(f'{d}_peak_intensity', 0) for d in DRUG_INFO)
    
    # Make sure to include drug_type and is_drug placeholders
    features['drug_type'] = 'unknown'  # Placeholder
    features['is_drug'] = 0  # Placeholder
    
    return features

def prepare_training_data(drug_files, non_drug_files):
    """Create enhanced training dataset"""
    samples = []
    
    # Process drug samples
    for drug, path in drug_files.items():
        try:
            df = pd.read_csv(path)
            # print(f"\nProcessing {drug}...")
            # analyze_spectrum(df, drug) # Commented out to reduce console output during training
            
            # Extract features for original df
            original_features = extract_features(df)
            original_features['drug_type'] = drug
            original_features['is_drug'] = 1
            samples.append(original_features)

            # Augment and extract features for augmented dfs
            augmented_dfs = augment_spectrum(df, n_augments=5)
            for aug_df in augmented_dfs:
                features = extract_features(aug_df)
                features['drug_type'] = drug
                features['is_drug'] = 1
                samples.append(features)
        except Exception as e:
            print(f"Error processing {drug}: {e}")
    
    # Process non-drug samples
    for i, path in enumerate(non_drug_files):
        try:
            df = pd.read_csv(path)
            # print(f"\nProcessing non-drug sample {i+1}...")
            # analyze_spectrum(df, f"non_drug_{i+1}") # Commented out
            
            # Extract features for original df
            original_features = extract_features(df)
            original_features['drug_type'] = f'non_drug_{i+1}'
            original_features['is_drug'] = 0
            samples.append(original_features)

            # More non-drug augmentations to balance dataset
            augmented_dfs = augment_spectrum(df, n_augments=10)
            for aug_df in augmented_dfs:
                features = extract_features(aug_df)
                features['drug_type'] = f'non_drug_{i+1}'
                features['is_drug'] = 0
                samples.append(features)
        except Exception as e:
            print(f"Error processing non-drug sample {i+1}: {e}")
    
    # Create DataFrame
    features_df = pd.DataFrame(samples)
    
    # Add additional engineered features (already present from extract_features but for clarity)
    features_df['total_drug_peaks'] = features_df[[f'{d}_peaks_matched' for d in DRUG_INFO]].sum(axis=1)
    features_df['total_drug_intensity'] = features_df[[f'{d}_peak_intensity' for d in DRUG_INFO]].sum(axis=1)
    
    # print("\n📊 Final dataset summary:")
    # print(f"  - Total samples: {len(features_df)}")
    # print(f"  - Drug samples: {features_df['is_drug'].sum()}")
    # print(f"  - Non-drug samples: {len(features_df) - features_df['is_drug'].sum()}")
    
    return features_df

def train_models(features_df):
    """Train XGBoost models with proper tuning"""
    # Binary classification (drug vs non-drug)
    X = features_df.drop(columns=['drug_type', 'is_drug'])
    y = features_df['is_drug']
    
    # Multiclass classification (only for drug samples)
    drug_samples = features_df[features_df['is_drug'] == 1]
    X_drug = drug_samples.drop(columns=['drug_type', 'is_drug'])
    y_drug = drug_samples['drug_type']
    
    le = LabelEncoder()
    y_drug_encoded = le.fit_transform(y_drug)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_drug, X_test_drug, y_train_drug, y_test_drug = train_test_split(
        X_drug, y_drug_encoded, test_size=0.2, random_state=42, stratify=y_drug_encoded
    )
    
    # print("\n🔨 Training Binary Classifier...")
    
    # Binary classifier pipeline
    binary_pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(k_neighbors=3, random_state=42)),
        ('xgb', XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,
            random_state=42
        ))
    ])
    
    binary_pipeline.fit(X_train, y_train)
    
    # Multiclass classifier
    # print("\n🔨 Training Multiclass Classifier...")
    
    multiclass_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            num_class=len(DRUG_INFO),
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42
        ))
    ])
    
    multiclass_pipeline.fit(X_train_drug, y_train_drug)
    
    # Evaluation (commented out for cleaner console during web app use)
    # print("\n" + "="*60)
    # print("MODEL EVALUATION")
    # print("="*60)
    
    # print("\n🔍 Binary Classification (Drug vs Non-Drug):")
    # y_pred = binary_pipeline.predict(X_test)
    # print(classification_report(y_test, y_pred, target_names=['Non-Drug', 'Drug']))
    # print("Confusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))
    
    # print("\n🔍 Multiclass Classification (Drug Types):")
    # y_pred_drug = multiclass_pipeline.predict(X_test_drug)
    # print(classification_report(y_test_drug, y_pred_drug, target_names=le.classes_))
    # print("Confusion Matrix:")
    # print(confusion_matrix(y_test_drug, y_pred_drug))
    
    # Save models
    joblib.dump(binary_pipeline, 'drug_binary_xgb.pkl')
    joblib.dump(multiclass_pipeline, 'drug_multiclass_xgb.pkl')
    joblib.dump(le, 'drug_label_encoder.pkl')
    
    # print("\n💾 Models saved successfully!")
    return binary_pipeline, multiclass_pipeline, le

def predict_sample(file_path, binary_model, multiclass_model, le):
    """Make predictions with detailed diagnostics"""
    try:
        # 1. Load and validate input data
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
            
        df = pd.read_csv(file_path)
        if 'wavenumber' not in df.columns or 'absorbance' not in df.columns:
            raise ValueError("Input CSV must contain 'wavenumber' and 'absorbance' columns")
            
        # print(f"\n🔬 Analyzing sample: {file_path}")
        
        # 2. Visual analysis
        top_peaks = analyze_spectrum(df, "Test Sample")
        # If analyze_spectrum returned empty, attempt a relaxed peak search to avoid empty detected_peaks
        if not top_peaks or len(top_peaks) == 0:
            try:
                # relaxed thresholds: lower height and prominence
                peaks_rel, props_rel = find_peaks(df['absorbance'], height=df['absorbance'].max() * 0.02, prominence=0.005, distance=5)
                peak_wavenumbers_rel = df['wavenumber'].iloc[peaks_rel].values
                peak_absorptions_rel = df['absorbance'].iloc[peaks_rel].values
                top_peaks = sorted(zip(peak_wavenumbers_rel, peak_absorptions_rel), key=lambda x: x[1], reverse=True)[:5]
            except Exception:
                top_peaks = []
        
        # 3. Feature extraction with validation
        raw_features = extract_features(df)
        
        # Create DataFrame with all possible features
        features_df = pd.DataFrame([raw_features])
        
        # Get feature names from the trained binary model
        # This ensures the features are in the correct order and all expected features are present
        try:
            # Access the actual trained booster to get feature names
            expected_features = binary_model.named_steps['xgb'].get_booster().feature_names
            if not expected_features: # Fallback if feature_names is None
                raise ValueError("Could not retrieve feature names from the model. Using fallback.")
        except Exception as e:
            # print(f"⚠️ Warning: Could not get feature names from binary model ({e}). Using predefined list.")
            expected_features = [
                'abs_mean', 'abs_std', 'abs_max', 'abs_min',
                'num_peaks', 'mean_peak_height', 'max_peak_height', 'mean_peak_width',
                'cocaine_peaks_matched', 'cocaine_peak_intensity', 'cocaine_is_present',
                'heroin_peaks_matched', 'heroin_peak_intensity', 'heroin_is_present',
                'methadone_peaks_matched', 'methadone_peak_intensity', 'methadone_is_present',
                'morphine_peaks_matched', 'morphine_peak_intensity', 'morphine_is_present',
                'meth_peaks_matched','meth_peak_intensity','meth_is_present',
                'deriv_mean', 'deriv_std', 'deriv_max',
                'total_drug_peaks', 'total_drug_intensity'
            ]
        
        # Ensure all expected features exist in the prediction DataFrame
        for feature in expected_features:
            if feature not in features_df.columns:
                features_df[feature] = 0.0 # Add missing features with a default value
                
        # Select and reorder columns to match training data
        features_df = features_df[expected_features]
        
        # 4. Binary classification
        drug_proba = binary_model.predict_proba(features_df)[0][1]
        is_drug = drug_proba > 0.7  # Higher threshold for confidence
        
        # print("\n" + "="*60)
        # print("PREDICTION RESULTS")
        # print("="*60)
        
        # print(f"\nBinary Classification:")
        # print(f"  - Drug Probability: {drug_proba:.1%}")
        # print(f"  - Conclusion: {'DRUG DETECTED' if is_drug else 'NO DRUG DETECTED'}")
        
        result = {
            'is_drug': bool(is_drug),
            'probability': float(drug_proba),
            'detected_peaks': [float(p[0]) for p in top_peaks],
            'peak_intensities': [float(p[1]) for p in top_peaks]
        }
        
        if is_drug:
            # 5. Multiclass prediction
            drug_probs = multiclass_model.predict_proba(features_df)[0]
            pred_idx = np.argmax(drug_probs)
            pred_drug = le.inverse_transform([pred_idx])[0]
            confidence = drug_probs[pred_idx]
            
            # print(f"\nDrug Type Identification:")
            # print(f"  - Predicted Drug: {pred_drug} ({confidence:.1%} confidence)")
            # print("\nAlternative Possibilities:")
            # for i, prob in enumerate(drug_probs):
            #     print(f"  - {le.classes_[i]}: {prob:.1%}")
            
            # Get drug info
            drug_info = DRUG_INFO.get(pred_drug, {})
            # print("\nDrug Information:")
            # for k, v in drug_info.items():
            #     if k not in ['key_peaks', 'min_peaks_required']:
            #         print(f"  - {k}: {v}")
            
            # Check if expected peaks were detected
            # print("\nPeak Matching Analysis:")
            expected_peaks = DRUG_INFO.get(pred_drug, {}).get('key_peaks', [])
            detected_peaks = [p[0] for p in top_peaks]
            
            # for peak in expected_peaks:
            #     closest = min(detected_peaks, key=lambda x: abs(x - peak)) if detected_peaks else None
            #     # Changed tolerance from 10 to 20 for matching
            #     status = '✅' if closest and abs(closest - peak) < 30 else '❌'
            #     print(f"  - Expected peak at {peak}nm: {status} (closest: {closest:.1f}nm)" if closest else f"  - Expected peak at {peak}nm: ❌ (no peaks detected)")
            
            # Update result with drug-specific info
            result.update({
                'drug_type': pred_drug,
                'confidence': float(confidence),
                'matched_peaks': int(raw_features.get(f'{pred_drug}_peaks_matched', 0)),
                'expected_peaks': expected_peaks,
                'drug_info': {k:v for k,v in drug_info.items() if k not in ['key_peaks', 'min_peaks_required']}
            })
        else:
            result['reason'] = "Low probability score (<70%)"
            
        return result
            
    except Exception as e:
        # print(f"\n❌ Prediction error: {str(e)}")
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

# Function to be called by app.py for training
def train_and_save_models():
    # Define input files
    TRAINING_DIR = os.path.join(BASE_DIR, 'Training Data')
    drug_files = {
        'cocaine': os.path.join(TRAINING_DIR, "cocaine.csv"),
        'heroin': os.path.join(TRAINING_DIR, "heroin.csv"),
        'methadone': os.path.join(TRAINING_DIR, "methadone.csv"),
        'morphine': os.path.join(TRAINING_DIR, "morphine.csv"),
        'meth' : os.path.join(TRAINING_DIR, "meth.csv")
    }
    non_drug_files = [
        os.path.join(TRAINING_DIR, "lactic.csv"), 
        os.path.join(TRAINING_DIR, "citric.csv"), 
        os.path.join(TRAINING_DIR, "ethanol.csv"), 
        os.path.join(TRAINING_DIR, "glucose.csv"),
        os.path.join(TRAINING_DIR, "sucrose.csv")
    ]
    
    # print("🚀 Starting Drug Detection System (Model Training)")
    
    try:
        # Prepare data
        features_df = prepare_training_data(drug_files, non_drug_files)
        
        # Train models
        binary_model, multiclass_model, le = train_models(features_df)
        # print("✅ Pure Compound Models trained and saved.")
            
    except Exception as e:
        print(f"\n❌ Fatal error during pure compound model training: {str(e)}")
        raise # Re-raise the exception to be caught by Flask app
