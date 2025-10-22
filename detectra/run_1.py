import numpy as np
from scipy.signal import savgol_filter, find_peaks

# Characteristic peaks (in cm⁻¹)
DRUGS = {
    "heroin": [1745, 1245, 1035, 950, 820, 4770, 5254],
    "morphine": [3400, 1320, 1250, 930],
    "cocaine": [1705, 1275, 1100, 860, 750, 1450],
    "meth": [2960, 2925, 1490, 1380, 1340, 1040],
    "methadone": [1715, 1285, 1125, 980, 890, 1450]
}

NON_DRUGS = {
    "sucrose": [3500, 2900, 1640, 1410, 1100, 1060],
    "citric_acid": [3500, 2950, 1700, 1400, 1250, 1100], # Renamed to match filename convention
    "ethanol": [3400, 2900, 1650, 1450, 1350, 1100],
    "lactic": [3500, 2950, 1720, 1450, 1400, 1250],
    "glucose": [3300, 2900, 1400, 1030, 900], # Added glucose as it's in training data
    "none": [] # For cases where no cutting agent is desired
}

# Spectrum configuration
WN_MIN = 800
WN_MAX = 5500
N_POINTS = 1000
wavenumbers = np.linspace(WN_MIN, WN_MAX, N_POINTS)

# Generate compound spectrum from peak positions
def generate_compound_spectrum(peaks, weight=1.0):
    spectrum = np.zeros_like(wavenumbers)
    # Ensure peaks are not empty to avoid errors
    if not peaks:
        return spectrum

    for peak in peaks:
        if peak: # Check if peak value is valid
            # Adjust the width of the Gaussian based on the peak value for more realistic spectra
            # Larger wavenumber peaks might have slightly broader bands
            peak_width = 0.03 * peak # Example: width scales with peak position
            spectrum += weight * np.exp(-0.5 * ((wavenumbers - peak) / peak_width)**2)
    return spectrum

# Generate synthetic spectrum of drug + cutting agent + noise
def generate_mixture(drug, drug_weight, non_drug, non_drug_weight):
    # Ensure non_drug exists in NON_DRUGS, default to 'none' if not found
    if non_drug not in NON_DRUGS:
        print(f"Warning: Non-drug '{non_drug}' not found in NON_DRUGS. Using 'none'.")
        non_drug = 'none'

    drug_spec = generate_compound_spectrum(DRUGS[drug], drug_weight / 100)
    non_drug_spec = generate_compound_spectrum(NON_DRUGS[non_drug], non_drug_weight / 100)
    noise = 0.01 * np.random.normal(size=N_POINTS)
    
    # Apply Savitzky-Golay filter for smoothing
    window_length = 11
    polyorder = 3
    # Ensure window_length is odd and less than or equal to the number of points
    if window_length > N_POINTS:
        window_length = N_POINTS - (N_POINTS % 2 == 0) # Make it the largest odd number <= N_POINTS
    if window_length % 2 == 0: # Ensure it's odd
        window_length -= 1
    if window_length < 3: # Minimum window length is 3
        window_length = 3

    return savgol_filter(drug_spec + non_drug_spec + noise, window_length=window_length, polyorder=polyorder)

# Match detected peaks with known characteristic peaks
def find_characteristic_peaks(spectrum, drug):
    # Find peaks with a minimum height and prominence to filter noise
    peaks, properties = find_peaks(spectrum, height=0.05 * np.max(spectrum), prominence=0.01)
    found_peaks = []

    # Ensure drug exists in DRUGS
    if drug not in DRUGS:
        print(f"Warning: Drug '{drug}' not found in DRUGS for peak matching.")
        return []

    for target in DRUGS[drug]:
        if len(peaks) > 0:
            # Find the closest detected peak to the target characteristic peak
            closest_idx = np.argmin(np.abs(wavenumbers[peaks] - target))
            closest_peak_wavenumber = wavenumbers[peaks][closest_idx]
            closest_peak_intensity = spectrum[peaks][closest_idx]

            # Define a tolerance for matching (e.g., within 15 cm⁻¹)
            if abs(closest_peak_wavenumber - target) < 15:
                found_peaks.append((target, closest_peak_wavenumber, closest_peak_intensity))

    return found_peaks

# User input interface (commented out as it's not used by Flask app)
# def get_user_input():
#     print("\nAvailable Drugs:")
#     for i, drug in enumerate(DRUGS.keys(), 1):
#         print(f"{i}. {drug}")

#     print("\nAvailable Cutting Agents:")
#     for i, agent in enumerate(NON_DRUGS.keys(), 1):
#         print(f"{i}. {agent}")

#     try:
#         drug_idx = int(input("\nEnter drug number (1-5): ")) - 1
#         drug = list(DRUGS.keys())[drug_idx]
#         drug_weight = float(input(f"Enter {drug} percentage (1-100): "))

#         agent_idx = int(input("\nEnter cutting agent number (1-5): ")) - 1
#         agent = list(NON_DRUGS.keys())[agent_idx]
#         max_agent_weight = 100 - drug_weight
#         agent_weight = float(input(f"Enter {agent} percentage (0-{max_agent_weight}): "))

#         return drug, drug_weight, agent, agent_weight
#     except (ValueError, IndexError):
#         print("❌ Invalid input. Please try again.")
#         exit()

# Main execution function (commented out as it's not used by Flask app)
# def main():
#     print("🔬 Drug Detector with Peak Verification\n")

#     drug, drug_weight, agent, agent_weight = get_user_input()
#     spectrum = generate_mixture(drug, drug_weight, agent, agent_weight)

#     found_peaks = find_characteristic_peaks(spectrum, drug)

#     print(f"\nCharacteristic Peaks Found for {drug}:")
#     if not found_peaks:
#         print("❌ No characteristic peaks detected!")
#     else:
#         print("Target(cm⁻¹)\tDetected(cm⁻¹)\tIntensity")
#         print("-" * 40)
#         for target, detected, intensity in found_peaks:
#             print(f"{target:.0f}\t\t{detected:.0f}\t\t{intensity:.3f}")

#         match_percentage = len(found_peaks) / len(DRUGS[drug]) * 100
#         print(f"\n✅ {match_percentage:.1f}% of characteristic peaks matched")

#     print(f"\n🧠 Prediction (based on peaks only): {drug}")

# if __name__ == "__main__":
#     main()
