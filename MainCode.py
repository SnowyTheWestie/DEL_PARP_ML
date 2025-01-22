import os
import pandas as pd

from src.load_and_preprocess_data import (
    load_data,
    assign_labels,
    get_chembl_actives,
    sanitize_and_generate_mols,
    calculate_morgan_fingerprints,
)

# Define folder paths for reproducibility
CURRENT_DIR = os.path.dirname(__file__)  # Current script location
DATA_DIR = os.path.join(CURRENT_DIR, '../data')  # Relative to the project directory

# Step 1: Load learning data
learning_data_filename = 'learning_data.csv'  # Replace with sample file in your GitHub
learning_data_df = load_data(learning_data_filename, DATA_DIR)

# Assign labels based on NSC threshold
threshold = 10
learning_data_df = assign_labels(learning_data_df, {'NSC': threshold})

# Step 2: Fetch ChEMBL data for prediction
chembl_id = 'CHEMBL5366'  # Example ChEMBL ID for validation
chembl_data_df = get_chembl_actives(chembl_id, ta=500)
chembl_data_df['Label'] = 1  # Mark as active compounds

# Step 3: Load additional inactive dataset
inactive_data_filename = 'inactive_data_sample.csv'  # Replace with sample file
inactive_data_df = load_data(inactive_data_filename, DATA_DIR, sample_size=1000)
inactive_data_df['Label'] = 0  # Mark as inactive compounds

# Step 4: Combine datasets for validation
combined_validation_data_df = pd.concat([chembl_data_df, inactive_data_df], ignore_index=True)

# Step 5: Load experimental data
experimental_data_filename = 'experimental_data.csv'  # Replace with sample file
experimental_data_df = load_data(experimental_data_filename, DATA_DIR)

# Print to confirm successful loading
print(f"Learning Data: {learning_data_df.shape}")
print(f"ChEMBL Data: {chembl_data_df.shape}")
print(f"Inactive Data: {inactive_data_df.shape}")
print(f"Combined Validation Data: {combined_validation_data_df.shape}")
print(f"Experimental Data: {experimental_data_df.shape}")

# Step 6: Sanitize SMILES and generate Mol objects
learning_data_df['SMILES'], learning_data_df['mol'] = sanitize_and_generate_mols(learning_data_df['SMILES'])
combined_validation_data_df['SMILES'], combined_validation_data_df['mol'] = sanitize_and_generate_mols(combined_validation_data_df['SMILES'])
experimental_data_df['SMILES'], experimental_data_df['mol'] = sanitize_and_generate_mols(experimental_data_df['SMILES'])

# Step 7: Calculate Morgan Fingerprints
MFP_radius = 7
MFP_bits = 2048

learning_data_df['MFP'] = calculate_morgan_fingerprints(learning_data_df['mol'], radius=MFP_radius, n_bits=MFP_bits)
learning_data_df.rename(columns={'NSC label': 'Label'}, inplace=True)

combined_validation_data_df['MFP'] = calculate_morgan_fingerprints(combined_validation_data_df['mol'], radius=MFP_radius, n_bits=MFP_bits)
experimental_data_df['MFP'] = calculate_morgan_fingerprints(experimental_data_df['mol'], radius=MFP_radius, n_bits=MFP_bits)
