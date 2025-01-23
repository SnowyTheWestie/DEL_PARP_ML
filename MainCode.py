import os
import pandas as pd

from src.load_and_preprocess_data import (
    load_data,
    assign_labels,
    get_chembl_actives,
    sanitize_and_generate_mols,
    calculate_morgan_fingerprints,
)
from src.ML import (
    compare_models,
    model_comparison_to_df,
    explore_sampling,
)

from src.plotting_data import plot_model_performance

# Define folder paths for reproducibility
folder_path = os.path.join(os.getcwd(), 'data')  # Dynamically find the 'data' folder
# Define the output folder relative to your project directory
output_folder = os.path.join(os.getcwd(), "output")
# Create the folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Step 1: Load learning data
learning_data_filename = 'learning_data.csv'  # Replace with sample file in your GitHub
file_path = os.path.join(folder_path, learning_data_filename)
print(f"Loading file from: {file_path}")  # Debugging line
learning_data_df = load_data(file_path)

# Assign labels based on NSC threshold
threshold = 10
learning_data_df = assign_labels(learning_data_df, {'NSC': threshold})

# Step 2: Fetch ChEMBL data for prediction
chembl_id = 'CHEMBL5366'  # Example ChEMBL ID for validation
chembl_data_df = get_chembl_actives(chembl_id, ta=500)
chembl_data_df['Label'] = 1  # Mark as active compounds

# Step 3: Load additional inactive dataset
inactive_data_filename = 'random_inactives.csv'  # Replace with sample file
file_path = os.path.join(folder_path, inactive_data_filename)
print(f"Loading file from: {file_path}")  # Debugging line
inactive_data_df = load_data(file_path, sample_size=1000)
inactive_data_df['Label'] = 0  # Mark as inactive compounds

# Step 4: Combine datasets for validation
combined_validation_data_df = pd.concat([chembl_data_df, inactive_data_df], ignore_index=True)

# Step 5: Load experimental data
experimental_data_filename = 'PARP2_experimental_data.csv'  # Replace with sample file
file_path = os.path.join(folder_path, experimental_data_filename)
print(f"Loading file from: {file_path}")  # Debugging line
experimental_data_df = load_data(file_path)

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
print(f"Sanitized SMILES and generation of Mol objects successful!")

# Step 7: Calculate Morgan Fingerprints
MFP_radius = 7
MFP_bits = 2048

learning_data_df['MFP'] = calculate_morgan_fingerprints(learning_data_df['mol'], radius=MFP_radius, n_bits=MFP_bits)
learning_data_df.rename(columns={'NSC label': 'Label'}, inplace=True)

combined_validation_data_df['MFP'] = calculate_morgan_fingerprints(combined_validation_data_df['mol'], radius=MFP_radius, n_bits=MFP_bits)
experimental_data_df['MFP'] = calculate_morgan_fingerprints(experimental_data_df['mol'], radius=MFP_radius, n_bits=MFP_bits)
print(f"Calculation of Morgan fingerprints successful!")

# Step 8: Explore sampling and plot performance
sampling_methods = ['combined', 'smoten', 'oversample', 'undersample']
ratios = [0, 100, 50, 20, 10, 5, 2, 1]

for sampling_method in sampling_methods:
    print(f"Running sampling method: {sampling_method}")
    for use_optimized_threshold in [False, True]:
        print(f"Using optimized threshold: {use_optimized_threshold}")
        results_df = explore_sampling(
            learning_data_df=learning_data_df,
            external_validation_data_df_1=combined_validation_data_df,
            external_validation_data_df_2=experimental_data_df,
            ratios=ratios,
            n_runs=5,
            sampling_method=sampling_method,
            use_optimized_threshold=use_optimized_threshold,
        )

        # Save and plot results
        filename_prefix = f"{sampling_method}_{'optimized' if use_optimized_threshold else 'default'}"
        output_plot_path = os.path.join(output_folder, f"{filename_prefix}_performance_plot.png")
        output_data_path = os.path.join(output_folder, f"{filename_prefix}_performance_data.csv")

        plot_model_performance(results_df, folder_path=output_folder, save_plots_and_data=True)
        print(f"Saved plot to {output_plot_path} and data to {output_data_path}")

# Step 9: Run model comparison
print("Running model comparison...")
ratios = [1]  # Using the combined method with a ratio of 1
n_runs = 5    # Number of iterations for averaging results

model_comparison_results = compare_models(
    learning_data_df=learning_data_df,
    external_validation_data_df_1=experimental_data_df,
    external_validation_data_df_2=combined_validation_data_df,
    ratios=ratios,
    n_runs=n_runs,
)

# Step 10: Convert results to DataFrame and save
print("Saving results...")
comparison_df = model_comparison_to_df(model_comparison_results)

comparison_file = os.path.join(output_folder, 'model_comparison_results.csv')
comparison_df.to_csv(comparison_file, index=False)

# Display first few rows of results
print(comparison_df.head())

print(f"Model comparison results saved to {comparison_file}")