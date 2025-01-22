

import os
import numpy as np
import matplotlib.pyplot as plt



def plot_model_performance(results_df, folder_path=None, save_plots_and_data=False):
    """
    Plot the model performance metrics (Recall and F1 Score) for internal and external datasets
    as a function of the inverse ratio of active to inactive samples in the training data.

    Parameters:
    ----------
    results_df : pd.DataFrame
        DataFrame containing model performance metrics, including mean and standard deviations 
        of Recall and F1 Score for internal and external datasets at different ratios.
        
    folder_path : str, optional
        Path to the folder where the plot and data will be saved (default is None).
        
    save_plots_and_data : bool, optional
        If True, saves the plot as a .png file and the data as a .csv file to `folder_path`.
        
    Returns:
    -------
    None
        Displays the plot and optionally saves it to disk.

    Example:
    -------
    plot_model_performance(results_df, folder_path='./results', save_plots_and_data=True)
    """
    
    # Ensure that the data is sorted by 'Ratio' to maintain consistency in the plot
    plot_data = results_df.sort_values(by='Ratio')

    # Initialize the plot with a specified figure size
    fig, ax = plt.subplots(figsize=(12, 8))

    # Update plot settings for font size
    plt.rcParams.update({'font.size': 14})

    # Plot Recall metrics for internal and external datasets
    ax.errorbar(1 / plot_data['Ratio'], plot_data['Recall_Internal_Mean'],
                yerr=plot_data['Recall_Internal_Std'], fmt='-o', label='Recall (Internal)', capsize=5, color='blue')
    ax.errorbar(1 / plot_data['Ratio'], plot_data['Recall_External1_Mean'],
                yerr=plot_data['Recall_External1_Std'], fmt='-o', label='Recall (External 1)', capsize=5, color='red')

    # Include the second external dataset if available
    if 'Recall_External2_Mean' in plot_data.columns:
        ax.errorbar(1 / plot_data['Ratio'], plot_data['Recall_External2_Mean'],
                    yerr=plot_data['Recall_External2_Std'], fmt='-o', label='Recall (External 2)', capsize=5, color='green')

    # Plot F1 Score metrics for internal and external datasets
    ax.errorbar(1 / plot_data['Ratio'], plot_data['F1_Score_Internal_Mean'],
                yerr=plot_data['F1_Score_Internal_Std'], fmt='--o', label='F1 Score (Internal)', capsize=5, color='blue')
    ax.errorbar(1 / plot_data['Ratio'], plot_data['F1_Score_External1_Mean'],
                yerr=plot_data['F1_Score_External1_Std'], fmt='--o', label='F1 Score (External 1)', capsize=5, color='red')

    # Include the second external dataset if available for F1 Score as well
    if 'F1_Score_External2_Mean' in plot_data.columns:
        ax.errorbar(1 / plot_data['Ratio'], plot_data['F1_Score_External2_Mean'],
                    yerr=plot_data['F1_Score_External2_Std'], fmt='--o', label='F1 Score (External 2)', capsize=5, color='green')

    # Set the limits for the x-axis and y-axis
    ax.set_xlim(left=0, right=(1 / plot_data['Ratio']).max())
    ax.set_ylim(0, 1)  # Y-axis from 0 to 1 to standardize the scale of the scores

    # Label the axes and set the title
    ax.set_ylabel('Score')
    ax.set_xlabel('1/Ratio of Actives to Inactives in Training Data')
    ax.set_title('Model Performance as a Function of Data Bias')
    ax.legend()  # Display the legend

    plt.tight_layout()  # Adjust layout for readability

    # If saving is requested, save plot as PNG and data as CSV
    if save_plots_and_data and folder_path:
        plt.savefig(os.path.join(folder_path, 'model_performance_plot.png'))
        plot_data.to_csv(os.path.join(folder_path, 'model_performance_data.csv'), index=False)

    # Display the plot
    plt.show()



def plot_probabilities_vs_labels(df, folder_path):
    """
    Plot average predicted probabilities for each label (active or inactive) with a slight jitter 
    for better visualization. Saves the plot as a PNG file.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing 'Label' and 'Average_Probability' columns. 
        'Label' should have binary values (0 for inactive, 1 for active).
        
    folder_path : str
        Path to the folder where the plot will be saved.

    Returns:
    -------
    None
        Displays the plot and saves it as 'bagged_probabilities.png' in the specified folder.

    Example:
    -------
    plot_probabilities_vs_labels(df, folder_path='./results')
    """
    
    # Separate data based on labels
    active_df = df[df['Label'] == 1]   # Data with label = 1 (active)
    inactive_df = df[df['Label'] == 0] # Data with label = 0 (inactive)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Jitter x-values around specific points for clarity in plotting
    jitter_inactive = 0.3 + np.random.uniform(-0.02, 0.02, len(inactive_df))  # Around x = 0.3 for inactive
    jitter_active = 0.7 + np.random.uniform(-0.02, 0.02, len(active_df))      # Around x = 0.7 for active

    # Plot the probabilities, adjusting color and alpha for better visibility
    ax.scatter(jitter_inactive, inactive_df['Average_Probability'], color='blue', label='Label 0 (Inactive)', alpha=0.6)
    ax.scatter(jitter_active, active_df['Average_Probability'], color='red', label='Label 1 (Active)', alpha=0.6)

    # Label the axes and title
    ax.set_xlabel('Labels', fontsize=16)
    ax.set_ylabel('Average Probability', fontsize=16)
    ax.set_title('Probabilities by Label', fontsize=18)

    # Set x-axis ticks and labels for inactive and active
    ax.set_xticks([0.3, 0.7])
    ax.set_xticklabels(['Inactive (0)', 'Active (1)'], fontsize=14)

    # Limit the x-axis to tighten space between the two datasets and set y-axis for clarity
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0, 1)

    # Add a legend
    ax.legend(fontsize=14)

    # Save the plot to the specified folder with a high resolution
    file_path = os.path.join(folder_path, 'bagged_probabilities.png')
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    
    # Display the plot
    plt.show()


def plot_comparison(results):
    """
    Plot a comparison of model performance metrics (Recall, Precision, F1 Score) for multiple models.

    Parameters:
    ----------
    results : dict
        A dictionary where each key is a model name (e.g., 'logistic_regression', 'random_forest') and
        each value is a DataFrame containing performance metrics such as Recall, Precision, and F1 Score
        for internal data across different runs. The DataFrame should include columns for mean and std
        values of each metric (e.g., 'Recall_Internal_Mean', 'Recall_Internal_Std').

    Returns:
    -------
    None
        Displays a plot comparing the performance metrics for different models.

    Example:
    -------
    plot_comparison(results)
    """
    
    # Metrics to plot and model methods to iterate through
    metrics = ['Recall', 'Precision', 'F1_Score']
    methods = list(results.keys())  # e.g., ['logistic_regression', 'random_forest', ...]

    # Create a single figure with 3 rows, each row for a metric
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))

    # Iterate over metrics and methods to plot each metric for each model
    for i, metric in enumerate(metrics):
        for method in methods:
            # Plot the metric with error bars (mean and standard deviation) for internal data
            ax[i].errorbar(
                range(len(results[method]['Ratio'])),  # X-axis as run index
                results[method][f'{metric}_Internal_Mean'],  # Y-axis: mean values for the metric
                yerr=results[method][f'{metric}_Internal_Std'],  # Y-axis error bars: standard deviation
                label=method,  # Legend label for the model
                fmt='-o'  # Line and marker style
            )
        
        # Customize the y-axis, labels, and legend
        ax[i].set_ylim(0, 1)  # All metrics are on a 0 to 1 scale
        ax[i].set_ylabel(f'{metric}', fontsize=14)
        ax[i].legend(fontsize=12)
    
    # Label x-axis only for the last subplot to avoid redundancy
    ax[2].set_xlabel('Run Index', fontsize=14)
    
    # Apply tight layout for clean figure formatting
    plt.tight_layout()
    
    # Display the plot
    plt.show()

