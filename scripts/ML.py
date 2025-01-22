

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTEN

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from sklearn.model_selection import train_test_split


def balance_data(X_train, y_train, sampling_method=None, sampling_ratio=1):
    """
    Balance the training data using specified sampling methods and ratios.

    Parameters:
    ----------
    X_train : np.array or pd.DataFrame
        Feature matrix for training data.
        
    y_train : np.array or pd.Series
        Labels for the training data.
        
    sampling_method : str, optional
        Method for balancing the dataset. Options:
        - 'undersample': Applies undersampling on majority class.
        - 'oversample': Applies oversampling on minority class.
        - 'smoten': Applies SMOTEN oversampling on minority class.
        - 'combined': Combines undersampling and oversampling.
        Default is None, which returns unbalanced data.
        
    sampling_ratio : float, optional
        Desired ratio of majority to minority class after sampling.
        Default is 1 (equal representation of classes).

    Returns:
    -------
    tuple : (X_train_balanced, y_train_balanced)
        The balanced feature matrix and label array.
    
    Raises:
    ------
    ValueError
        If an invalid sampling method is provided or if the sampling ratio exceeds the class ratio.

    Example:
    -------
    X_train_balanced, y_train_balanced = balance_data(X_train, y_train, sampling_method='combined', sampling_ratio=0.5)
    """

    # Return original data if no sampling is required
    if (sampling_ratio == 0) or (sampling_method is None):
        return X_train, y_train

    # Define available sampling methods
    sampling_methods = {
        'undersample': RandomUnderSampler,
        'oversample': RandomOverSampler,
        'smoten': SMOTEN,
        'combined': None
    }

    # Validate the provided sampling method
    if sampling_method not in sampling_methods:
        raise ValueError(f"The sampling method '{sampling_method}' is not defined. Please choose a valid sampling method.")

    # Calculate class distribution and ratio
    num_minority = sum(y_train == 1)
    num_majority = sum(y_train == 0)
    class_ratio = num_majority / num_minority

    # Validate the sampling ratio
    if sampling_ratio > class_ratio:
        raise ValueError(f"The sampling ratio {sampling_ratio} is greater than the class ratio {class_ratio}. Please choose a smaller sampling ratio.")

    if sampling_method == 'combined':
        # For 'combined' sampling, calculate undersampling and oversampling factors
        f = np.sqrt(class_ratio / sampling_ratio)
        undersampling_strategy = 1 / (sampling_ratio * f)
        oversampling_strategy = 1 / sampling_ratio

        # Apply undersampling on majority class
        undersampler = RandomUnderSampler(sampling_strategy=undersampling_strategy, random_state=None)
        X_train_undersampled, y_train_undersampled = undersampler.fit_resample(X_train, y_train)

        # Apply oversampling on minority class
        oversampler = RandomOverSampler(sampling_strategy=oversampling_strategy, random_state=None)
        X_train_balanced, y_train_balanced = oversampler.fit_resample(X_train_undersampled, y_train_undersampled)

    else:
        # Apply specified single sampling method
        sampler = sampling_methods.get(sampling_method)
        sampler_instance = sampler(sampling_strategy=1 / sampling_ratio, random_state=None)
        X_train_balanced, y_train_balanced = sampler_instance.fit_resample(X_train, y_train)

    return X_train_balanced, y_train_balanced





def model_evaluation(y_true, y_pred, dataset_name='Validation', verbose=True):
    """
    Evaluate model performance by calculating key classification metrics.

    Parameters:
    ----------
    y_true : array-like
        True labels of the dataset.
        
    y_pred : array-like
        Predicted labels from the model.
        
    dataset_name : str, optional
        Name of the dataset being evaluated, used in print statements (default is 'Validation').
        
    verbose : bool, optional
        If True, prints a detailed evaluation report including accuracy, recall, precision, F1 score,
        confusion matrix, and classification report. Default is True.

    Returns:
    -------
    tuple
        Contains the following four metrics:
        - accuracy : float
        - recall : float
        - precision : float
        - f1 : float

    Example:
    -------
    accuracy, recall, precision, f1 = model_evaluation(y_true, y_pred, dataset_name='Test')
    """

    # Calculate classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Print detailed report if verbose is True
    if verbose:
        print(f'{dataset_name} Evaluation:')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}')
        print(f'Classification Report:\n{classification_report(y_true, y_pred, zero_division=0)}')

    return accuracy, recall, precision, f1





def roc_analysis(y_true, avg_probs, dataset_name='Validation'):
    """
    Perform Receiver Operating Characteristic (ROC) analysis, plot the ROC curve,
    and return the optimal decision threshold.

    Parameters:
    ----------
    y_true : array-like
        True binary labels of the dataset.
        
    avg_probs : array-like
        Predicted probabilities of the positive class (label 1).
        
    dataset_name : str, optional
        Name of the dataset, used in the plot title and legend (default is 'Validation').

    Returns:
    -------
    float
        Optimal threshold for classification that maximizes the difference between
        True Positive Rate (TPR) and False Positive Rate (FPR).

    Example:
    -------
    optimal_threshold = roc_analysis(y_true, avg_probs, dataset_name='Test')
    """

    # Calculate False Positive Rate (FPR) and True Positive Rate (TPR) for various thresholds
    fpr, tpr, thresholds = roc_curve(y_true, avg_probs)

    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC curve ({dataset_name})')
    plt.plot([0, 1], [0, 1], 'k--')  # Dashed line for a random classifier
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'Receiver Operating Characteristic ({dataset_name})', fontsize=16)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # Determine the optimal threshold (point maximizing TPR - FPR)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold




def run_logistic_regression(learning_data_df, external_validation_data_df_1, external_validation_data_df_2, train_ratio=0.8, sampling_method=None, sampling_ratio=1, n_runs=5, use_optimized_threshold=True, verbose=True, store_probabilities=False, experimental_data_df=None):
    """
    Perform logistic regression on a dataset with repeated runs for performance evaluation. Optionally, use an optimized threshold and store predicted probabilities.

    Parameters:
    ----------
    learning_data_df : pd.DataFrame
        DataFrame with learning data, including molecular fingerprints (MFP) and labels.
    
    external_validation_data_df_1 : pd.DataFrame
        First external validation set, with molecular fingerprints and labels.

    external_validation_data_df_2 : pd.DataFrame or None
        Optional second external validation set, with molecular fingerprints and labels.
    
    train_ratio : float, default=0.8
        Proportion of data to use for training, with the remainder used for testing.
    
    sampling_method : str or None, default=None
        Sampling method for balancing ('undersample', 'oversample', etc.). If None, no sampling is applied.
    
    sampling_ratio : float, default=1
        Ratio for balancing classes; only relevant if sampling_method is specified.
    
    n_runs : int, default=5
        Number of times to repeat training and testing to gather average performance metrics.
    
    use_optimized_threshold : bool, default=True
        If True, uses an ROC-based optimized threshold; otherwise, defaults to a threshold of 0.5.
    
    verbose : bool, default=True
        If True, prints evaluation metrics for each run.
    
    store_probabilities : bool, default=False
        If True, stores predicted probabilities for the experimental dataset.

    experimental_data_df : pd.DataFrame or None
        DataFrame with experimental data for which predicted probabilities are stored if store_probabilities is True.
    
    Returns:
    -------
    tuple
        Mean and standard deviation of accuracy, recall, precision, and F1 scores for internal, external 1, and (if provided) external 2 datasets.
    
    Notes:
    ------
    - Balanced training data uses `balance_data`.
    - Results from each run are averaged and returned, with optional storage of predicted probabilities.
    """
    
    # Prepare data
    X = np.vstack(learning_data_df['MFP'].values)  # Features for learning data
    y = learning_data_df['Label']                  # Labels for learning data
    X_val_1 = np.vstack(external_validation_data_df_1['MFP'].values)  # Features for first validation set
    y_val_1 = external_validation_data_df_1['Label'].values           # Labels for first validation set

    # Initialize metrics containers for internal and external datasets
    metrics_internal = {'accuracy': [], 'recall': [], 'precision': [], 'f1': []}
    metrics_external_1 = {'accuracy': [], 'recall': [], 'precision': [], 'f1': []}
    
    # Prepare second validation set if provided
    if external_validation_data_df_2 is not None:
        X_val_2 = np.vstack(external_validation_data_df_2['MFP'].values)
        y_val_2 = external_validation_data_df_2['Label'].values
        metrics_external_2 = {'accuracy': [], 'recall': [], 'precision': [], 'f1': []}
    else:
        X_val_2, y_val_2, metrics_external_2 = None, None, None

    # Optional storage of probabilities for experimental data
    if store_probabilities:
        experimental_probs = np.zeros((len(experimental_data_df), n_runs))

    # Run logistic regression multiple times to gather metrics
    for run_idx in range(n_runs):
        # Split learning data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=None)

        # Balance training data if a sampling method is specified
        X_train_balanced, y_train_balanced = balance_data(X_train, y_train, sampling_method, sampling_ratio)

        # Ensure that balanced data has both classes present
        if len(np.unique(y_train_balanced)) < 2:
            print("Warning: The balanced training data contains only one class. Skipping this run.")
            continue

        # Train logistic regression model
        lr_model = LogisticRegression(max_iter=500, random_state=None)
        lr_model.fit(X_train_balanced, y_train_balanced)

        # Predict probabilities for test and validation sets
        prob_test = lr_model.predict_proba(X_test)[:, 1]
        prob_val_1 = lr_model.predict_proba(X_val_1)[:, 1]
        if X_val_2 is not None:
            prob_val_2 = lr_model.predict_proba(X_val_2)[:, 1]
        
        # Store probabilities for experimental dataset if requested
        if store_probabilities and experimental_data_df is not None:
            X_exp = np.vstack(experimental_data_df['MFP'].values)
            prob_exp = lr_model.predict_proba(X_exp)[:, 1]
            experimental_probs[:, run_idx] = prob_exp

        # Determine classification threshold
        if use_optimized_threshold:
            optimal_threshold = roc_analysis(y_test, prob_test, dataset_name='Internal')
            y_pred_best = (prob_test >= optimal_threshold).astype(int)
            y_pred_val_1_best = (prob_val_1 >= optimal_threshold).astype(int)
            if X_val_2 is not None:
                y_pred_val_2_best = (prob_val_2 >= optimal_threshold).astype(int)
        else:
            y_pred_best = (prob_test >= 0.5).astype(int)
            y_pred_val_1_best = (prob_val_1 >= 0.5).astype(int)
            if X_val_2 is not None:
                y_pred_val_2_best = (prob_val_2 >= 0.5).astype(int)

        # Evaluate model performance for internal dataset
        acc_int, rec_int, prec_int, f1_int = model_evaluation(y_test, y_pred_best, dataset_name='Internal', verbose=verbose)
        metrics_internal['accuracy'].append(acc_int)
        metrics_internal['recall'].append(rec_int)
        metrics_internal['precision'].append(prec_int)
        metrics_internal['f1'].append(f1_int)

        # Evaluate performance for first external validation dataset
        acc_ext_1, rec_ext_1, prec_ext_1, f1_ext_1 = model_evaluation(y_val_1, y_pred_val_1_best, dataset_name='External 1', verbose=verbose)
        metrics_external_1['accuracy'].append(acc_ext_1)
        metrics_external_1['recall'].append(rec_ext_1)
        metrics_external_1['precision'].append(prec_ext_1)
        metrics_external_1['f1'].append(f1_ext_1)

        # Evaluate performance for second external validation dataset if provided
        if X_val_2 is not None:
            acc_ext_2, rec_ext_2, prec_ext_2, f1_ext_2 = model_evaluation(y_val_2, y_pred_val_2_best, dataset_name='External 2', verbose=verbose)
            metrics_external_2['accuracy'].append(acc_ext_2)
            metrics_external_2['recall'].append(rec_ext_2)
            metrics_external_2['precision'].append(prec_ext_2)
            metrics_external_2['f1'].append(f1_ext_2)

    # Calculate mean and standard deviation for internal metrics
    avg_internal = {k: np.mean(v) for k, v in metrics_internal.items()}
    std_internal = {k: np.std(v) for k, v in metrics_internal.items()}

    # Calculate mean and standard deviation for first external dataset metrics
    avg_external_1 = {k: np.mean(v) for k, v in metrics_external_1.items()}
    std_external_1 = {k: np.std(v) for k, v in metrics_external_1.items()}

    # Calculate mean and standard deviation for second external dataset metrics if available
    avg_external_2, std_external_2 = None, None
    if X_val_2 is not None:
        avg_external_2 = {k: np.mean(v) for k, v in metrics_external_2.items()}
        std_external_2 = {k: np.std(v) for k, v in metrics_external_2.items()}

    # Store average probabilities in the experimental data DataFrame if requested
    if store_probabilities and experimental_data_df is not None:
        experimental_data_df['Average_Probability'] = np.mean(experimental_probs, axis=1)

    return avg_internal, std_internal, avg_external_1, std_external_1, avg_external_2, std_external_2



def explore_sampling(learning_data_df, external_validation_data_df_1, external_validation_data_df_2, ratios, n_runs=5, sampling_method='combined', use_optimized_threshold=True, model_name='logistic_regression'):
    """
    Explore model performance by varying the ratio of active to inactive compounds in training data.

    Parameters:
    ----------
    learning_data_df : pd.DataFrame
        DataFrame containing training data with molecular fingerprints and labels.
    
    external_validation_data_df_1 : pd.DataFrame
        First external validation dataset with molecular fingerprints and labels.

    external_validation_data_df_2 : pd.DataFrame or None
        Optional second external validation dataset with molecular fingerprints and labels.
    
    ratios : list of float or int
        List of sampling ratios to explore. A value of 0 applies the natural class ratio of the dataset.
    
    n_runs : int, default=5
        Number of training runs to average metrics for each ratio.
    
    sampling_method : str, default='combined'
        Method to balance the data, such as 'undersample', 'oversample', etc.
    
    use_optimized_threshold : bool, default=True
        If True, uses an optimized ROC threshold for predictions; otherwise, a 0.5 threshold is applied.
    
    model_name : str, default='logistic_regression'
        Specifies the machine learning model to use; options include 'logistic_regression', 'random_forest', etc.
    
    Returns:
    -------
    pd.DataFrame
        DataFrame containing the mean and standard deviation of accuracy, recall, precision, and F1 score 
        for each dataset (internal, external 1, external 2) across all runs and sampling ratios.

    Notes:
    ------
    - Calls `run_ml_models` for each sampling ratio, retrieving average metrics for the given ratio.
    - If `ratio` is 0, the natural class ratio in the training data is used for balancing.
    """
    
    results = []  # Initialize list to store results

    for i in ratios:
        print("---------------------------------------------------------------")
        print(f"Ratio: {i}")

        # Run model training and evaluation for the specified ratio
        avg_internal, std_internal, avg_external_1, std_external_1, avg_external_2, std_external_2 = run_ml_models(
            learning_data_df, external_validation_data_df_1, external_validation_data_df_2,
            model_name=model_name, sampling_method=sampling_method, sampling_ratio=i, n_runs=n_runs, use_optimized_threshold=use_optimized_threshold
        )

        # Calculate natural class ratio if ratio is set to 0
        if i == 0:
            y = learning_data_df['Label']
            num_minority = sum(y == 1)
            num_majority = sum(y == 0)
            ratio_value = num_majority / num_minority
            print("This is the ratio:", ratio_value)
        else:
            ratio_value = i
            
        # Append metrics for each dataset and ratio to the results list
        results.append({
            'Ratio': ratio_value,
            'Accuracy_Internal_Mean': avg_internal['accuracy'],
            'Recall_Internal_Mean': avg_internal['recall'],
            'Precision_Internal_Mean': avg_internal['precision'],
            'F1_Score_Internal_Mean': avg_internal['f1'],
            'Accuracy_Internal_Std': std_internal['accuracy'],
            'Recall_Internal_Std': std_internal['recall'],
            'Precision_Internal_Std': std_internal['precision'],
            'F1_Score_Internal_Std': std_internal['f1'],
            'Accuracy_External1_Mean': avg_external_1['accuracy'],
            'Recall_External1_Mean': avg_external_1['recall'],
            'Precision_External1_Mean': avg_external_1['precision'],
            'F1_Score_External1_Mean': avg_external_1['f1'],
            'Accuracy_External1_Std': std_external_1['accuracy'],
            'Recall_External1_Std': std_external_1['recall'],
            'Precision_External1_Std': std_external_1['precision'],
            'F1_Score_External1_Std': std_external_1['f1'],
            'Accuracy_External2_Mean': avg_external_2['accuracy'],
            'Recall_External2_Mean': avg_external_2['recall'],
            'Precision_External2_Mean': avg_external_2['precision'],
            'F1_Score_External2_Mean': avg_external_2['f1'],
            'Accuracy_External2_Std': std_external_2['accuracy'],
            'Recall_External2_Std': std_external_2['recall'],
            'Precision_External2_Std': std_external_2['precision'],
            'F1_Score_External2_Std': std_external_2['f1']
        })

    # Convert the results list to a DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df


def extract_building_blocks(identifier):
    """
    Extracts numerical building block values from a compound identifier.
    
    Parameters:
    ----------
    identifier : str
        String identifier in the format 'Axxx_Bxxx', where 'A' and 'B' are 
        prefixes for building blocks and 'xxx' represents numeric values.
    
    Returns:
    -------
    tuple of int
        A tuple containing the integer values of building block A and B.

    Example:
    -------
    >>> extract_building_blocks("A012_B034")
    (12, 34)
    """
    
    # Split identifier into components by underscore
    a_block, b_block = identifier.split('_')
    
    # Convert the numeric parts to integers after removing the prefixes
    a_block_num = int(a_block[1:])  # Extract integer from 'Axxx'
    b_block_num = int(b_block[1:])  # Extract integer from 'Bxxx'
    
    return a_block_num, b_block_num




def leave_one_out_analysis(learning_data_df):
    """
    Perform a leave-one-out analysis on active molecules in the dataset.

    This function iterates over each active molecule in `learning_data_df`, trains a logistic regression 
    model on all other molecules, and predicts the probability for the left-out molecule. The results 
    include x and y coordinates parsed from identifiers and predicted probabilities for plotting.

    Parameters:
    ----------
    learning_data_df : pd.DataFrame
        DataFrame containing molecular data, including fingerprints (as 'MFP'), labels (as 'Label'), 
        and unique identifiers.

    Returns:
    -------
    tuple of lists
        - x_coords : list of int
            X-coordinates (parsed from identifiers) for each active molecule.
        
        - y_coords : list of int
            Y-coordinates (parsed from identifiers) for each active molecule.
        
        - probabilities : list of float
            Predicted probabilities of being active for each active molecule when left out.

    Notes:
    ------
    - Only molecules labeled as active (Label = 1) are included in the leave-one-out process.
    - `extract_building_blocks` is assumed to be a helper function that parses the building blocks 
      from each identifier.
    - The predicted probability for each left-out molecule is stored to assess model performance 
      and visualize data.
    """
    
    active_molecules = learning_data_df[learning_data_df['Label'] == 1]  # Filter for active molecules
    
    # Lists to store coordinates and probabilities for plotting
    x_coords, y_coords, probabilities = [], [], []

    # Iterate through each active molecule
    for index, row in active_molecules.iterrows():
        # Leave-one-out: Train on all but the current molecule
        train_data = learning_data_df.drop(index)
        X_train = np.vstack(train_data['MFP'].values)
        y_train = train_data['Label'].values
        
        # Initialize and train logistic regression model
        lr_model = LogisticRegression(max_iter=500)
        lr_model.fit(X_train, y_train)
        
        # Predict probability of being active for the left-out molecule
        left_out_mfp = row['MFP'].reshape(1, -1)
        prob = lr_model.predict_proba(left_out_mfp)[0][1]  # Probability of being active
        
        # Parse coordinates from the molecule identifier
        a_block, b_block = extract_building_blocks(row['Identifier'])
        
        # Store x and y coordinates and predicted probability
        x_coords.append(a_block)
        y_coords.append(b_block)
        probabilities.append(prob)

    # Return data for plotting
    return x_coords, y_coords, probabilities



def run_ml_models(learning_data_df, external_validation_data_df_1, external_validation_data_df_2, model_name='logistic_regression', train_ratio=0.8, sampling_method=None, sampling_ratio=1, n_runs=5, use_optimized_threshold=True):
    """
    Train and evaluate multiple machine learning models on internal and external datasets.

    Parameters:
    ----------
    learning_data_df : pd.DataFrame
        DataFrame containing training data, including molecular fingerprints ('MFP') and labels ('Label').
        
    external_validation_data_df_1 : pd.DataFrame
        DataFrame with the first external validation dataset, with molecular fingerprints ('MFP') and labels.
        
    external_validation_data_df_2 : pd.DataFrame
        DataFrame with the second external validation dataset, with molecular fingerprints ('MFP') and labels.
        
    model_name : str, default='logistic_regression'
        The name of the model to use. Options include 'random_forest', 'svm', 'neural_network', 'naive_bayes', 'elastic_net',
        'hist_gradient_boosting', and 'logistic_regression'.
        
    train_ratio : float, default=0.8
        Ratio of data to use for training. The rest is used for internal testing.
        
    sampling_method : str, optional
        Method to balance data, if any. Options include 'undersample', 'oversample', 'smoten', 'combined'.
        
    sampling_ratio : int, default=1
        Ratio for data balancing.
        
    n_runs : int, default=5
        Number of iterations to run the model training and evaluation.
        
    use_optimized_threshold : bool, default=True
        If True, uses optimized thresholding based on ROC analysis.

    Returns:
    -------
    tuple
        Average and standard deviation of metrics (accuracy, recall, precision, F1 score) for internal,
        external validation 1, and external validation 2 datasets.
    """

    # Prepare training and validation data
    X = np.vstack(learning_data_df['MFP'].values)
    y = learning_data_df['Label'].values
    X_val_1 = np.vstack(external_validation_data_df_1['MFP'].values)
    y_val_1 = external_validation_data_df_1['Label'].values
    X_val_2 = np.vstack(external_validation_data_df_2['MFP'].values)
    y_val_2 = external_validation_data_df_2['Label'].values
    
    # Model selection based on specified model name
    if model_name == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=None)
    elif model_name == 'svm':
        model = SVC(probability=True, random_state=None)
    elif model_name == 'neural_network':
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=None)
    elif model_name == 'naive_bayes':
        model = GaussianNB()
    elif model_name == 'elastic_net':
        model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=500)
    elif model_name == 'hist_gradient_boosting':
        model = HistGradientBoostingClassifier(random_state=None)
    elif model_name == 'logistic_regression':
        model = LogisticRegression(max_iter=500, random_state=None)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Initialize performance metrics
    metrics_internal = {'accuracy': [], 'recall': [], 'precision': [], 'f1': []}
    metrics_external_1 = {'accuracy': [], 'recall': [], 'precision': [], 'f1': []}
    metrics_external_2 = {'accuracy': [], 'recall': [], 'precision': [], 'f1': []}

    # Run multiple rounds of training and evaluation
    for run_idx in range(n_runs):
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=None)
        
        # Balance training data
        X_train_balanced, y_train_balanced = balance_data(X_train, y_train, sampling_method, sampling_ratio)

        # Train model
        model.fit(X_train_balanced, y_train_balanced)

        # Predictions on internal and external datasets
        y_pred_test = model.predict(X_test)
        y_pred_val_1 = model.predict(X_val_1)
        y_pred_val_2 = model.predict(X_val_2)
        
        # Evaluate and collect metrics for internal set
        acc_int, rec_int, prec_int, f1_int = model_evaluation(y_test, y_pred_test, dataset_name='Internal', verbose=False)
        metrics_internal['accuracy'].append(acc_int)
        metrics_internal['recall'].append(rec_int)
        metrics_internal['precision'].append(prec_int)
        metrics_internal['f1'].append(f1_int)
        
        # Evaluate and collect metrics for external validation set 1
        acc_ext_1, rec_ext_1, prec_ext_1, f1_ext_1 = model_evaluation(y_val_1, y_pred_val_1, dataset_name='External 1', verbose=False)
        metrics_external_1['accuracy'].append(acc_ext_1)
        metrics_external_1['recall'].append(rec_ext_1)
        metrics_external_1['precision'].append(prec_ext_1)
        metrics_external_1['f1'].append(f1_ext_1)
        
        # Evaluate and collect metrics for external validation set 2
        acc_ext_2, rec_ext_2, prec_ext_2, f1_ext_2 = model_evaluation(y_val_2, y_pred_val_2, dataset_name='External 2', verbose=False)
        metrics_external_2['accuracy'].append(acc_ext_2)
        metrics_external_2['recall'].append(rec_ext_2)
        metrics_external_2['precision'].append(prec_ext_2)
        metrics_external_2['f1'].append(f1_ext_2)
        
        # Print confusion matrices for each run
        print(f"Run {run_idx + 1} - Confusion Matrix (Internal):\n{confusion_matrix(y_test, y_pred_test)}")
        print(f"Run {run_idx + 1} - Confusion Matrix (External 1):\n{confusion_matrix(y_val_1, y_pred_val_1)}")
        print(f"Run {run_idx + 1} - Confusion Matrix (External 2):\n{confusion_matrix(y_val_2, y_pred_val_2)}")

    # Calculate average and standard deviation of metrics for all runs
    def calculate_stats(metrics):
        avg = {k: np.mean(v) for k, v in metrics.items()}
        std = {k: np.std(v) for k, v in metrics.items()}
        return avg, std

    avg_internal, std_internal = calculate_stats(metrics_internal)
    avg_external_1, std_external_1 = calculate_stats(metrics_external_1)
    avg_external_2, std_external_2 = calculate_stats(metrics_external_2)
    
    return avg_internal, std_internal, avg_external_1, std_external_1, avg_external_2, std_external_2


def compare_models(learning_data_df, external_validation_data_df_1, external_validation_data_df_2, ratios, n_runs=5):
    """
    Compare performance of multiple machine learning models on training and validation datasets.

    Parameters:
    ----------
    learning_data_df : pd.DataFrame
        DataFrame containing training data with molecular fingerprints ('MFP') and labels ('Label').

    external_validation_data_df_1 : pd.DataFrame
        DataFrame with the first external validation dataset, containing molecular fingerprints and labels.

    external_validation_data_df_2 : pd.DataFrame
        DataFrame with the second external validation dataset, containing molecular fingerprints and labels.

    ratios : list
        List of ratios for data balancing, which indicates active-to-inactive ratio in the training set.
        
    n_runs : int, default=5
        Number of iterations to run each model for averaging performance.

    Returns:
    -------
    dict
        Dictionary where each key is a model name, and each value is a DataFrame containing the model's
        performance metrics (accuracy, recall, precision, F1 score) for each dataset across different ratios.
    """
    
    # Define models to compare
    methods = [
        'logistic_regression', 'random_forest', 'svm', 'neural_network', 
        'naive_bayes', 'elastic_net', 'hist_gradient_boosting'
    ]
    
    # Dictionary to store performance results for each model
    model_results = {}

    # Loop through each model and perform sampling, training, and evaluation
    for method in methods:
        print(f"Running {method.upper()} model")
        
        # Perform sampling and evaluation using the specified method
        results = explore_sampling(
            learning_data_df, external_validation_data_df_1, external_validation_data_df_2,
            ratios=ratios, n_runs=n_runs, sampling_method='undersample', 
            use_optimized_threshold=False, model_name=method
        )
        
        # Store results in dictionary with model name as key
        model_results[method] = results

    return model_results


def model_comparison_to_df(model_comparison_results):
    """
    Convert the results of model comparisons into a structured pandas DataFrame.

    Parameters:
    ----------
    model_comparison_results : dict
        Dictionary containing performance metrics for various models.
        Each key is a model name, and the value is another dictionary with keys like 
        'Ratio', 'Accuracy_Internal_Mean', 'Recall_Internal_Mean', etc.

    Returns:
    -------
    pd.DataFrame
        A DataFrame where each row corresponds to a single model and ratio combination, 
        containing all associated performance metrics.

    Example:
    -------
    results_df = model_comparison_to_df(model_comparison_results)
    """
    rows = []
    
    # Loop through each model in the results dictionary
    for model_name, result in model_comparison_results.items():
        for i, ratio in enumerate(result['Ratio']):
            # Append metrics for the current model and ratio
            rows.append({
                'Model': model_name,
                'Ratio': ratio,
                'Accuracy_Internal_Mean': result['Accuracy_Internal_Mean'][i],
                'Accuracy_Internal_Std': result['Accuracy_Internal_Std'][i],
                'Recall_Internal_Mean': result['Recall_Internal_Mean'][i],
                'Recall_Internal_Std': result['Recall_Internal_Std'][i],
                'Precision_Internal_Mean': result['Precision_Internal_Mean'][i],
                'Precision_Internal_Std': result['Precision_Internal_Std'][i],
                'F1_Score_Internal_Mean': result['F1_Score_Internal_Mean'][i],
                'F1_Score_Internal_Std': result['F1_Score_Internal_Std'][i],
                'Accuracy_External1_Mean': result['Accuracy_External1_Mean'][i],
                'Accuracy_External1_Std': result['Accuracy_External1_Std'][i],
                'Recall_External1_Mean': result['Recall_External1_Mean'][i],
                'Recall_External1_Std': result['Recall_External1_Std'][i],
                'Precision_External1_Mean': result['Precision_External1_Mean'][i],
                'Precision_External1_Std': result['Precision_External1_Std'][i],
                'F1_Score_External1_Mean': result['F1_Score_External1_Mean'][i],
                'F1_Score_External1_Std': result['F1_Score_External1_Std'][i],
                'Accuracy_External2_Mean': result['Accuracy_External2_Mean'][i],
                'Accuracy_External2_Std': result['Accuracy_External2_Std'][i],
                'Recall_External2_Mean': result['Recall_External2_Mean'][i],
                'Recall_External2_Std': result['Recall_External2_Std'][i],
                'Precision_External2_Mean': result['Precision_External2_Mean'][i],
                'Precision_External2_Std': result['Precision_External2_Std'][i],
                'F1_Score_External2_Mean': result['F1_Score_External2_Mean'][i],
                'F1_Score_External2_Std': result['F1_Score_External2_Std'][i]
            })
    
    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(rows)
    return df

