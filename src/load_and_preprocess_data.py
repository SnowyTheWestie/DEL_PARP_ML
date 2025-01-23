
import os
import warnings
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from chembl_webresource_client.new_client import new_client
from rdkit import DataStructs
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem import rdFingerprintGenerator

def load_data(file_name=None, folder_path='./data', sample_size=None):
    """
    Load data from a CSV file, with options to specify the file name and sample a subset of rows.
    
    Parameters:
    ----------
    file_name : str, optional
        Name of the CSV file to load (default is None, which prompts user input).
        
    folder_path : str, optional
        Path to the folder containing the CSV file (default is 'data' in the current working directory).
        
    sample_size : int, optional
        Number of rows to sample from the dataset. If None, the entire dataset is loaded.
        
    Returns:
    -------
    pd.DataFrame or None
        Loaded DataFrame if successful, None if the file is not found or another error occurs.
    
    Example:
    -------
    df = load_data(file_name='dataset.csv', folder_path='path/to/data', sample_size=100)
    """
    
    # Prompt user for filename if not provided
    if file_name is None:
        file_name = input("Please enter the CSV file name (with .csv extension): ")
    
    # Generate the complete file path
    file_path = os.path.join(folder_path, file_name)
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"The file {file_path} does not exist. Please check the filename and try again.")
        return None
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Optionally sample a subset of the data
    if sample_size is not None and sample_size > 0:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    return df



def assign_labels(df, threshold_dict):
    """
    Assign binary labels to DataFrame columns based on specified thresholds.
    
    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing the columns to be labeled.
        
    threshold_dict : dict
        Dictionary where keys are column names in `df` and values are threshold values.
        Rows with values above or equal to the threshold are labeled 1, and those below are labeled 0.
    
    Returns:
    -------
    pd.DataFrame
        DataFrame with additional columns containing binary labels for each specified column.
        Each new label column is named as "<original_column_name> label".
    """
    
    # Iterate over each column-threshold pair in the dictionary
    for column, threshold in threshold_dict.items():
        # Define the label column name
        label_column = f"{column} label"
        
        # Assign binary labels based on threshold
        df[label_column] = (df[column] >= threshold).astype(int)
    
    return df



def get_chembl_actives(ChEMBL_ID, ta=100):
    """
    Retrieve active compounds for a specified ChEMBL target based on IC50 bioactivity threshold.
    
    Parameters:
    ----------
    ChEMBL_ID : str
        The ChEMBL ID of the target for which active compounds are to be retrieved.
        
    ta : float, optional, default=100
        Threshold for IC50 values (in nM) to classify compounds as active. Only compounds
        with IC50 <= ta are included.
    
    Returns:
    -------
    pd.DataFrame
        DataFrame containing the SMILES strings and unique identifiers of active compounds,
        formatted as ["SMILES", "Identifier"].
    
    Notes:
    -----
    - The function filters for bioactivity data with IC50 values in nM only.
    - Requires the `new_client` object for ChEMBL API interaction.
    
    Example:
    -------
    bioactivities_df = get_chembl_actives("CHEMBL204", ta=100)
    """
    
    # Retrieve bioactivity data for the specified ChEMBL target ID
    bioactivities = new_client.activity
    bioactivities = bioactivities.filter(
        target_chembl_id=ChEMBL_ID, 
        type="IC50", 
        relation="=", 
        assay_type="B"
    ).only([
        "canonical_smiles", "molecule_chembl_id", "type", 
        "standard_units", "relation", "standard_value"
    ])
    
    # Convert bioactivities to DataFrame and ensure correct data types
    bioactivities_df = pd.DataFrame.from_dict(bioactivities)
    bioactivities_df.drop(["units", "value"], axis=1, inplace=True)
    bioactivities_df = bioactivities_df.astype({"standard_value": "float64"})
    
    # Filter out rows with NaN values and limit to IC50 values in nM
    bioactivities_df.dropna(axis=0, how="any", inplace=True)
    bioactivities_df = bioactivities_df[bioactivities_df["standard_units"] == "nM"]
    
    # Filter for compounds with IC50 values below or equal to threshold `ta`
    bioactivities_df = bioactivities_df[bioactivities_df["standard_value"] <= ta]
    
    # Remove duplicates, keep the first instance based on molecule ID, and reset index
    bioactivities_df.drop_duplicates("molecule_chembl_id", keep="first", inplace=True)
    bioactivities_df.reset_index(drop=True, inplace=True)
    
    # Drop unnecessary columns and rename for clarity
    bioactivities_df.drop(["relation", "standard_units", "standard_value", "type"], axis=1, inplace=True)
    bioactivities_df.columns = ["SMILES", "Identifier"]
    
    return bioactivities_df




def sanitize_molecule_list(smiles_list, return_mols=False):
    """
    Sanitize a list of SMILES strings by retaining only valid molecules and, if applicable,
    only the largest fragment in each molecule.
    
    Parameters:
    ----------
    smiles_list : list of str
        List of SMILES strings to sanitize.
    
    return_mols : bool, optional, default=False
        If True, returns both sanitized SMILES strings and corresponding RDKit Mol objects.
    
    Returns:
    -------
    list of str
        Sanitized SMILES strings with valid molecules only.
        
    tuple of lists (if return_mols=True)
        - Sanitized SMILES strings.
        - Corresponding RDKit Mol objects.
        
    Notes:
    -----
    - The function processes each SMILES string individually, converting it to a Mol object.
    - If a molecule has multiple fragments, only the largest fragment is retained.
    - Invalid molecules are skipped.
    
    Example:
    -------
    sanitized_smiles = sanitize_molecule_list(smiles_list)
    sanitized_smiles, sanitized_mols = sanitize_molecule_list(smiles_list, return_mols=True)
    """
    
    # Pre-allocate lists for sanitized SMILES strings and Mol objects
    sanitized_smiles = [''] * len(smiles_list)
    sanitized_mols = [None] * len(smiles_list)

    # Process each SMILES string
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)  # Convert SMILES to Mol
        if mol is None:
            continue  # Skip if invalid molecule
        
        # Retain only the largest fragment if there are multiple
        fragments = Chem.GetMolFrags(mol, asMols=True)
        if len(fragments) > 1:
            largest_fragment = max(fragments, key=lambda frag: frag.GetNumAtoms())
            mol = largest_fragment
        
        # Sanitize and convert to SMILES if valid
        try:
            Chem.SanitizeMol(mol)
            sanitized_smiles[i] = Chem.MolToSmiles(mol)
            sanitized_mols[i] = mol
        except Exception:
            continue  # Skip if sanitization fails
    
    # Remove empty SMILES and None Mol objects
    sanitized_smiles = [s for s in sanitized_smiles if s]
    if return_mols:
        sanitized_mols = [m for m in sanitized_mols if m is not None]
        return sanitized_smiles, sanitized_mols
    
    return sanitized_smiles




def convert_to_numpy_array(fp):
    """
    Converts a fingerprint (fp) to a NumPy array.

    Parameters:
    ----------
    fp : RDKit ExplicitBitVect
        Fingerprint object representing molecular fingerprints.

    Returns:
    -------
    np.ndarray
        A 1D NumPy array of type int8 containing the fingerprint bits.
    
    Notes:
    -----
    This function is useful for converting RDKit fingerprints into a format 
    compatible with machine learning models.
    """
    
    # Initialize a NumPy array to store the fingerprint data
    arr = np.zeros((1,), dtype=np.int8)
    
    # Convert the RDKit fingerprint to a NumPy array
    DataStructs.ConvertToNumpyArray(fp, arr)
    
    return arr


def sanitize_and_generate_mols(smiles_list):
    """
    Sanitize a list of SMILES strings and generate RDKit Mol objects.
    
    Args:
        smiles_list (list): List of SMILES strings.

    Returns:
        tuple: A tuple containing:
            - List of sanitized SMILES strings.
            - List of RDKit Mol objects.
    """
    sanitized_smiles, mol_objects = [], []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            try:
                Chem.SanitizeMol(mol)
                sanitized_smiles.append(Chem.MolToSmiles(mol))
                mol_objects.append(mol)
            except Exception:
                continue
    return sanitized_smiles, mol_objects


def calculate_morgan_fingerprints(mol_list, radius=7, n_bits=2048):
    """
    Calculate Morgan fingerprints for a list of RDKit Mol objects using the MorganGenerator.

    Parameters:
    ----------
    mol_list : list of rdkit.Chem.Mol
        List of RDKit molecule objects.
    radius : int, optional
        Radius of the Morgan fingerprint (default is 2).
    n_bits : int, optional
        Number of bits in the fingerprint (default is 2048).

    Returns:
    -------
    list of numpy.ndarray
        List of Morgan fingerprints represented as NumPy arrays.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fingerprints = []

    for mol in mol_list:
        if mol:
            fp = generator.GetFingerprint(mol)
            arr = np.zeros((1,), dtype=np.int8)
            ConvertToNumpyArray(fp, arr)
            fingerprints.append(arr)
        else:
            fingerprints.append(None)
    return fingerprints