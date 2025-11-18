# Extract drug features from SMILES strings, then perform oversampling
import pickle
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors


def extract_drug_features_from_smiles(smiles_string, fingerprint_bits=1024, feature_type='all'):
    """
    Extract feature values from drug SMILES strings

    Args:
        smiles_string: Drug SMILES string
        fingerprint_bits: Number of bits for Morgan fingerprint (default 1024)
        feature_type: 'fingerprint'|'descriptors'|'all' - Feature type selection

    Returns:
        numpy array: Drug feature vector
    """
    try:
        # Convert SMILES to molecule object
        mol = Chem.MolFromSmiles(smiles_string)

        if mol is None:
            print(f"Warning: Unable to parse SMILES string: {smiles_string}")
            return _get_default_features(fingerprint_bits, feature_type)

        features = []

        # Extract molecular descriptors
        if feature_type in ['descriptors', 'all']:
            desc_features = _extract_molecular_descriptors(mol)
            features.extend(desc_features)

        # Extract Morgan fingerprints
        if feature_type in ['fingerprint', 'all']:
            fp_features = _extract_morgan_fingerprints(mol, fingerprint_bits)
            features.extend(fp_features)

        return np.array(features)

    except Exception as e:
        print(f"Error extracting drug features: {e}")
        return _get_default_features(fingerprint_bits, feature_type)

def _extract_molecular_descriptors(mol):
    """Extract molecular descriptor features"""
    try:
        # Get all available descriptors
        descriptor_list = [desc[0] for desc in Descriptors._descList]
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_list)
        descriptors = calculator.CalcDescriptors(mol)
        # Handle NaN values
        return np.nan_to_num(descriptors).tolist()
    except Exception as e:
        print(f"Error extracting molecular descriptors: {e}")
        return [0.0] * len(Descriptors._descList)

def _extract_morgan_fingerprints(mol, fingerprint_bits=1024, radius=2):
    """Extract Morgan fingerprint features"""
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fingerprint_bits)
        return list(fp)
    except Exception as e:
        print(f"Error extracting Morgan fingerprints: {e}")
        return [0] * fingerprint_bits

def _get_default_features(fingerprint_bits, feature_type):
    """Get default feature vector (for error cases)"""
    default_features = []

    if feature_type in ['descriptors', 'all']:
        default_features.extend([0.0] * len(Descriptors._descList))

    if feature_type in ['fingerprint', 'all']:
        default_features.extend([0] * fingerprint_bits)

    return np.array(default_features)

def get_feature_dimensions(fingerprint_bits=1024, feature_type='all'):
    """
    Get dimensionality information of feature vectors

    Returns:
        dict: Dictionary containing dimensionality information of features
    """
    dimensions = {}

    if feature_type in ['descriptors', 'all']:
        dimensions['descriptors'] = len(Descriptors._descList)

    if feature_type in ['fingerprint', 'all']:
        dimensions['fingerprint'] = fingerprint_bits

    dimensions['total'] = sum(dimensions.values())

    return dimensions

if __name__ == "__main__":
    drug_df = pd.read_csv('yamanishi_data/e-smile.csv')
    smiles_features_list= []
    for i, row in drug_df.iterrows():
        smiles = row['smiles']
        features = extract_drug_features_from_smiles(smiles)

        if features is None:
            continue
        features_np = np.array(features)
        # Save SMILES and feature numpy array as tuple to list
        smiles_features_list.append((smiles, features_np))

        # Optional: Print progress
        if i % 100 == 0:
            print(f"Processed {i + 1}/{len(drug_df)} compounds")

        # Save to file
    output_file = 'smiles_features.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(smiles_features_list, f)

    print(f"Saved {len(smiles_features_list)} compounds to {output_file}")

    # Verify saved data
    print("\nVerifying saved data...")
    with open(output_file, 'rb') as f:
        loaded_data = pickle.load(f)