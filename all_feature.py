#Extract features of drugs and proteins and use them to construct a feature matrix, followed by undersampling.
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
import warnings
from rdkit import RDLogger
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')


def extract_drug_features_from_smiles(smiles_string, fingerprint_bits=1024, feature_type='all'):
    """
    Extracting feature values from drug SMILES strings

    Args:
        smiles_string: The SMILES string of the drug
        fingerprint_bits: Number of bits for Morgan fingerprint (default 1024)
        feature_type: 'fingerprint'|'descriptors'|'all' - Feature type selection

    Returns:
        numpy array: Drug feature vector
    """
    try:
        # 转换SMILES为分子对象
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

        return features

    except Exception as e:
        print(f"Error occurred while extracting drug features: {e}")
        return _get_default_features(fingerprint_bits, feature_type)

def _extract_molecular_descriptors(mol):
    """Extract molecular descriptor features"""
    try:
        # Get all available descriptors
        descriptor_list = [desc[0] for desc in Descriptors._descList]
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_list)
        descriptors = calculator.CalcDescriptors(mol)
        # Handling NaN values
        return np.nan_to_num(descriptors).tolist()
    except Exception as e:
        print(f"Error occurred while extracting molecular descriptors: {e}")
        return [0.0] * len(Descriptors._descList)

def _extract_morgan_fingerprints(mol, fingerprint_bits=1024, radius=2):
    """Extract Morgan fingerprint features"""
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fingerprint_bits)
        return list(fp)
    except Exception as e:
        print(f"Error occurred while extracting Morgan fingerprints: {e}")
        return [0] * fingerprint_bits

def _get_default_features(fingerprint_bits, feature_type):
    """Get default feature vector (for error cases)"""
    default_features = []

    if feature_type in ['descriptors', 'all']:
        default_features.extend([0.0] * len(Descriptors._descList))

    if feature_type in ['fingerprint', 'all']:
        default_features.extend([0] * fingerprint_bits)

    return default_features


def extract_protein_features(sequence):
    """
    Extract more detailed protein features from protein sequences
    """
    try:
        if pd.isna(sequence) or len(sequence) == 0:
            return [0] * 25

        prot_analysis = ProteinAnalysis(str(sequence))
        features = []

        # 1. Amino Acid Composition (10 Key Amino Acids)
        aa_composition = prot_analysis.get_amino_acids_percent()
        key_aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'R', 'S']
        for aa in key_aas:
            features.append(aa_composition.get(aa, 0))

        # 2. Physicochemical properties
        features.append(prot_analysis.molecular_weight())
        features.append(prot_analysis.aromaticity())
        features.append(prot_analysis.instability_index())
        features.append(prot_analysis.isoelectric_point())
        features.append(prot_analysis.gravy())
        features.append(len(sequence))

        # 3. Secondary structure propensity
        sec_struct = prot_analysis.secondary_structure_fraction()
        features.extend(sec_struct)

        # 4. Charge and Polarity Characteristics
        features.append(aa_composition.get('D', 0) + aa_composition.get('E', 0))
        features.append(aa_composition.get('K', 0) + aa_composition.get('R', 0) + aa_composition.get('H', 0))
        features.append(aa_composition.get('S', 0) + aa_composition.get('T', 0))

        # 5. Special amino acid ratio
        features.append(aa_composition.get('C', 0))
        features.append(aa_composition.get('P', 0))
        features.append(aa_composition.get('G', 0))

        # 6. Add more sequence-based features
        # Hydrophobic amino acid ratio (A, V, I, L, F, W, M)
        hydrophobic_aas = ['A', 'V', 'I', 'L', 'F', 'W', 'M']
        hydrophobic_ratio = sum(aa_composition.get(aa, 0) for aa in hydrophobic_aas)
        features.append(hydrophobic_ratio)

        # Polar amino acid ratio (N, Q, S, T, Y, C)
        polar_aas = ['N', 'Q', 'S', 'T', 'Y', 'C']
        polar_ratio = sum(aa_composition.get(aa, 0) for aa in polar_aas)
        features.append(polar_ratio)

        return features[:25]
    except:
        return [0] * 25

def calculate_interaction_features(drug_features, protein_features):
    """
    Computing combinatorial features of drug-protein interactions
    """
    interaction_features = []

    # 1. Molecular weight ratio
    drug_mw = drug_features[0] if len(drug_features) > 0 else 1
    protein_mw = protein_features[10] if len(protein_features) > 10 else 1
    interaction_features.append(drug_mw / protein_mw if protein_mw != 0 else 0)

    # 2. Hydrophobicity matching
    drug_logp = drug_features[1] if len(drug_features) > 1 else 0
    protein_gravy = protein_features[14] if len(protein_features) > 14 else 0
    interaction_features.append(abs(drug_logp - protein_gravy))

    # 3. Charge complementarity
    drug_h_donors = drug_features[2] if len(drug_features) > 2 else 0
    drug_h_acceptors = drug_features[3] if len(drug_features) > 3 else 0
    protein_acidic = protein_features[16] if len(protein_features) > 16 else 0
    protein_basic = protein_features[17] if len(protein_features) > 17 else 0

    hbond_complementarity = min(drug_h_donors, protein_acidic) + min(drug_h_acceptors, protein_basic)
    interaction_features.append(hbond_complementarity)

    # 4. Size matching
    drug_atoms = drug_features[10] if len(drug_features) > 10 else 0
    protein_length = protein_features[15] if len(protein_features) > 15 else 1
    interaction_features.append(drug_atoms / protein_length if protein_length != 0 else 0)

    # 5. Hydrophobic complementarity
    drug_hydrophobic_atoms = drug_features[10] if len(drug_features) > 10 else 0  # 使用重原子数作为近似
    protein_hydrophobic = protein_features[23] if len(protein_features) > 23 else 0
    interaction_features.append(drug_hydrophobic_atoms * protein_hydrophobic)

    return interaction_features


def generate_features(input_file, output_file):
    """
    Generate features and save them in a format suitable for input
    """
    # Read data
    print("Reading data...")
    df = pd.read_csv(input_file)
    print(f"Raw data shape: {df.shape}")

    # Check if required columns exist
    required_columns = ['target_id', 'drug_id', 'interaction', 'smiles', 'sequence']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required columns: {col}")

    # Extract drug features
    print("Extracting drug features...")
    drug_features = []
    for i, smiles in enumerate(df['smiles']):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(df)} drugs")
        features = extract_drug_features_from_smiles(smiles)
        drug_features.append(features)

    # Extract protein features
    print("Extracting protein features...")
    protein_features = []
    for i, seq in enumerate(df['sequence']):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(df)} proteins")
        features = extract_protein_features(seq)
        protein_features.append(features)

    # Calculate interaction features
    print("Calculating interaction features...")
    interaction_features = []
    for i in range(len(df)):
        if i % 1000 == 0:
            print(f"Calculated {i}/{len(df)} interactions")
        features = calculate_interaction_features(drug_features[i], protein_features[i])
        interaction_features.append(features)

    # Create feature column names
    drug_feature_names = [f'fp_{i}' for i in range(1241)]

    protein_feature_names = [f'aa_{aa}' for aa in ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'R', 'S']] + [
        'prot_weight', 'aromaticity', 'instability', 'isoelectric_point',
        'gravy', 'seq_length', 'helix', 'turn', 'sheet',
        'acidic_aa', 'basic_aa', 'hydroxyl_aa', 'cysteine', 'proline', 'glycine'
    ]

    interaction_feature_names = [
        'mw_ratio', 'hydrophobicity_diff', 'hbond_complementarity', 'size_ratio', 'hydrophobic_complementarity'
    ]

    # Create feature DataFrame
    drug_df = pd.DataFrame(drug_features, columns=drug_feature_names)
    protein_df = pd.DataFrame(protein_features, columns=protein_feature_names)
    interaction_df = pd.DataFrame(interaction_features, columns=interaction_feature_names)

    # Combine all features
    feature_df = pd.concat([drug_df, protein_df, interaction_df], axis=1)

    # Combine identifier columns with features
    id_columns = df[['target_id', 'drug_id', 'interaction']]
    result_df = pd.concat([id_columns, feature_df], axis=1)

    # Handle missing values
    result_df = result_df.fillna(0)

    # Save results
    result_df.to_csv(output_file, index=False)
    print(f"Feature generation completed! Saved to: {output_file}")
    print(f"Final data shape: {result_df.shape}")
    print(f"Number of feature columns: {len(feature_df.columns)}")

    return result_df


# Main program
if __name__ == "__main__":
    # Input file path
    input_file = "drug_target_pairs.csv"
    output_file = "feature.csv"

    # Generate features
    featured_df = generate_features(input_file, output_file)

    # Display data information
    print("\nGenerated data information:")
    print(f"Column names: {list(featured_df.columns)}")
    print(f"Positive samples: {len(featured_df[featured_df['interaction'] == 1])}")
    print(f"Negative samples: {len(featured_df[featured_df['interaction'] == 0])}")

    # Display feature statistics
    print("\nFeature statistics:")
    print(featured_df.describe())

    # Now you can run your undersampling code
    print("\nNow you can run the fuzzy undersampling code...")