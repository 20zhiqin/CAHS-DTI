import pickle
from rdkit.Chem import AllChem, rdMolDescriptors
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import Descriptors
import pandas as pd
import signal
import time
from contextlib import contextmanager
import multiprocessing
from multiprocessing import TimeoutError
import platform


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Timeout context manager for Unix systems"""

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    # Set signal handler (Unix systems only)
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Cancel alarm
        signal.alarm(0)


def brics_worker(mol_a, mol_b):
    """Worker process function"""
    return brics_based_assembly_simplified(mol_a, mol_b)


def brics_based_assembly_simplified_with_timeout_mp(mol_a, mol_b, timeout_seconds=600):
    """
    BRICS assembly with timeout using multiprocessing (cross-platform)
    """
    # Create process pool
    pool = multiprocessing.Pool(processes=1)

    try:
        # Execute task asynchronously
        result = pool.apply_async(brics_worker, (mol_a, mol_b))

        # Wait for result with timeout
        return result.get(timeout=timeout_seconds)

    except TimeoutError:
        print(f"Timeout: BRICS assembly did not complete within {timeout_seconds} seconds, skipping {mol_a} and {mol_b}")
        pool.terminate()
        pool.join()
        return None
    except Exception as e:
        print(f"Error during BRICS assembly: {e}")
        pool.terminate()
        pool.join()
        return None
    finally:
        pool.close()
        pool.join()


def brics_based_assembly_simplified_with_timeout_signal(mol_a, mol_b, timeout_seconds=600):
    """
    BRICS assembly with timeout using signals (Unix systems)
    """
    try:
        with time_limit(timeout_seconds):
            return brics_based_assembly_simplified(mol_a, mol_b)
    except TimeoutException:
        print(f"Timeout: BRICS assembly did not complete within {timeout_seconds} seconds, skipping {mol_a} and {mol_b}")
        return None
    except Exception as e:
        print(f"Error during BRICS assembly: {e}")
        return None


def get_brics_function():
    """Return appropriate BRICS function based on operating system"""
    if platform.system() == "Windows":
        return brics_based_assembly_simplified_with_timeout_mp
    else:
        return brics_based_assembly_simplified_with_timeout_signal


def is_valid_smiles(smiles):
    """Validate if SMILES string is valid"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def cluster_and_generate_samples(features, smiles_list,
                                 n_clusters=5, similarity_threshold=0.3,
                                 max_pairs_per_cluster=10, random_state=42,
                                 timeout_seconds=600):
    """
    Cluster features and generate new samples (with timeout functionality)

    Args:
        features: Feature matrix (n_samples, n_features)
        smiles_list: Corresponding SMILES list
        n_clusters: Number of clusters
        similarity_threshold: Similarity threshold (point pairs with distance less than this are considered similar)
        max_pairs_per_cluster: Maximum number of similar pairs to process per cluster
        random_state: Random seed
        timeout_seconds: Timeout in seconds

    Returns:
        dict: Dictionary containing original data and newly generated samples
    """
    # Get appropriate BRICS function
    brics_func = get_brics_function()

    # 1. Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 2. K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)

    # 3. Find similar point pairs for each cluster
    new_smiles = []
    new_features = []
    new_cluster_labels = []
    existing_df = pd.read_csv("yamanishi_data/e-smile.csv")

    for cluster_id in range(n_clusters):
        # Get current cluster indices
        cluster_indices = np.where(cluster_labels == cluster_id)[0]

        if len(cluster_indices) < 2:  # Need at least two points to form a pair
            continue

        # Extract current cluster features
        cluster_features = features_scaled[cluster_indices]

        # Calculate intra-cluster pairwise distances
        distance_matrix = pairwise_distances(cluster_features)

        # Find similar point pairs (distance less than threshold)
        similar_pairs = []
        n_points = len(cluster_indices)

        for i in range(n_points):
            for j in range(i + 1, n_points):
                if distance_matrix[i, j] < similarity_threshold:
                    similar_pairs.append((i, j))

        # Limit number of similar pairs per cluster
        if len(similar_pairs) > max_pairs_per_cluster:
            similar_pairs = similar_pairs[:max_pairs_per_cluster]

        # Generate new samples for each similar point pair
        for i, j in similar_pairs:
            # Get original indices
            orig_i = cluster_indices[i]
            orig_j = cluster_indices[j]

            # Generate new sample using timeout function
            start_time = time.time()
            result = brics_func(
                smiles_list[orig_i], smiles_list[orig_j], timeout_seconds
            )
            end_time = time.time()

            elapsed_time = end_time - start_time
            print(f"Processing {smiles_list[orig_i]} and {smiles_list[orig_j]}: took {elapsed_time:.2f} seconds")

            # Check if valid result returned
            if result is None:
                print(f"Warning: Unable to generate new sample for {smiles_list[orig_i]} and {smiles_list[orig_j]}")
                continue

            # Unpack result
            new_smile, score = result
            # Add new row to DataFrame
            if 'score' not in existing_df.columns:
                existing_df['score'] = None
            if 'A' not in existing_df.columns:
                existing_df['A'] = None
            if 'B' not in existing_df.columns:
                existing_df['B'] = None
                # Add new row to DataFrame
            existing_df.loc[len(existing_df)] = {'smiles': new_smile, 'score': score, 'A': smiles_list[orig_i],
                                                 'B': smiles_list[orig_j]}
            # Save back to CSV file
            new_smiles.append(new_smile)

    existing_df.to_csv("e-smile-ok.csv", index=False)

    # 4. Prepare return results
    result = {
        'original': {
            'smiles': smiles_list,
            'features': features,
            'cluster_labels': cluster_labels
        },
        'new': {
            'smiles': new_smiles,
            'features': np.array(new_features) if new_features else np.array([]),
            'cluster_labels': new_cluster_labels
        },
        'kmeans': kmeans,
        'scaler': scaler
    }

    # 5. Print statistics
    print(f"Clustering completed: {n_clusters} clusters")
    print(f"Found {len(new_smiles)} similar pairs, generated {len(new_smiles)} new samples")
    print(f"  Distance matrix shape: {distance_matrix.shape}")
    print(f"  Minimum distance: {np.min(distance_matrix[distance_matrix > 0]):.4f}")
    print(f"  Maximum distance: {np.max(distance_matrix):.4f}")
    print(f"  Average distance: {np.mean(distance_matrix):.4f}")
    # Print sample count per cluster
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"Cluster {cluster_id}: {count} original samples")

    return result


def brics_based_assembly_simplified(mol_a, mol_b):
    """
    Simplified BRICS assembly implementation using RDKit built-in functionality
    """
    if not is_valid_smiles(mol_a):
        return None
    if not is_valid_smiles(mol_b):
        return None
    mol_a = Chem.MolFromSmiles(mol_a)
    mol_b = Chem.MolFromSmiles(mol_b)

    # Get all BRICS fragments for both molecules (returns SMILES strings)
    fragments_a_smiles = list(BRICS.BRICSDecompose(mol_a))
    fragments_b_smiles = list(BRICS.BRICSDecompose(mol_b))

    # Convert SMILES strings back to molecule objects
    fragments_a = [Chem.MolFromSmiles(smi) for smi in fragments_a_smiles]
    fragments_b = [Chem.MolFromSmiles(smi) for smi in fragments_b_smiles]

    # Filter out any failed conversions (those returning None)
    fragments_a = [mol for mol in fragments_a if mol is not None]
    fragments_b = [mol for mol in fragments_b if mol is not None]

    if not fragments_a or not fragments_b:
        raise ValueError("Unable to get valid BRICS fragments from input molecules")

    # Use BRICSBuild to try building new molecules
    # BRICSBuild requires a list of molecule objects
    fragments = fragments_a + fragments_b

    # Build molecules using BRICSBuild
    try:
        products = BRICS.BRICSBuild(fragments)

        result_molecules = []
        for i, product in enumerate(products):
            if product is not None:
                try:
                    # Clean molecule
                    product = Chem.RemoveAllHs(product)
                    smiles = Chem.MolToSmiles(product)
                    result_molecules.append(smiles)
                except Exception as e:
                    print(f"Error processing product: {e}")
                    continue

        if result_molecules:
            return select_best_product(result_molecules, mol_a, mol_b)
        else:
            return None

    except Exception as e:
        print(f"Error during BRICS building: {e}")
        return None


def select_best_product(products, original_a, original_b):
    """
    Select the best molecule from BRICS assembly products

    Selection criteria:
    1. Chemical reasonability (synthesizability, stability)
    2. Similar properties to original molecules
    3. Moderate molecular size
    4. Reasonable drug-like properties
    """
    # Calculate reference properties of original molecules
    props_a = calculate_molecular_properties(original_a)
    props_b = calculate_molecular_properties(original_b)

    # Calculate target property reference range (average of both molecules)
    target_props = {
        'mw': (props_a['mw'] + props_b['mw']) / 2,
        'logp': (props_a['logp'] + props_b['logp']) / 2,
        'hbd': (props_a['hbd'] + props_b['hbd']) / 2,
        'hba': (props_a['hba'] + props_b['hba']) / 2,
        'rotatable_bonds': (props_a['rotatable_bonds'] + props_b['rotatable_bonds']) / 2,
        'tpsa': (props_a['tpsa'] + props_b['tpsa']) / 2
    }

    scored_products = []

    for smiles in products:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # Calculate current molecule properties
            current_props = calculate_molecular_properties(mol)

            # Calculate comprehensive score
            score = calculate_fitness_score(current_props, target_props, mol)

            scored_products.append({
                'smiles': smiles,
                'mol': mol,
                'score': score,
                'properties': current_props
            })

        except Exception as e:
            print(f"Error evaluating molecule {smiles}: {e}")
            continue

    if not scored_products:
        return None

    # Sort by score (higher is better)
    scored_products.sort(key=lambda x: x['score'], reverse=True)

    # Return highest scoring molecule
    best_product = scored_products[0]

    print(f"Best molecule score: {best_product['score']:.3f}")
    print(f"Molecular properties: MW={best_product['properties']['mw']:.1f}, "
          f"LogP={best_product['properties']['logp']:.2f}, "
          f"HBD={best_product['properties']['hbd']}, "
          f"HBA={best_product['properties']['hba']}")

    return best_product['smiles'], best_product['score']


def calculate_molecular_properties(mol):
    """Calculate various molecular properties"""
    return {
        'mw': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'hbd': Descriptors.NumHDonors(mol),
        'hba': Descriptors.NumHAcceptors(mol),
        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'tpsa': Descriptors.TPSA(mol),
        'heavy_atoms': mol.GetNumHeavyAtoms(),
        'aromatic_rings': Descriptors.NumAromaticRings(mol)
    }


def calculate_fitness_score(current_props, target_props, mol):
    """
    Calculate comprehensive fitness score for molecule

    Score considerations:
    1. Property similarity (40%)
    2. Drug-like rule compliance (30%)
    3. Structural reasonability (30%)
    """
    # 1. Property similarity score
    property_score = calculate_property_similarity(current_props, target_props)

    # 2. Drug-likeness score
    druglikeness_score = calculate_druglikeness_score(current_props)

    # 3. Structural reasonability score
    structure_score = calculate_structure_score(mol, current_props)

    # Weighted comprehensive score
    total_score = (
            0.4 * property_score +
            0.3 * druglikeness_score +
            0.3 * structure_score
    )

    return total_score


def calculate_property_similarity(current_props, target_props):
    """Calculate property similarity score"""
    weights = {
        'mw': 0.25,  # Molecular weight
        'logp': 0.25,  # LogP
        'hbd': 0.15,  # Hydrogen bond donors
        'hba': 0.15,  # Hydrogen bond acceptors
        'tpsa': 0.20  # Polar surface area
    }

    similarity_score = 0
    total_weight = 0

    for prop, weight in weights.items():
        if prop in current_props and prop in target_props:
            # Calculate relative difference (smaller is better)
            if target_props[prop] > 0:
                relative_diff = abs(current_props[prop] - target_props[prop]) / target_props[prop]
                # Convert difference to similarity score (0-1)
                prop_score = max(0, 1 - relative_diff)
                similarity_score += prop_score * weight
                total_weight += weight

    return similarity_score / total_weight if total_weight > 0 else 0


def calculate_druglikeness_score(props):
    """Calculate drug-likeness score (based on Lipinski rules and REOS rules)"""
    score = 0
    criteria_met = 0
    total_criteria = 6

    # Lipinski rules (rule of five)
    if props['mw'] <= 500: criteria_met += 1
    if props['logp'] <= 5: criteria_met += 1
    if props['hbd'] <= 5: criteria_met += 1
    if props['hba'] <= 10: criteria_met += 1

    # Extended rules
    if 150 <= props['mw'] <= 600: criteria_met += 1  # Reasonable molecular weight range
    if -2 <= props['logp'] <= 6: criteria_met += 1  # Reasonable LogP range

    return criteria_met / total_criteria


def calculate_structure_score(mol, props):
    """Calculate structural reasonability score"""
    score = 0

    try:
        # Check if molecule has reasonable atom count
        if 10 <= props['heavy_atoms'] <= 80:
            score += 0.3

        # Check for aromatic rings (usually beneficial for stability)
        if props['aromatic_rings'] > 0:
            score += 0.2

        # Check rotatable bond count (moderate is good)
        if 2 <= props['rotatable_bonds'] <= 12:
            score += 0.3

        # Check polar surface area (moderate range)
        if 20 <= props['tpsa'] <= 140:
            score += 0.2

        # Additional reasonability checks
        if Chem.Descriptors.FractionCSP3(mol) > 0.1:  # Some degree of saturated carbons
            score += 0.1

        if Chem.rdMolDescriptors.CalcNumRings(mol) > 0:  # Has ring systems
            score += 0.1

    except Exception as e:
        print(f"Structure score calculation error: {e}")

    return min(score, 1.0)  # Ensure doesn't exceed 1.0


# Usage example
if __name__ == "__main__":
    # Note: When running multiprocessing code on Windows, ensure it's within if __name__ == '__main__'

    # Assume you've already loaded your data
    # smiles_list, features_array = load_your_data()

    # Example data (for testing)
    # Replace with real data

    with open('smiles_features.pkl', 'rb') as f:
        data = pickle.load(f)
    smile_list = []
    feature_list = []
    for i in range(len(data)):
        smile_list.append(data[i][0])
        feature_list.append(data[i][1])
    features = np.array(feature_list)

    # Set timeout to 600 seconds (10 minutes)
    result = cluster_and_generate_samples(
        features=features,
        smiles_list=smile_list,
        n_clusters=5,
        similarity_threshold=40,
        max_pairs_per_cluster=300,
        timeout_seconds=300  # 10 minute timeout
    )

    # Use results
    print(f"Newly generated sample count: {len(result['new']['smiles'])}")