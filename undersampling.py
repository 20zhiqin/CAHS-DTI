import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
import random
import os


def batch_fuzzy_undersampling(input_file, output_file, label_column='interaction',
                              drug_id_column='drug_id', protein_id_column='target_id',
                              batch_size=10000, tau=2, max_iter=100, tol=1e-5):
    """
    Batch processing fuzzy C-means undersampling method for the entire dataset
    Process 10,000 samples at a time, save intermediate results, and merge at the end
    """

    # Read entire dataset
    print("Reading data...")
    df = pd.read_csv(input_file)

    print("Original data distribution:")
    negative_samples = df[df[label_column] == 0]
    positive_samples = df[df[label_column] == 1]
    print(f"Negative samples: {len(negative_samples)}")
    print(f"Positive samples: {len(positive_samples)}")
    print(f"Total samples: {len(df)}")

    # If negative samples are less than or equal to positive samples, no undersampling needed
    if len(negative_samples) <= 55000:
        print("No undersampling needed, saving results directly")
        df.to_csv(output_file, index=False)
        return df

    # Calculate target negative sample count
    target_negative_count = 55000
    print(f"Target negative sample count: {target_negative_count}")

    # Create temporary directory for intermediate results
    temp_dir = "temp_undersampling"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Split negative samples into multiple batches
    negative_indices = negative_samples.index.tolist()
    num_batches = (len(negative_indices) + batch_size - 1) // batch_size
    print(f"Splitting negative samples into {num_batches} batches, up to {batch_size} samples per batch")

    # Calculate sample count for each batch (proportional allocation)
    batch_targets = []
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(negative_indices))
        batch_ratio = (batch_end - batch_start) / len(negative_indices)
        batch_target = max(1, int(target_negative_count * batch_ratio))
        batch_targets.append(batch_target)

    # Adjust last batch target to ensure correct total
    total_batch_target = sum(batch_targets)
    if total_batch_target != target_negative_count:
        diff = target_negative_count - total_batch_target
        batch_targets[-1] += diff

    # Process each batch
    all_undersampled_negative = []

    for batch_idx in range(num_batches):
        print(f"\nProcessing batch {batch_idx + 1}/{num_batches}...")

        # Get current batch sample indices
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(negative_indices))
        batch_indices = negative_indices[batch_start:batch_end]
        batch_samples = df.loc[batch_indices]

        # Process current batch
        batch_target = batch_targets[batch_idx]
        undersampled_batch = process_batch(
            batch_samples, batch_target, label_column,
            drug_id_column, protein_id_column, tau, max_iter, tol
        )

        if undersampled_batch is not None:
            # Save current batch results
            temp_file = os.path.join(temp_dir, f"batch_{batch_idx + 1}.csv")
            undersampled_batch.to_csv(temp_file, index=False)
            all_undersampled_negative.append(undersampled_batch)
            print(f"Batch {batch_idx + 1} processed, sampled samples: {len(undersampled_batch)}")
        else:
            # If processing fails, use random sampling
            print(f"Batch {batch_idx + 1} failed, using random sampling")
            if len(batch_samples) > batch_target:
                selected_indices = np.random.choice(len(batch_samples), batch_target, replace=False)
                undersampled_batch = batch_samples.iloc[selected_indices]
            else:
                undersampled_batch = batch_samples

            temp_file = os.path.join(temp_dir, f"batch_{batch_idx + 1}.csv")
            undersampled_batch.to_csv(temp_file, index=False)
            all_undersampled_negative.append(undersampled_batch)

    # Merge all batch negative samples and positive samples
    print("\nMerging all batch results...")
    if all_undersampled_negative:
        # Method 1: Direct DataFrame concatenation (sufficient memory)
        final_negative = pd.concat(all_undersampled_negative, ignore_index=True)
        final_result = pd.concat([positive_samples, final_negative], ignore_index=True)

        # Method 2: Merge from temporary files (insufficient memory)
        # final_result = merge_from_temp_files(temp_dir, positive_samples)
    else:
        print("No successfully processed batches, returning original data")
        final_result = df

    # Save final results
    final_result.to_csv(output_file, index=False)
    print(f"Final results saved to: {output_file}")

    # Clean up temporary files
    for i in range(num_batches):
        temp_file = os.path.join(temp_dir, f"batch_{i + 1}.csv")
        if os.path.exists(temp_file):
            os.remove(temp_file)

    # Delete temporary directory (if empty)
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass  # Directory not empty, keep it

    print("\nData distribution after undersampling:")
    print(f"Negative samples: {len(final_result[final_result[label_column] == 0])}")
    print(f"Positive samples: {len(final_result[final_result[label_column] == 1])}")
    print(f"Total samples: {len(final_result)}")

    return final_result


def process_batch(batch_samples, batch_target, label_column, drug_id_column,
                  protein_id_column, tau=2, max_iter=100, tol=1e-5):
    """
    Process undersampling for a single batch
    """
    try:
        # If batch sample count is less than or equal to target, return directly
        if len(batch_samples) <= batch_target:
            return batch_samples

        # Separate feature columns and identifier columns
        id_columns = [drug_id_column, protein_id_column, label_column]
        feature_columns = [col for col in batch_samples.columns if col not in id_columns]

        # Extract feature matrix for negative samples
        X_negative = batch_samples[feature_columns].values

        # Standardize features
        scaler = StandardScaler()
        X_negative_scaled = scaler.fit_transform(X_negative)

        # Apply fuzzy C-means clustering undersampling
        selected_indices = fuzzy_c_means_undersampling_with_indices(
            X_negative_scaled, batch_target, tau, max_iter, tol
        )

        # Select corresponding samples
        undersampled_batch = batch_samples.iloc[selected_indices].reset_index(drop=True)

        return undersampled_batch

    except Exception as e:
        print(f"Error processing batch: {e}")
        return None


def fuzzy_c_means_undersampling_with_indices(X_negative, N_min, tau=2, max_iter=100, tol=1e-5):
    """
    Fuzzy C-means clustering-based undersampling, return selected sample indices
    """
    N_maj = X_negative.shape[0]
    if N_maj <= N_min:
        return list(range(N_maj))

    # Calculate number of clusters
    K_maj = int(np.ceil(np.sqrt(N_maj)))

    # Initialize cluster centers
    n_samples, n_features = X_negative.shape
    random_indices = np.random.choice(N_maj, size=K_maj, replace=False)
    centers = X_negative[random_indices, :]

    # Initialize fuzzy membership matrix
    U = np.random.rand(n_samples, K_maj)
    U = U / np.sum(U, axis=1, keepdims=True)

    # Iterative updates
    for iteration in range(max_iter):
        centers_old = centers.copy()

        # Update cluster centers
        for j in range(K_maj):
            numerator = np.sum((U[:, j] ** tau).reshape(-1, 1) * X_negative, axis=0)
            denominator = np.sum(U[:, j] ** tau)
            centers[j] = numerator / denominator if denominator != 0 else centers_old[j]

        # Calculate distance matrix
        distances = euclidean_distances(X_negative, centers)

        # Update membership matrix
        for i in range(n_samples):
            for j in range(K_maj):
                if distances[i, j] == 0:
                    U[i, j] = 1.0 if j == np.argmin(distances[i]) else 0.0
                else:
                    sum_term = np.sum([
                        (distances[i, j] / distances[i, k]) ** (2 / (tau - 1))
                        for k in range(K_maj) if distances[i, k] != 0
                    ])
                    U[i, j] = 1.0 / sum_term if sum_term != 0 else 0.0

        # Check convergence
        if np.linalg.norm(centers - centers_old) < tol:
            break

    # Calculate sample density
    densities = np.zeros(N_maj)
    for j in range(K_maj):
        cluster_indices = np.where(U.argmax(axis=1) == j)[0]
        if len(cluster_indices) > 1:
            cluster_data = X_negative[cluster_indices]
            cluster_distances = euclidean_distances(cluster_data)
            np.fill_diagonal(cluster_distances, np.inf)

            for idx, sample_idx in enumerate(cluster_indices):
                nearest_neighbor_idx = cluster_indices[np.argmin(cluster_distances[idx])]
                densities[nearest_neighbor_idx] += 1

    # Roulette wheel selection
    selected_indices = []

    while len(selected_indices) < N_min:
        total_density = np.sum(densities)
        if total_density == 0:
            probabilities = np.ones(N_maj) / N_maj
        else:
            probabilities = densities / total_density

        cumulative_probs = np.cumsum(probabilities)
        r = random.random()
        selected_idx = np.searchsorted(cumulative_probs, r)

        if selected_idx not in selected_indices:
            selected_indices.append(selected_idx)

    return selected_indices


def merge_from_temp_files(temp_dir, positive_samples):
    """
    Merge results from temporary files
    """
    chunks = [positive_samples]
    batch_files = [f for f in os.listdir(temp_dir) if f.startswith("batch_") and f.endswith(".csv")]
    batch_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    for batch_file in batch_files:
        batch_path = os.path.join(temp_dir, batch_file)
        chunk = pd.read_csv(batch_path)
        chunks.append(chunk)

    result = pd.concat(chunks, ignore_index=True)
    return result


# Example usage code
if __name__ == "__main__":
    # Apply batch undersampling
    result_df = batch_fuzzy_undersampling(
        input_file='feature.csv',
        output_file='undersampled_batch.csv',
        label_column='interaction',
        drug_id_column='drug_id',
        protein_id_column='target_id',
        batch_size=10000  # Process 10,000 samples per batch
    )