import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve  # Added import
import warnings
import numpy as np
from tqdm import tqdm  # Import tqdm
import torch
import torch.nn as nn
import os
import random

warnings.filterwarnings('ignore')

def set_seed(seed=42):
    """Set all random seeds to ensure experiment reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Set seed to {seed} for reproducibility")
# Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class ProteinSequenceEncoder(nn.Module):
    """Protein sequence encoder"""

    def __init__(self, vocab_size=25, embedding_dim=128, hidden_dim=256, num_layers=2):
        super(ProteinSequenceEncoder, self).__init__()
        # Amino acid vocabulary (20 standard amino acids + special characters)
        self.amino_acid_vocab = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                                 'Y': 19,
                                 'X': 20, 'B': 21, 'Z': 22, 'U': 23, 'O': 24}  # Special characters

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=20)  # Use X for padding
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, batch_first=True)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        lstm_out, (hidden, _) = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim*2)

        # Self-attention mechanism
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Global average pooling
        protein_features = torch.mean(attended, dim=1)  # (batch_size, hidden_dim*2)
        return protein_features

class DrugSMILESEncoder(nn.Module):
    """Drug SMILES encoder"""

    def __init__(self, vocab_size=65, embedding_dim=128, hidden_dim=256, num_layers=2):
        super(DrugSMILESEncoder, self).__init__()
        # SMILES character vocabulary
        self.smiles_vocab = {
            ' ': 0, '#': 1, '(': 2, ')': 3, '+': 4, '-': 5, '.': 6, '0': 7, '1': 8, '2': 9, '3': 10, '4': 11,
            '5': 12, '6': 13, '7': 14, '8': 15, '9': 16, '=': 17, '@': 18, 'A': 19, 'B': 20, 'C': 21, 'F': 22,
            'H': 23, 'I': 24, 'N': 25, 'O': 26, 'P': 27, 'S': 28, '[': 29, '\\': 30, ']': 31, 'a': 32, 'b': 33,
            'c': 34, 'e': 35, 'g': 36, 'i': 37, 'l': 38, 'n': 39, 'o': 40, 'p': 41, 'r': 42, 's': 43, 't': 44
        }

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1d_1 = nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv1d_3 = nn.Conv1d(128, 256, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = embedded.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)

        # 1D convolutional layers
        conv1 = torch.relu(self.bn1(self.conv1d_1(embedded)))
        conv2 = torch.relu(self.bn2(self.conv1d_2(conv1)))
        conv3 = torch.relu(self.bn3(self.conv1d_3(conv2)))

        # Global max pooling
        drug_features = self.pool(conv3).squeeze(-1)  # (batch_size, 256)
        return drug_features

class DeepDTI(nn.Module):
    """Deep-DTI main model"""

    def __init__(self, protein_vocab_size=25, drug_vocab_size=65):
        super(DeepDTI, self).__init__()
        self.protein_encoder = ProteinSequenceEncoder(vocab_size=protein_vocab_size)
        self.drug_encoder = DrugSMILESEncoder(vocab_size=drug_vocab_size)

        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(512 + 256, 256),  # Protein features 512 + Drug features 256
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.7),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.6),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, protein_seq, drug_smiles):
        protein_features = self.protein_encoder(protein_seq)  # (batch_size, 512)
        drug_features = self.drug_encoder(drug_smiles)  # (batch_size, 256)

        combined = torch.cat([protein_features, drug_features], dim=1)  # (batch_size, 768)
        output = self.fusion_layers(combined)  # (batch_size, 1)
        return output.squeeze()

class DTIDataset(Dataset):
    """Custom dataset class"""

    def __init__(self, protein_seqs, drug_smiles, labels, scores , protein_vocab, drug_vocab, max_protein_len=1000,
                 max_drug_len=100):
        self.protein_seqs = protein_seqs
        self.drug_smiles = drug_smiles
        self.labels = labels
        self.protein_vocab = protein_vocab
        self.drug_vocab = drug_vocab
        self.max_protein_len = max_protein_len
        self.max_drug_len = max_drug_len
        self.scores = scores

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        protein_seq = self.protein_seqs[idx]
        drug_smiles = self.drug_smiles[idx]
        label = self.labels[idx]
        score = self.scores[idx]

        # Encode protein sequence
        protein_encoded = self.encode_sequence(protein_seq, self.protein_vocab, self.max_protein_len)
        # Encode SMILES
        drug_encoded = self.encode_sequence(drug_smiles, self.drug_vocab, self.max_drug_len)

        return (torch.LongTensor(protein_encoded),
                torch.LongTensor(drug_encoded),
                torch.FloatTensor([label]),
                torch.FloatTensor([score]))



    def encode_sequence(self, sequence, vocab, max_len):
        """Encode sequence to indices"""
        encoded = []
        for char in sequence[:max_len]:
            encoded.append(vocab.get(char, vocab.get('X', 20)))  # Unknown characters replaced with X

        # Padding
        if len(encoded) < max_len:
            encoded.extend([vocab.get('X', 20)] * (max_len - len(encoded)))
        else:
            encoded = encoded[:max_len]

        return encoded

def load_and_preprocess_data(data_path):
    """Load and preprocess data from a single file"""

    # Read data
    data = pd.read_csv(data_path)

    # Check if required columns exist
    required_columns = ['drug_id', 'target_id', 'interaction', 'smiles', 'sequence']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in data")

    # Basic data information
    print(f"Total interactions: {len(data)}")
    print(f"Unique drugs: {data['drug_id'].nunique()}")
    print(f"Unique targets: {data['target_id'].nunique()}")
    print(f"Interaction distribution:\n{data['interaction'].value_counts()}")

    # Build sample list
    samples = []
    for _, row in data.iterrows():
        samples.append({
            'protein_seq': row['sequence'],
            'drug_smile': row['smiles'],
            'label': row['interaction'],
            'drug_id': row['drug_id'],
            'score': row['score']
        })

    print(f"Total samples: {len(samples)}")
    return samples

def data_split(seed=42, test_size=0.3):
    """Data splitting function, ensuring same positive/negative ratio in training and test sets"""
    # Set random seed for data splitting
    random.seed(seed)
    np.random.seed(seed)

    data_path = "final_data.csv"
    samples = load_and_preprocess_data(data_path)

    if len(samples) == 0:
        print("No samples found!")
        return

    # Separate positive and negative samples
    positive_samples = [s for s in samples if s['label'] == 1]
    negative_samples = [s for s in samples if s['label'] == 0]

    print(f"Positive samples: {len(positive_samples)}")
    print(f"Negative samples: {len(negative_samples)}")

    # Shuffle positive and negative samples
    random.shuffle(positive_samples)
    random.shuffle(negative_samples)

    # Split positive samples by ratio
    positive_split_index = int(len(positive_samples) * test_size)
    test_positive = positive_samples[:positive_split_index]
    train_positive = positive_samples[positive_split_index:]

    # Split negative samples by ratio
    negative_split_index = int(len(negative_samples) * test_size)
    test_negative = negative_samples[:negative_split_index]
    train_negative = negative_samples[negative_split_index:]

    # Build training and test sets
    train_samples = train_positive + train_negative
    test_samples = test_positive + test_negative

    # Shuffle order
    random.shuffle(train_samples)
    random.shuffle(test_samples)

    # Validate ratios
    train_positive_count = len(train_positive)
    train_negative_count = len(train_negative)
    test_positive_count = len(test_positive)
    test_negative_count = len(test_negative)

    print(f"Training set: {len(train_samples)} (Positive: {train_positive_count}, Negative: {train_negative_count})")
    print(f"Training set positive/negative ratio: {train_positive_count/len(train_samples):.3f} : {train_negative_count/len(train_samples):.3f}")
    print(f"Test set: {len(test_samples)} (Positive: {test_positive_count}, Negative: {test_negative_count})")
    print(f"Test set positive/negative ratio: {test_positive_count/len(test_samples):.3f} : {test_negative_count/len(test_samples):.3f}")

    return train_samples, test_samples

def train_model(seed=33):
    """Modified training function with random seed"""
    # Set global random seed
    set_seed(seed)

    # Record experiment settings
    experiment_info = {
        'seed': seed,
        'device': str(device),
        'batch_size': 32,
        'learning_rate': 0.01,
        'epochs': 120
    }

    print("Experiment configuration:")
    for key, value in experiment_info.items():
        print(f"  {key}: {value}")

    # Load data
    train_samples, test_samples = data_split(seed)

    # Extract features and labels
    train_protein = [s['protein_seq'] for s in train_samples]
    train_drug = [s['drug_smile'] for s in train_samples]
    train_labels = [s['label'] for s in train_samples]
    train_scores = []

    test_protein = [s['protein_seq'] for s in test_samples]
    test_drug = [s['drug_smile'] for s in test_samples]
    test_labels = [s['label'] for s in test_samples]
    test_scores = []

    for s in train_samples:
        if 'score' in s and not pd.isna(s['score']) and s['score'] != ' ':
            train_scores.append(float(s['score']))
        else:
            train_scores.append(1.0)

    for s in test_samples:
        if 'score' in s and not pd.isna(s['score']) and s['score'] != ' ':
            test_scores.append(float(s['score']))
        else:
            test_scores.append(1.0)

    print(f"Training set size: {len(train_labels)}")
    print(f"Training set positive/negative ratio: {sum(train_labels)}/{len(train_labels)} = {sum(train_labels) / len(train_labels):.3f}")

    protein_encoder = ProteinSequenceEncoder()
    drug_encoder = DrugSMILESEncoder()

    # Create datasets
    train_dataset = DTIDataset(train_protein, train_drug, train_labels, train_scores,
                               protein_encoder.amino_acid_vocab, drug_encoder.smiles_vocab,
                               )
    test_dataset = DTIDataset(test_protein, test_drug, test_labels, test_scores,
                              protein_encoder.amino_acid_vocab, drug_encoder.smiles_vocab,
                            )

    # Create data loaders with fixed random seed
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              worker_init_fn=seed_worker, generator=g,
                              drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             worker_init_fn=seed_worker, generator=g,
                             drop_last=True)

    # Initialize model
    model = DeepDTI().to(device)

    def weighted_bce_loss(output, target, weight=None):
        bce_loss = nn.BCELoss(reduction='none')(output, target)
        if weight is not None:
            weighted_loss = bce_loss * weight
            return weighted_loss.mean()
        else:
            return bce_loss.mean()

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    num_epochs = 200
    best_auc = 0

    # Record best results
    best_results = {
        'epoch': 0,
        'auc': 0,
        'ef_10': 0,
        'accuracy': 0
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        train_loss = 0
        weighted_train_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc="Training"):
            if len(batch) == 4:
                protein_batch, drug_batch, labels_batch, scores_batch = batch
                scores_batch = scores_batch.to(device)
            else:
                protein_batch, drug_batch, labels_batch = batch
                scores_batch = torch.ones_like(labels_batch).to(device)

            protein_batch = protein_batch.to(device)
            drug_batch = drug_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(protein_batch, drug_batch)

            loss = weighted_bce_loss(outputs, labels_batch.squeeze(), weight=scores_batch.squeeze())
            loss.backward()
            optimizer.step()

            standard_loss = nn.BCELoss()(outputs, labels_batch.squeeze())
            train_loss += standard_loss.item()
            weighted_train_loss += loss.item()

            predicted = (outputs > 0.5).float()
            total += labels_batch.size(0)
            correct += (predicted == labels_batch.squeeze()).sum().item()

        # Validation
        print("Validating...")
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for protein_batch, drug_batch, labels_batch, scores_batch in tqdm(test_loader, desc="Validation"):
                scores_batch = scores_batch.to(device)
                protein_batch = protein_batch.to( device)
                drug_batch = drug_batch.to(device)
                labels_batch = labels_batch.to(device)

                outputs = model(protein_batch, drug_batch)
                loss = weighted_bce_loss(outputs, labels_batch.squeeze(), weight=scores_batch.squeeze())
                test_loss += loss.item()

                predicted = (outputs > 0.5).float()
                test_total += labels_batch.size(0)
                test_correct += (predicted == labels_batch.squeeze()).sum().item()

                all_outputs.extend(outputs.cpu().numpy())
                all_labels.extend(labels_batch.squeeze().cpu().numpy())

        train_accuracy = 100 * correct / total
        test_accuracy = 100 * test_correct / test_total

        all_outputs = np.array(all_outputs)
        all_labels = np.array(all_labels)
        auc = roc_auc_score(all_labels, all_outputs)

        def calculate_ef(labels, predictions, percentage=0.1):
            n_total = len(labels)
            n_top = int(n_total * percentage)
            sorted_indices = np.argsort(predictions)[::-1]
            top_labels = labels[sorted_indices[:n_top]]
            hits_s = np.sum(top_labels == 1)
            hits_t = np.sum(labels == 1)
            ef = (hits_s / n_top) / (hits_t / n_total) if hits_t > 0 else 0
            return ef

        ef_10 = calculate_ef(all_labels, all_outputs, percentage=0.1)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss / len(train_loader):.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, '
              f'Test Loss: {test_loss / len(test_loader):.4f}, '
              f'Test Acc: {test_accuracy:.2f}%, '
              f'Test AUC: {auc:.4f}, '
              f'10% EF: {ef_10:.2f}')
        print(f'New best model saved with AUC: {best_auc:.4f} and 10% EF: {ef_10:.2f}')

        # Save best model
        if auc > best_auc:
            best_auc = auc
            best_results = {
                'epoch': epoch + 1,
                'auc': auc,
                'ef_10': ef_10,
                'accuracy': test_accuracy
            }
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'auc': auc,
                'seed': seed,
                'experiment_info': experiment_info,
                'best_results': best_results
            }, 'best_deep_dti_model.pth')
            print(f'New best model saved with AUC: {best_auc:.4f} and 10% EF: {ef_10:.2f}')

        scheduler.step()

    # Save final experiment results
    results_df = pd.DataFrame([best_results])
    results_df['seed'] = seed
    results_df.to_csv(f'experiment_results_seed_{seed}.csv', index=False)

    print(f'Training completed. Best test AUC: {best_auc:.4f}, Best 10% EF: {ef_10:.2f}')
    print(f"Best results: Epoch {best_results['epoch']}, AUC: {best_results['auc']:.4f}, "
          f"EF10: {best_results['ef_10']:.2f}, Accuracy: {best_results['accuracy']:.2f}%")

def predict_interaction(model, protein_seq, drug_smiles, protein_vocab, drug_vocab):
    """Use trained model to predict interaction"""
    model.eval()

    # Encode input
    dataset = DTIDataset([protein_seq], [drug_smiles], [0], protein_vocab, drug_vocab)
    protein_tensor, drug_tensor, _ = dataset[0]

    with torch.no_grad():
        protein_tensor = protein_tensor.unsqueeze(0).to(device)
        drug_tensor = drug_tensor.unsqueeze(0).to(device)
        prediction = model(protein_tensor, drug_tensor)

    return prediction.item()


def predict_from_dataframe(model, data_df, protein_vocab, drug_vocab):
    """Predict multiple samples in DataFrame"""
    model.eval()
    predictions = []

    for _, row in data_df.iterrows():
        protein_seq = row['sequence']
        drug_smiles = row['smiles']

        # Encode input
        dataset = DTIDataset([protein_seq], [drug_smiles], [0], protein_vocab, drug_vocab)
        protein_tensor, drug_tensor, _ = dataset[0]

        with torch.no_grad():
            protein_tensor = protein_tensor.unsqueeze(0).to(device)
            drug_tensor = drug_tensor.unsqueeze(0).to(device)
            prediction = model(protein_tensor, drug_tensor)
            predictions.append(prediction.item())

    return predictions


if __name__ == "__main__":
    # Train model
    train_model()

