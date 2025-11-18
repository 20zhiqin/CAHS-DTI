# Drug-Target Interaction Prediction Pipeline

This project implements a comprehensive pipeline for drug-target interaction (DTI) prediction using machine learning and deep learning techniques. The pipeline includes feature extraction, data balancing (both undersampling and oversampling), and model training with a deep neural network architecture.

## 📁 Project Structure

```
├── all_feature.py          # Document 1: Feature extraction from drugs and proteins
├── k-measns.py          # Document 2: BRICS-based oversampling for drug data
├── model.py              # Document 3: Deep learning model training (DeepDTI)
├── smiles-feature.py     # Document 4: Drug feature extraction for oversampling
└── undersampling.py         # Document 5: Fuzzy  undersampling implementation
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install rdkit biopython scikit-learn torch pandas numpy
```

### Basic Usage

1. **Prepare your data** in CSV format with columns: `target_id`, `drug_id`, `interaction`, `smiles`, `sequence`

2. **Extract features**:
```python
python all_feature.py
```

3. **Balance the dataset**:
```python
python undersampling.py
```

4. **Train the model**:
```python
python model.py
```

## 📊 Pipeline Overview

### 1. Feature Extraction (`feature_extraction.py`)
Extracts comprehensive features from drug SMILES strings and protein sequences:

- **Drug Features**: Molecular descriptors + Morgan fingerprints (1024 bits)
- **Protein Features**: Amino acid composition + physicochemical properties
- **Interaction Features**: Molecular weight ratios, hydrophobicity matching, etc.

### 2. Data Balancing

#### Undersampling (`fuzzy_undersampling.py`)
- Implements fuzzy C-means clustering for intelligent undersampling
- Handles large datasets with batch processing (10,000 samples per batch)
- Maintains data distribution while reducing majority class samples

#### Oversampling (`brics_oversampling.py`)
- Uses BRICS (Breaking Retrosynthetically Interesting Chemical Substructures) for molecular assembly
- Generates novel drug molecules by combining fragments from similar compounds
- Includes timeout mechanisms and cross-platform compatibility

### 3. Model Architecture (`model_training.py`)
**DeepDTI Model**:
- **Protein Encoder**: LSTM with attention mechanism for sequence processing
- **Drug Encoder**: 1D CNN for SMILES string feature extraction
- **Fusion Network**: Multi-layer perceptron for interaction prediction

## 🔧 Configuration

### Key Parameters

**Feature Extraction**:
- `fingerprint_bits`: 1024 (Morgan fingerprint size)
- `feature_type`: 'all' (descriptors + fingerprints)

**Undersampling**:
- `batch_size`: 10000 (samples per batch)
- `tau`: 2 (fuzzy parameter)
- `target_negative_count`: 55000

**Model Training**:
- `batch_size`: 32
- `learning_rate`: 0.001
- `epochs`: 200
- `embedding_dim`: 128

## 📈 Performance Metrics

The model evaluates using:
- **AUC-ROC**: Area Under the Receiver Operating Characteristic curve
- **EF10%**: Enrichment Factor at 10% of screened compounds
- **Accuracy**: Overall prediction accuracy

## 🧪 Example Usage

### Feature Extraction
```python
from feature_extraction import generate_features

# Generate features from drug-target pairs
result_df = generate_features(
    input_file="drug_target_pairs.csv", 
    output_file="feature_matrix.csv"
)
```

### Undersampling
```python
from fuzzy_undersampling import batch_fuzzy_undersampling

# Balance the dataset
balanced_df = batch_fuzzy_undersampling(
    input_file='feature_matrix.csv',
    output_file='balanced_data.csv'
)
```

### Model Training
```python
from model_training import train_model

# Train the DeepDTI model
train_model(seed=42)
```

## 🔬 Advanced Features

### BRICS Molecular Generation
- Automatic fragment decomposition and reassembly
- Multi-criteria molecular selection (drug-likeness, stability, similarity)
- Cross-platform timeout handling (multiprocessing for Windows, signals for Unix)

### Fuzzy Undersampling
- Density-based sample selection using fuzzy clustering
- Batch processing for memory efficiency
- Proportional allocation across clusters

### Deep Learning Model
- Attention mechanisms for protein sequence processing
- Weighted loss function for imbalanced data
- Comprehensive evaluation metrics
- Reproducible training with seed control

## 📋 Input Data Format

Required CSV columns:
- `target_id`: Protein identifier
- `drug_id`: Drug identifier  
- `interaction`: Binary label (1=interaction, 0=no interaction)
- `smiles`: Drug SMILES string
- `sequence`: Protein amino acid sequence

## 💾 Output Files

- `feature.csv`: Extracted feature matrix
- `undersampled_batch.csv`: Balanced dataset after undersampling
- `best_deep_dti_model.pth`: Trained model weights
- `experiment_results_seed_*.csv`: Training results and metrics

## 🛠️ Customization

### Adding New Features
Modify `extract_drug_features_from_smiles()` and `extract_protein_features()` functions to add custom molecular descriptors or protein properties.

### Model Architecture
Adjust the DeepDTI class to modify network architecture, add attention mechanisms, or change fusion strategies.

### Sampling Strategies
Implement custom sampling methods by extending the fuzzy undersampling or BRICS oversampling classes.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{dti_prediction_pipeline,
  title = {Drug-Target Interaction Prediction Pipeline},
  author = {wu zhen},
  year = {2025},
  url = {https://github.com/20zhiqin/CAHS-DTI}
  email = 2474633381@qq.com
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Note

- RDKit installation can be challenging on some systems; consider using conda for easier installation
- Protein sequences should be in standard amino acid code
- SMILES strings should be valid and canonicalized
- Large datasets may require significant memory and processing time

For questions and support, please open an issue in the GitHub repository.
