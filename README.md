# Single-Cell RNA-seq Foundation Model Project

This project implements and compares various deep learning foundation models for single-cell RNA sequencing (scRNA-seq) analysis, focusing on trajectory inference and cell state generation across three biological processes: epithelial-mesenchymal transition (EMT), hematopoiesis, and thymocyte development. 

## Project Structure

```
fm-project/
├── data/                   # Datasets and generated samples
├── experiments/            # Jupyter notebooks for model experiments
├── models/                 # Saved model checkpoints
├── src/                    # Source code implementations
├── utils/                  # Utility functions
```

## Directory Contents

### 🧪 `experiments/` - Model Training and Evaluation Notebooks
Jupyter notebooks for preprocessing data and running different models:

**Data Preprocessing:**
- `emt_preprocess.ipynb` - EMT dataset preprocessing
- `hematopoiesis_preprocess.ipynb` - Hematopoiesis dataset preprocessing  
- `thymocyte_preprocess.ipynb` - Thymocyte dataset preprocessing

**Model Experiments:**

Each model has dedicated notebooks for each dataset:
- **scNODE**: `scnode_[dataset].ipynb` - Neural ODE-based generative model
- **scDiffusion**: `scdiff_[dataset].ipynb` - Diffusion model for cell generation
- **scGPT**: `scgpt_[dataset].ipynb` - GPT-based foundation model
- **scVI**: `scvi_[dataset].ipynb` - Variational inference model

### 💻 `src/` - Source Code Implementations
Contains the implementation of two main models:

#### `src/scNODE/` - Neural ODE Model Implementation
Cloned from https://github.com/rsinghlab/scNODE


#### `src/scDiffusion/` - Diffusion Model Implementation
Cloned from https://github.com/EperLuo/scDiffusion

### 🔧 `utils/` - Utility Functions
Helper functions for data processing and evaluation:
- `__init__.py` - Package initialization
- `adata.py` - AnnData object utilities
- `evaluation.py` - Evaluation metrics including marker gene monotonicity
- `latent.py` - Latent space analysis utilities
- `plot.py` - Plotting utilities


