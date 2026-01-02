<img src="assets/contra-ctgan.png" alt="ContraCTGAN" width="900"><br>
  <a href="https://drive.google.com/file/d/17HE3AtMUaq8K_VrRJRbNasXjsSph8V9S/view?usp=drive_link">
    <img
      src="https://img.shields.io/badge/Read%20Paper-PDF-black?style=for-the-badge&labelColor=0057FF&logo=adobeacrobat&logoColor=white"
      alt="Read Paper"
    />
  </a>


A tabular data generation framework that integrates SimCLR-style contrastive loss into CTGAN to generate realistic, privacy-preserving credit card transaction data. By enhancing the discriminator with an auxiliary contrastive task, ContraCTGAN better preserves inter-feature dependencies and marginal distributions in highly imbalanced datasets.

<p>
  <a href="#overview"><img src="https://img.shields.io/badge/Overview-111111?style=for-the-badge" alt="Overview"></a>
  <a href="#prerequisites"><img src="https://img.shields.io/badge/Prerequisites-111111?style=for-the-badge" alt="Prerequisites"></a>
  <a href="#project-structure"><img src="https://img.shields.io/badge/Structure-111111?style=for-the-badge" alt="Project Structure"></a>
  <a href="#installation"><img src="https://img.shields.io/badge/Install-111111?style=for-the-badge" alt="Installation"></a>
  <a href="#usage"><img src="https://img.shields.io/badge/Usage-111111?style=for-the-badge" alt="Usage"></a>
  <a href="#results"><img src="https://img.shields.io/badge/Results-111111?style=for-the-badge" alt="Results"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Citation-111111?style=for-the-badge" alt="Citation"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-111111?style=for-the-badge" alt="License"></a>
</p>


## Overview

This project enhances the standard CTGAN architecture by introducing a contrastive branch to the discriminator:

1. **Data Preprocessing** — Mode-specific normalization for continuous columns and conditional sampling for discrete columns.
2. **Embedding** — Map tabular inputs (real + condition vector) into a latent space via an MLP Embedder.
3. **Contrastive Augmentation** — Generate lightly noised copies of real batches () to create positive pairs.
4. **Auxiliary Loss Integration** — Optimize the discriminator using both WGAN-GP loss and NT-Xent (SimCLR) loss.
5. **Generator Training** — Train the generator to minimize the refined discriminator's feedback, resulting in higher fidelity synthetic data.

The model minimizes the following objective:

$$\mathcal{L}_D^{\text{total}} = \mathcal{L}_D^{\text{WGAN-GP}} + \lambda_{\text{contrastive}} \cdot \mathcal{L}_{\text{NT-Xent}}$$



## Prerequisites

* **Python** 3.10+
* **PyTorch** 2.6+
* **CUDA** 12.1+ (for GPU acceleration)

**Tested on:** NVIDIA RTX 3090 (24GB) • Ryzen 7 7800X3D • 32GB RAM



## Project Structure

```
ContraCTGAN/
├── Datasets/                   # Dataset directory (not tracked)
├── Models/                     # Trained model weights (LFS tracked)
├── Notebooks/
│   └── synthetic_evaluation.ipynb  # Fidelity + utility evaluation
├── Scripts/
│   ├── contrastive_ctgan.py    # Main training script (single run)
│   ├── contrastive_ctganF.py   # Hyperparameter sweep script
│   └── utils/
├── SyntheticDatasets/          # Generated output CSVs
├── requirements.txt            # Python dependencies
└── README.md

```



## Installation

1. **Clone the repository**
```bash
git clone https://github.com/johnamit/ContraCTGAN.git
cd ContraCTGAN

```


2. **Install dependencies**
It is recommended to use a virtual environment (conda or venv).
```bash
pip install -r requirements.txt

```


3. **Prepare your dataset**
Download the split [Credit Card Fraud Detection dataset](https://drive.google.com/drive/folders/1aHzWk2kSIdyNrQwaKhpYbTWOrBAYJQGf?usp=sharing) and place it in the `Datasets` folder:
```
Datasets/
├── creditcard_train.csv    # 80% stratified split
└── creditcard_test.csv     # 20% stratified split

```





## Usage

### Training

#### Default Configuration

To run a training sweep or default training session:

```bash
python Scripts/contrastive_ctganF.py

```

This will:

1. Load `creditcard_train.csv`.
2. Train the ContraCTGAN model.
3. Save weights to `Models/` and synthetic samples to `SyntheticDatasets/`.

#### Configuration Parameters

You can modify the training parameters within the script constructors. Key arguments include:

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `contrastive_lambda` | float | `0.5` | Weight of the NT-Xent contrastive loss |
| `contrastive_temperature` | float | `0.5` | Temperature parameter for contrastive scaling |
| `epochs` | int | `100` | Number of training epochs |
| `batch_size` | int | `500` | Batch size for training |
| `use_amp` | bool | `True` | Enable Automatic Mixed Precision |



## Results

### Fidelity and Utility Metrics

Comparison of ContraCTGAN variants against baseline CTGAN and TVAE on the Credit Card Fraud dataset.

* **Fidelity:** Measured by Wasserstein Distance (WSD), Jensen–Shannon Divergence (JSD), and L2 Pearson correlation distance.
* **Utility:** Measured by Accuracy, AUC, and F1 score of an XGBoost classifier trained on the synthetic data and tested on real data.

| Model | WSD ↓ | JSD ↓ | L2 Pearson ↓ | Accuracy ↑ | AUC ↑ | F1 ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| **Real Data** | – | – | – | 0.9996 | 0.9777 | 0.8990 |
| Baseline CTGAN | 167.66 | 0.19825 | 8.19410 | 0.9792 | 0.9762 | 0.1392 |
| Baseline TVAE | 240.79 | **0.16331** | NaN | **0.9982** | 0.5000 | 0.0000 |
| **ContraCTGAN (=0.5, =0.5)** | 109.73 | 0.19842 | **7.73692** | 0.9919 | 0.9673 | **0.2883** |
| ContraCTGAN (=0.5, =0.1) | 122.52 | 0.20566 | 8.18397 | 0.9894 | **0.9774** | 0.2418 |
| ContraCTGAN (=0.2, =1.0) | 187.77 | 0.20333 | 7.92103 | 0.9931 | 0.9808 | 0.3160 |
| ContraCTGAN (=0.8, =0.8) | **81.34** | 0.19956 | 8.56982 | 0.9884 | 0.9690 | 0.2259 |

**Key Findings:**

* **Utility Boost:** The highlighted ContraCTGAN configuration () more than doubles the F1 score compared to the baseline CTGAN (0.2883 vs 0.1392), indicating significantly better capture of the minority fraud class.
* **Fidelity:** ContraCTGAN generally reduces Wasserstein Distance and L2 Pearson distance, preserving marginal distributions and correlations better than the baseline.



## Citation

If you use this code in your research, please cite:

```bibtex
@misc{john2025contractgan,
  title  = {ContraCTGAN: Enhancing CTGAN’s Tabular Data Generation with Contrastive Loss Integration},
  author = {Amit John},
  year   = {2025},
  note   = {GitHub repository: https://github.com/johnamit/ContraCTGAN}
}

```



## License

* **Code:** [MIT License](https://www.google.com/search?q=LICENSE)
* **Dataset:** [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/)
