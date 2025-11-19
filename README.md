# ContraCTGAN: Enhancing CTGAN's Tabular Data Generation with Contrastive Loss Integration

## 1. What's this project about?

**Short version**: I wanted to generate **realistic but privacy-preserving credit-card transactions** for fraud detection, without leaking sensitive information. CTGAN is great, but it can struggle to capture the subtle relationships between features that real fraud models rely on. So I applied a **SimCLR-style X-ent contrastive loss** to its discriminator and created **ContraCTGAN**.

### What was the motivation behind this?
- Real financial data is **sensitive, regulated and hard to share**.
- Fraud datasets are **extremely imbalanced** (very few fraud cases).
- Synthetic data lets us:
  - Prototype models and experiments without touching real customers.
  - Share datasets and reproduce work while staying on the right side of privacy.


### What ContraCTGAN actually changes

Starting point: [CTGAN](https://github.com/sdv-dev/CTGAN), which already gives you:

- Mode-specific normalisation for continuous columns.
- Conditional generator + training-by-sampling for imbalanced discrete columns.
- WGAN-GP + PacGAN to stabilise training and reduce mode collapse.

On top of that, ContraCTGAN introduces:
- An **Embedder** MLP that takes the discriminator’s tabular input (real data + condition vector) and maps it into a latent space.
- A **contrastive branch**:
  - For each real batch, we create two lightly noised copies  
    $x_{\text{aug1}} = x + \varepsilon_1,\; x_{\text{aug2}} = x + \varepsilon_2$ with Gaussian noise (std = 0.01). 
  - Feed both through the Embedder to get $z_i$ and $z_j$.
  - Optimise an **NT-Xent (SimCLR) loss** over positives (same row, two views) vs negatives (other rows).
- The discriminator now minimises

$$\mathcal{L}_D^{\text{total}} = \mathcal{L}_D^{\text{WGAN-GP}} + \lambda_{\text{contrastive}} \cdot \mathcal{L}_{\text{NT-Xent}}$$

while the generator still uses the $-\mathbb{E}[D(x_{\text{fake}})]$ objective.   

The hope: a better-trained critic → sharper feedback → synthetic data that **preserves marginal distributions *and* inter-feature dependencies** more faithfully.


## 2. Getting set up (prereqs & install)
### Environment & requirements
Everything you need is in `requirements.txt`, including:
- PyTorch
- CTGAN / SDV stack
- SciPy, scikit-learn, XGBoost, etc.

**Cloning the environment**
```bash
git clone https://github.com/<your-username>/ContraCTGAN.git
cd ContraCTGAN
```
**Installing the requirements**
It's recommended to create an environment before installing the requirements to avoid up/downgrading dependencies your other projects may rely on. I used miniconda.
```bash
pip install -r requirements.txt
```

### What i trained on (for reference)
The main experiments in the paper were run on:
- CPU: AMD Ryzen 7 7800X3D
- GPU: NVIDIA RTX 3090 (24G, CUDA)
- OS: Ubuntu 24.04.2
- Python: 3.12.9
- PytorchL 2.6.0 + cu124

Mixed-precision training (use_amp=True) is supported and enabled in the training scripts to speed things up and save VRAM.

## 3. What dataset was used?
This project uses the classic [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset consisting of **284,807** transactions from European cardholders in Sept 2013, of which **492** are labelled as fraud (severe class imbalance).

Features:
- 28 PCA-transformed features V1–V28
- Time (seconds since first transaction)
- Amount (transaction value in euros)
- Class (0 = legit, 1 = fraud)

### Where is the data located
The repo assumes that you have a train/test split saved as:
```text
Datasets/
├── creditcard_train.csv
└── creditcard_test.csv
```
The split datasets can be downloaded from [here](https://drive.google.com/drive/folders/1aHzWk2kSIdyNrQwaKhpYbTWOrBAYJQGf?usp=sharing)

The train/test split itself is done outside the training script using stratified sampling (80/20) to preserve the fraud/legit ratio.

### Pre-processing
Inside ContrastiveCTGAN.fit() the following happens:

**Null check** 
- CTGAN does not support nulls in continuous columns, so the script explicitly checks and raises if any are found.

**DataTransformer (from CTGAN)**
- Fits VGM-based mode-specific normalisation to continuous columns.
- One-hot encodes discrete columns (here just `Class`).
- Produces a concatenated representation of all columns.

**DataSampler** 
- Handles conditional vectors and training-by-sampling over discrete columns to fight class imbalance.

## 4. Repo tour (where things live & how to run them)
```text
ContraCTGAN/
├── Datasets/                        # Real data (not tracked in git)
│   └── README.md                    # Notes on how to obtain / mount
│
├── Models/                          # Trained models (.pth) – via Git LFS
│   ├── BaselineCTGAN.pth
│   └── ContrastiveCTGAN_*.pth
│
├── Notebooks/
│   └── synthetic_data_evaluation.ipynb   # Fidelity + utility evaluation
│
├── Plots/                           # Final small figures for README/paper
│   └── loss_curves_ContrastiveCTGAN_lambda0_1_temp0_1.png  # example
│
├── Scripts/
│   ├── contrastive_ctgan.py         # Main training script (single run)
│   └── contrastive_ctganF.py        # Sweeps over λ / τ variants
│
├── SyntheticDatasets/
│   ├── Eval/                        # Synthetic CSVs used in the notebook
│   └── synthetic_data_*.csv         # Outputs from different runs
│
├── requirements.txt
├── README.md
└── LICENSE

```
### Storage philosophy
- GitHub repo: scripts, notebooks, configs, small plots, README, paper.
- GitHub LFS: The trained models `Models/*.pth`
- Google Drive: [the dataset](https://drive.google.com/drive/folders/1aHzWk2kSIdyNrQwaKhpYbTWOrBAYJQGf?usp=sharing), [generated synthetic datasets](https://drive.google.com/drive/folders/1XEd2CICzLWKg4YKTReMOhIrYvInCY3zx?usp=sharing), [training plots](https://drive.google.com/drive/folders/1AF4UA2gTr5_ZnEeT-E60uv0DHDzALL65?usp=sharing)

### How to train ContraCTGAN
The simplest end-to-end run:
```bash
python Scripts/contrastive_ctganF.py
```

This will:
1. Load Datasets/creditcard_train.csv.
2. Run the CTGAN + contrastive training loop for the configured epochs (default: 100).
3. Log generator, discriminator and contrastive losses into self.loss_values.
4. Save the trained model to `Models/*.pth`, synthetic samples to `SyntheticDatasets/*.csv`, loss plot to `Plots/*.png`

The script names files like:
```text
Models/ContrastiveCTGAN_lambda0_1_temp0_1.pth
SyntheticDatasets/synthetic_data_ContrastiveCTGAN_lambda0_1_temp0_1.csv
Plots/loss_curves_ContrastiveCTGAN_lambda0_1_temp0_1.png
```
and you can tweak `contrastive_lambda`($\lambda$), `contrastive_temperature`($\tau$), `epochs`, `batch_size`, `use_amp`, etc. via the constructor arguements inside the script.

## 5. Results and evaluation
### Metrics
This project used metric to evaluate both the fidelity and utility of the generated synthetic data.

**Fidelity (do the synthetics look like real data?)**
- Wasserstein Distance (WD) – average WD over all numeric features.
- Jensen–Shannon Divergence (JSD) – histogram-based divergence for numeric features.
- L2 distance between Pearson correlation matrices – measures how well inter-feature correlations are preserved.

**Utility (are synthetics useful for downstream fraud models?)**
- Train an XGBoost classifier on synthetic data.
- Test on the real hold-out set.
- Report the Accuracy, AUC (Area Under ROC) and F1 score (harmonic mean of precision & recall).

All of this is implemented in `Notebooks/synthetic_data_evaluation.ipynb`.

### Main results (from the paper)
Below is the table of results we observed and discussed in the paper.

| Model                           | WSD ↓  | JSD ↓   | L2 Pearson ↓ | Accuracy ↑ | AUC ↑  | F1 ↑   |
| ------------------------------- | ------ | ------- | ------------ | ---------- | ------ | ------ |
| **Real Data**                   | –      | –       | –            | 0.9996     | 0.9777 | 0.8990 |
| **Baseline CTGAN**              | 167.66 | 0.19825 | 8.19410      | 0.9792     | 0.9762 | 0.1392 |
| **Baseline TVAE**               | 240.79 | 0.16331 | NaN          | 0.9982     | 0.5000 | 0.0000 |
| **ContraCTGAN (λ=0.5, τ=0.5)**  | 109.73 | 0.19842 | 7.73692      | 0.9919     | 0.9673 | 0.2883 |
| **ContraCTGAN (λ=0.5, τ=0.1)**  | 122.52 | 0.20566 | 8.18397      | 0.9894     | 0.9774 | 0.2418 |
| **ContraCTGAN (λ=0.2, τ=1.0)**  | 187.77 | 0.20333 | 7.92103      | 0.9931     | 0.9808 | 0.3160 |
| **ContraCTGAN (λ=0.8, τ=0.8)**  | 81.34  | 0.19956 | 8.56982      | 0.9884     | 0.9690 | 0.2259 |
| **ContraCTGAN (λ=1.0, τ=0.07)** | 96.35  | 0.19801 | 7.97287      | 0.9870     | 0.9773 | 0.2019 |

### A brief interpretation: 
- TVAE looks great on accuracy but completely collapses on AUC and F1, so it’s basically predicting the majority class only.
- Baseline CTGAN does OK, but its F1 score (0.1392) shows it struggles with the minority fraud class.
- ContraCTGAN variants:
    - In general, they reduce WD and/or L2 Pearson vs CTGAN, which means better marginal distributions and correlations.
    - They double (or more) the F1 score in several configurations, making classifiers trained on synthetic data more useful.
    - The $\lambda$=0.5, $\tau$=0.5 variant is highlighted in the paper as the best overall trade-off between fidelity and utility, It's not the absolute best at any single metric, but the best balance observed.
 
## 6. Acknowledgements and References
Some key ingredients that made this project possible:

### Dataset
- Machine Learning Group – ULB, Credit Card Fraud Detection (Kaggle dataset).

### Publications
- Ting Chen et al., [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709).
- Lei Xu et al., [Modeling Tabular Data using Conditional GAN](https://arxiv.org/abs/1907.00503).
- Zilong Zhao et al., [CTAB-GAN: Effective Table Data Synthesizing](https://arxiv.org/abs/2102.08369).
- Mehdi Mirza et & Simon Osindero, [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784).

Further publications used for this project are included in the reference pages of the paper

### Base models
- [SDV-dev CTGAN implementation](https://github.com/sdv-dev/CTGAN)
- [TVAE implementation](https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/tvaesynthesizer) from the SDV ecosystem.

### This repo/paper
If you use ContraCTGAN in your own work, a quick citation is always appreciated:
```bibtex
@misc{john2025contractgan,
  title  = {ContraCTGAN: Enhancing CTGAN’s Tabular Data Generation with Contrastive Loss Integration},
  author = {Amit John},
  year   = {2025},
  note   = {GitHub repository: https://github.com/johnamit/ContraCTGAN}
}
```

## 8. Licences
**Code License**: All code in this repo is released under the MIT License. See `LICENSE` for details.
**Dataset License**: The original Credit Card Fraud Detection dataset is released under the Open Database License (ODbL) 1.0 by the Machine Learning Group – ULB (via Kaggle).
