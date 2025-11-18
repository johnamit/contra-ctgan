import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm
import os
import warnings

# PyTorch and mixed precision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast, GradScaler

# CTGAN components
from ctgan import CTGAN
from ctgan.synthesizers.base import BaseSynthesizer
from ctgan.synthesizers.ctgan import Discriminator, Generator
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer

def nt_xent_loss(z_i, z_j, temperature=0.5): # NT-Xent loss for contrastive learning
    batch_size = z_i.size(0)
    z_i = nn.functional.normalize(z_i, dim=1)
    z_j = nn.functional.normalize(z_j, dim=1)
    z = torch.cat([z_i, z_j], dim=0)
    
    # Compute cosine similarity matrix in the current precision (fp16)
    sim_matrix = torch.matmul(z, z.T) / temperature
    
    # Temporarily cast to float32 for masking
    sim_matrix = sim_matrix.float()
    
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim_matrix.masked_fill_(mask, -1e9)
    
    # Now, use the sim_matrix (in float32) for cross entropy loss calculation
    pos_indices = torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]).to(z.device)
    loss = nn.CrossEntropyLoss()(sim_matrix, pos_indices)
    return loss

class Embedder(nn.Module): # embedder to project discriminator inputs into an embedding space
    def __init__(self, input_dim, embed_dim):
        super(Embedder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def augment_data(x, noise_std=0.01): # for augmenting the real data to create a positive pair
    noise = torch.randn_like(x) * noise_std
    return x + noise

class ContrastiveCTGAN(CTGAN, BaseSynthesizer):
    def __init__(self, 
                 embedding_dim=128,             # Noise dimension
                 generator_dim=(256, 256),
                 discriminator_dim=(256, 256),
                 generator_lr=2e-4,
                 generator_decay=1e-6,
                 discriminator_lr=2e-4,
                 discriminator_decay=1e-6,
                 batch_size=500,
                 discriminator_steps=1,
                 log_frequency=True,
                 verbose=False,
                 epochs=1000,
                 pac=10,
                 random_state=42,
                 device='cuda',
                 use_amp=True,                  # Mixed precision training
                 contrastive_lambda=0.1,        # Weight for contrastive loss term
                 contrastive_temperature=0.5,   # Temperature for NT-Xent loss
                 embed_dim=128                 # Embedding dimension for contrastive learning
                ):
        self._device = device if torch.cuda.is_available() else 'cpu'
        print(f"Model will use device: {self._device}")
        self._device = torch.device(self._device)
        self.set_random_state(random_state)
        self.use_amp = use_amp
        self.contrastive_lambda = contrastive_lambda
        self.contrastive_temperature = contrastive_temperature
        
        super(ContrastiveCTGAN, self).__init__(
            embedding_dim=embedding_dim,
            generator_dim=generator_dim,
            discriminator_dim=discriminator_dim,
            generator_lr=generator_lr,
            generator_decay=generator_decay,
            discriminator_lr=discriminator_lr,
            discriminator_decay=discriminator_decay,
            batch_size=batch_size,
            discriminator_steps=discriminator_steps,
            log_frequency=log_frequency,
            verbose=verbose,
            epochs=epochs,
            pac=pac,
            cuda=(device=='cuda')
        )
        
        print(f"Epochs set to: {self._epochs}")
        print(f"Batch size set to: {self._batch_size}")
        print(f"Using mixed precision training: {self.use_amp}")
        
        # The embedder will be initialized after data transformation, when we know the discriminator input dimension.
        self._embedder = None

    def _generate_noise(self): # from CTGAN - noise for the latent input z passed to the generator
        return torch.randn(self._batch_size, self._embedding_dim, device=self._device)
    
    def _validate_null_data(self, train_data, discrete_columns): # from CTGAN - explicity added since it could not recognize the function from importing CTGAN. Checks for any null values in continuous data columns and raises an error if found.
        if isinstance(train_data, pd.DataFrame):
            continuous_cols = list(set(train_data.columns) - set(discrete_columns))
            any_nulls = train_data[continuous_cols].isna().any().any()
        else:
            continuous_cols = [i for i in range(train_data.shape[1]) if i not in discrete_columns]
            any_nulls = pd.DataFrame(train_data)[continuous_cols].isna().any().any()

        if any_nulls:
            raise ValueError(
                'CTGAN does not support null values in the continuous training data. '
                'Please remove all null values from your continuous training data.'
            )   
    
    def fit(self, train_data, discrete_columns):
        # Validate and transform the data as in the original CTGAN.
        self._validate_discrete_columns(train_data, discrete_columns)
        self._validate_null_data(train_data, discrete_columns)
        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)
        train_data_transformed = self._transformer.transform(train_data)
        self._data_sampler = DataSampler(train_data_transformed, self._transformer.output_info_list, self._log_frequency)
        data_dim = self._transformer.output_dimensions
        
        print(f"Data dimensions: {data_dim}")
        print(f"Condition vector dimension: {self._data_sampler.dim_cond_vec()}")
        print(f"Total input dimension for generator: {self._embedding_dim + self._data_sampler.dim_cond_vec()}")
        
        # Initialize generator and discriminator.
        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim
        ).to(self._device)
        
        self._discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac
        ).to(self._device)
        
        # Initialize the embedder for contrastive learning.
        disc_input_dim = data_dim + self._data_sampler.dim_cond_vec()
        self._embedder = Embedder(disc_input_dim, embed_dim=self._embedding_dim).to(self._device)
        
        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )
        optimizerD = optim.Adam(
            list(self._discriminator.parameters()) + list(self._embedder.parameters()),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )
        
        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Discriminator Loss', 'Contrastive Loss'])
        scaler = GradScaler('cuda') if self.use_amp else None
        
        epoch_iterator = tqdm(range(self._epochs), disable=(not self._verbose))
        if self._verbose:
            epoch_iterator.set_description('Epoch 0')
        
        steps_per_epoch = max(len(train_data_transformed) // self._batch_size, 1)
        
        for epoch in epoch_iterator:
            torch.cuda.empty_cache()
            for _ in range(steps_per_epoch):
                # --- Train Discriminator (with Contrastive Loss) ---
                for _ in range(self._discriminator_steps):
                    # Generate noise and sample condition vector
                    noise = self._generate_noise()
                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        real = self._data_sampler.sample_data(train_data_transformed, self._batch_size, None, None)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        noise = torch.cat([noise, c1], dim=1)
                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(train_data_transformed, self._batch_size, col[perm], opt[perm])
                    
                    if isinstance(real, np.ndarray):
                        real = torch.from_numpy(real.astype('float32'))
                    real = real.to(self._device)
                    
                    # Create the input for discriminator (concatenate condition vector if exists)
                    if condvec is not None:
                        disc_input_real = torch.cat([real, c1], dim=1)
                    else:
                        disc_input_real = real
                    
                    if self.use_amp:
                        with autocast('cuda'):
                            fake = self._generator(noise)
                            fake_act = self._apply_activate(fake)
                            if condvec is not None:
                                disc_input_fake = torch.cat([fake_act, c1], dim=1)
                            else:
                                disc_input_fake = fake_act
                            y_fake = self._discriminator(disc_input_fake)
                            y_real = self._discriminator(disc_input_real)
                            pen = self._discriminator.calc_gradient_penalty(disc_input_real, disc_input_fake, self._device, self.pac)
                            loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                            
                            # --- Contrastive Loss on Real Data ---
                            # Create an augmented view of real data.
                            real_aug1 = augment_data(disc_input_real)
                            real_aug2 = augment_data(disc_input_real)
                            
                            emb_real_aug1 = self._embedder(real_aug1)
                            emb_real_aug2 = self._embedder(real_aug2)
                            loss_contrastive = nt_xent_loss(emb_real_aug1, emb_real_aug2, temperature=self.contrastive_temperature)
                            
                            total_loss_d = loss_d + self.contrastive_lambda * loss_contrastive
                        optimizerD.zero_grad(set_to_none=True)
                        scaler.scale(pen).backward(retain_graph=True)
                        scaler.scale(total_loss_d).backward()
                        scaler.step(optimizerD)
                        scaler.update()
                    else:
                        fake = self._generator(noise)
                        fake_act = self._apply_activate(fake)
                        if condvec is not None:
                            disc_input_fake = torch.cat([fake_act, c1], dim=1)
                        else:
                            disc_input_fake = fake_act
                        y_fake = self._discriminator(disc_input_fake)
                        y_real = self._discriminator(disc_input_real)
                        pen = self._discriminator.calc_gradient_penalty(disc_input_real, disc_input_fake, self._device, self.pac)
                        loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                        
                        real_aug1 = augment_data(disc_input_real)
                        real_aug2 = augment_data(disc_input_real)
                        
                        emb_real_aug1 = self._embedder(real_aug1)
                        emb_real_aug2 = self._embedder(real_aug2)
                        loss_contrastive = nt_xent_loss(emb_real_aug1, emb_real_aug2, temperature=self.contrastive_temperature)
                        
                        total_loss_d = loss_d + self.contrastive_lambda * loss_contrastive
                        optimizerD.zero_grad(set_to_none=True)
                        pen.backward(retain_graph=True)
                        total_loss_d.backward()
                        optimizerD.step()
                
                # --- Train Generator ---
                noise = self._generate_noise()
                condvec = self._data_sampler.sample_condvec(self._batch_size)
                if condvec is not None:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    noise = torch.cat([noise, c1], dim=1)
                if self.use_amp:
                    with autocast('cuda'):
                        fake = self._generator(noise)
                        fake_act = self._apply_activate(fake)
                        if condvec is not None:
                            y_fake = self._discriminator(torch.cat([fake_act, c1], dim=1))
                        else:
                            y_fake = self._discriminator(fake_act)
                        loss_g = -torch.mean(y_fake)
                    optimizerG.zero_grad(set_to_none=True)
                    scaler.scale(loss_g).backward()
                    scaler.step(optimizerG)
                    scaler.update()
                else:
                    fake = self._generator(noise)
                    fake_act = self._apply_activate(fake)
                    if condvec is not None:
                        y_fake = self._discriminator(torch.cat([fake_act, c1], dim=1))
                    else:
                        y_fake = self._discriminator(fake_act)
                    loss_g = -torch.mean(y_fake)
                    optimizerG.zero_grad(set_to_none=True)
                    loss_g.backward()
                    optimizerG.step()
                
                # Record losses for later visualization.
                total_loss_d_val = total_loss_d.detach().cpu().item()
                loss_g_val = loss_g.detach().cpu().item()
                loss_contrastive_val = loss_contrastive.detach().cpu().item()
                epoch_loss_df = pd.DataFrame({
                    'Epoch': [epoch],
                    'Generator Loss': [loss_g_val],
                    'Discriminator Loss': [loss_d.detach().cpu().item()],
                    'Contrastive Loss': [loss_contrastive_val],
                })
                if not self.loss_values.empty:
                    self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(drop=True)
                else:
                    self.loss_values = epoch_loss_df
            print(f"Epoch {epoch+1}/{self._epochs}: Gen Loss = {loss_g_val:.4f}, Disc Loss = {loss_d.detach().cpu().item():.4f}, Contrastive Loss = {loss_contrastive_val:.4f}")
            if self._verbose:
                epoch_iterator.set_description(f"Epoch {epoch+1}: Gen {loss_g_val:.2f} | Disc {loss_d.detach().cpu().item():.2f}")
        return self.loss_values
    
def main():
    # Set seeds for reproducibility.
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Ensure required directories exist.
    os.makedirs('Models', exist_ok=True)
    os.makedirs('SyntheticDatasets', exist_ok=True)
    os.makedirs('Datasets', exist_ok=True)
    
    # GPU checks.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
        except Exception as e:
            print(f"Error getting GPU info: {e}")
            device = 'cpu'
    
    # Load and validate data.
    print("PROCESS: Loading the data...")
    data_path = 'Datasets/creditcard_train.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    real_data_df = pd.read_csv(data_path)
    if real_data_df.empty:
        raise ValueError("Loaded DataFrame is empty")
    
    # Specify discrete columns.
    discrete_columns = ["Class"]
    
    # Initialize the ContrastiveCTGAN model.
    try:
        contrastive_ctgan = ContrastiveCTGAN(
            epochs=100,             # Adjust epochs for debugging or production
            batch_size=500,
            device=device,
            random_state=42,
            use_amp=True,
            contrastive_lambda=0.5,
            contrastive_temperature=0.1,
            embedding_dim=128,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model: {e}")
    
    # Train the model.
    print("\nPROCESS: Starting model training...")
    start_time = time.time()
    try:
        loss_df = contrastive_ctgan.fit(real_data_df, discrete_columns)
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")
    training_time = time.time() - start_time
    print(f"Training complete in {training_time:.2f} seconds.")
    
    # Save the trained model.
    print("PROCESS: Saving the model...")
    model_path = "Models/ContrastiveCTGAN_full_hpt_corrected.pth"
    torch.save(contrastive_ctgan, model_path)
    print(f"Model saved at {model_path}")
    
    # Generate synthetic data.
    print("PROCESS: Generating synthetic data...")
    with torch.inference_mode():
        synthetic_data = contrastive_ctgan.sample(5000)
    
    # Save synthetic data.
    print("PROCESS: Saving synthetic data...")
    synthetic_data_path = 'SyntheticDatasets/synthetic_data_contrastive_ctgan_full_hpt_corrected.csv'
    pd.DataFrame(synthetic_data).to_csv(synthetic_data_path, index=False)
    print(f"Synthetic data saved at {synthetic_data_path}")
    
    # Plot loss curves.
    print("PROCESS: Plotting the loss curves...")
    plt.figure(figsize=(10, 6))
    plt.plot(loss_df['Epoch'], loss_df['Generator Loss'], label='Generator Loss')
    plt.plot(loss_df['Epoch'], loss_df['Discriminator Loss'], label='Discriminator Loss')
    plt.plot(loss_df['Epoch'], loss_df['Contrastive Loss'], label='Contrastive Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves (Contrastive CTGAN)')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curves_contrastive_ctgan_full_hpt_corrected.png')
    plt.show()
    
    print("PROCESS: Done!")
    
if __name__ == "__main__":
    main()