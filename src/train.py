"""
Training module for hurricane track generation model using PyTorch.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import read_hurdat, tracks_batch_standardization, prediction_data_generate, output_prediction
from visual import draw_track_with_wind

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class CVAE(nn.Module):
    """Convolutional Variational Autoencoder for hurricane track generation."""
    
    def __init__(self, latent_dim, sample_rate, channels=3):
        super(CVAE, self).__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        self.sample_rate = sample_rate
        self._build_encoders()
        self._build_decoders()
        self._build_mixer()
    
    def _build_encoders(self):
        """Build encoder networks for each channel."""
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(self.sample_rate),
                nn.Linear(self.sample_rate, self.sample_rate * 10),
                nn.LeakyReLU(),
                nn.Linear(self.sample_rate * 10, self.sample_rate * 8),
                nn.LeakyReLU(),
                nn.Linear(self.sample_rate * 8, self.sample_rate * 6),
                nn.LeakyReLU(),
                nn.BatchNorm1d(self.sample_rate * 6),
                nn.Linear(self.sample_rate * 6, self.latent_dim)
            ) for _ in range(self.channels)
        ])
    
    def _build_decoders(self):
        """Build decoder networks for each channel."""
        self.sub_decoders = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(self.latent_dim),
                nn.Linear(self.latent_dim, self.sample_rate * 10),
                nn.LeakyReLU(),
                nn.Linear(self.sample_rate * 10, self.sample_rate * 8),
                nn.LeakyReLU(),
                nn.Linear(self.sample_rate * 8, self.sample_rate * 6),
                nn.LeakyReLU(),
                nn.Linear(self.sample_rate * 6, self.sample_rate),
                nn.LeakyReLU()
            ) for _ in range(self.channels)
        ])
    
    def _build_mixer(self):
        """Build mixer network for combining latent representations."""
        self.mixer = nn.Sequential(
            nn.BatchNorm1d(self.latent_dim * 3),
            nn.Linear(self.latent_dim * 3, self.sample_rate * 141),
            nn.LeakyReLU(),
            nn.Linear(self.sample_rate * 141, self.sample_rate * 133),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.sample_rate * 133),
            nn.Linear(self.sample_rate * 133, self.sample_rate * 171),
            nn.LeakyReLU(),
            nn.Linear(self.sample_rate * 171, self.latent_dim * self.channels),
            nn.LeakyReLU()
        )

    def encode(self, x):
        """Encode input data into latent space."""
        inputs = torch.split(x, self.sample_rate, dim=1)
        codes = [encoder(input) for encoder, input in zip(self.encoders, inputs)]
        return torch.cat(codes, dim=1)

    def decode(self, z, apply_var=False):
        """Decode latent representation back to data space."""
        if not self.training:
            scale = 0.04
            rate = 1 + (-1)**torch.randint(0, 2, (1,)).item() * scale/100 * torch.randint(0, 101, (1,)).item()
            z = z * rate
        
        mixed_codes = torch.split(z, self.latent_dim, dim=1)
        decodes = [decoder(code) for decoder, code in zip(self.sub_decoders, mixed_codes)]
        return torch.cat(decodes, dim=1)

    def sample(self, latent_space):
        """Add noise to the latent space."""
        noise = torch.randn(latent_space.shape, device=latent_space.device)
        return self.decode(latent_space + noise, apply_var=True)

def compute_loss(model, x):
    """Compute reconstruction loss."""
    z = model.encode(x)
    x_logit = model.decode(z)
    return nn.MSELoss()(x, x_logit)

def train_step(model, x, optimizer):
    """Execute one training step."""
    optimizer.zero_grad()
    loss = compute_loss(model, x)
    loss.backward()
    optimizer.step()
    return loss.item()

def data_preparing(sample_rate=50, ex_channels=['current_Min_pressure'], output_data=0):
    """
    Prepare data for training or prediction.
    
    Args:
        sample_rate (int): Number of points per track
        ex_channels (list): Additional channels to process
        output_data (int): Number of output samples
        
    Returns:
        tuple: (train_dataset, test_dataset, sample_rate)
    """
    if output_data == 0:
        # Training mode
        tracks = read_hurdat('data/hurdat2-1851-2021-100522.txt', only_track=False)
        std_tracks, hur_start_time = tracks_batch_standardization(tracks, sample_rate=sample_rate, ex_channels=ex_channels)
        std_tracks_np = np.hstack((std_tracks[:, :, 0], std_tracks[:, :, 1], std_tracks[:, :, 2]))
        std_tracks_df = pd.DataFrame.from_dict(std_tracks_np)
        std_tracks_df['Date'] = hur_start_time
        std_tracks_df.to_csv('run/oriwind_tracks.csv')
        
        draw_track_with_wind(std_tracks, True)
    else:
        # Prediction mode
        tracks = prediction_data_generate(output_data=output_data)
        std_tracks = tracks.reshape((tracks.shape[0], 20, 3), order='F')

    # Normalize data
    max_x = np.max(std_tracks[:, :, 0])
    max_y = np.max(std_tracks[:, :, 1])
    max_z = np.max(std_tracks[:, :, 2])
    
    norm_x = std_tracks[:, :, 0]/max_x
    norm_y = std_tracks[:, :, 1]/max_y
    norm_strength = std_tracks[:, :, 2]/max_z
    flat_std_tracks = np.hstack((norm_x, norm_y, norm_strength))

    # Convert to PyTorch tensors
    flat_std_tracks = torch.FloatTensor(flat_std_tracks)
    
    # Create datasets
    dataset = torch.utils.data.TensorDataset(flat_std_tracks)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2]) if output_data == 0 else (dataset, dataset)
    
    return train_dataset, test_dataset, sample_rate, max_x, max_y, max_z

def train_model(sample_rate=20, epochs=100, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
    """
    Train the VAE model.
    
    Args:
        sample_rate (int): Number of points per track
        epochs (int): Number of training epochs
        device (str): Device to train on ('cuda' or 'cpu')
    """
    # Initialize model and optimizer
    model = CVAE(latent_dim=int(sample_rate * 3), sample_rate=sample_rate, channels=3).to(device)
    optimizer = optim.Adamax(model.parameters(), lr=1e-3)
    
    # Prepare data
    train_dataset, _, sample_rate, _, _, _ = data_preparing(sample_rate=sample_rate)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
    
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}')
        for train_x, in pbar:
            train_x = train_x.to(device)
            loss = train_step(model, train_x, optimizer)
            pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        # Save model at last epoch
        if epoch == epochs:
            torch.save(model.state_dict(), f"run/weights_{sample_rate}_{loss}.pt")
        print(f"Epoch {epoch} completed")

if __name__ == "__main__":
    train_model() 