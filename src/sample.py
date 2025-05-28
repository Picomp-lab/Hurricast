"""
Sampling and generation module for hurricane tracks using PyTorch.
"""

import torch
from train import CVAE, data_preparing
from utils import output_prediction

def generate_tracks(sample_rate=20, num_tracks=116, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Generate new hurricane tracks using trained model.
    
    Args:
        sample_rate (int): Number of points per track
        num_tracks (int): Number of tracks to generate
        device (str): Device to use for computation
    """
    # Load model
    model = CVAE(latent_dim=int(sample_rate * 3), sample_rate=sample_rate, channels=3).to(device)
    model.load_state_dict(torch.load(f"run/weights_20_0.007728016469627619.pt"))
    
    # Generate tracks
    epochs = 20
    for epoch in range(1, epochs + 1):
        # Prepare data and get scaling factors
        _, test_dataset, sample_rate, max_x, max_y, max_z = data_preparing(sample_rate=sample_rate, output_data=num_tracks)
        
        # # Calculate scaling factors
        # calculate_scaling_factors(test_dataset, sample_rate)
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=num_tracks, shuffle=True)
        output_prediction(model, epoch, 'run', test_loader, device, max_x, max_y, max_z, 
                          save_figure=True, figure_name='sampled_tracks')

if __name__ == "__main__":
    generate_tracks() 
    