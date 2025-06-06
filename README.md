# Hurricast: Synthetic Tropical Cyclone Track Generation for Hurricane Forecasting

### Overview
Hurricast is a deep learning model for generating synthetic hurricane tracks using a Conditional Variational Autoencoder (CVAE) architecture. The model is implemented in PyTorch and can generate realistic hurricane trajectories based on historical data.

### Paper,Poster and Slides
- [Read the paper](https://ui.adsabs.harvard.edu/abs/2023arXiv230907174G/abstract)
- 📄 [View Poster](https://github.com/Picomp-lab/Hurricast/blob/main/HurriCast%20Poster.pdf)
- 📊 [View Slides](https://github.com/Picomp-lab/Hurricast/blob/main/HurriCast%20Slides.pdf)


### Requirements
- Python 3.8+
- PyTorch
- NumPy
- Pandas
- tqdm

### Installation
```bash
# Clone the repository
git clone [repository-url]

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation
```
data/
  └── hurdat2-1851-2021-100522.txt
```

### Usage

#### Training the Model
```python
from train import train_model

# Train the model with default parameters
train_model(sample_rate=20, epochs=100)
```

#### Generating Tracks
```python
from sample import generate_tracks

# Generate 116 tracks with 20 points each
generate_tracks(sample_rate=20, num_tracks=116)
```

### Model Architecture
The model uses a CVAE architecture with:
- Separate encoders for each channel (latitude, longitude, wind speed)
- Decoders for reconstructing the track data

### Output
Generated tracks are saved in the `run` directory:
- `oriwind_tracks.csv`: Original tracks with wind data
- `sampled_tracks.png`: Visualization of generated tracks
- Model weights are saved as `weights_[sample_rate]_[loss].pt`
