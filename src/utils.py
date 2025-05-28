"""
Utility functions for hurricane track processing and analysis.
"""

import numpy as np
import pandas as pd
from scipy import interpolate
from global_land_mask import globe
import collections as cc
import random
import torch
from visual import draw_track_with_wind

def calculate_scaling_factors(dataset, sample_rate):
    """
    Calculate scaling factors from dataset.
    
    Args:
        dataset (torch.utils.data.TensorDataset): Dataset containing track data
        sample_rate (int): Number of points per track
        
    Returns:
        tuple: (max_x, max_y, max_z) scaling factors
    """
    global max_x, max_y, max_z
    
    data = dataset.tensors[0].numpy()
    max_x = np.max(data[:, :sample_rate])
    max_y = np.max(data[:, sample_rate:2*sample_rate])
    max_z = np.max(data[:, 2*sample_rate:])
    
    return max_x, max_y, max_z

def scale_predictions(predictions, max_x, max_y, max_z):
    """
    Scale predictions using global scaling factors.
    
    Args:
        predictions (torch.Tensor): Model predictions (batch_size, sample_rate*3)
        max_x (float): Maximum x value for normalization
        max_y (float): Maximum y value for normalization
        max_z (float): Maximum z value for normalization
        
    Returns:
        torch.Tensor: Scaled predictions
    """
    # Split predictions into channels
    channels = 3
    sample_rate = predictions.shape[1] // channels
    scaled_predictions = predictions.clone()
    scaled_predictions[:, :sample_rate] *= max_x
    scaled_predictions[:, sample_rate:2*sample_rate] *= max_y
    scaled_predictions[:, 2*sample_rate:] *= max_z
    
    return scaled_predictions

def read_hurdat(path, select_bound=20, only_track=True):
    """
    Read and process hurricane track data from HURDAT2 format.
    
    Args:
        path (str): Path to HURDAT2 data file
        select_bound (int): Minimum number of points required for a track
        only_track (bool): Whether to return only track coordinates
        
    Returns:
        list: Processed hurricane tracks
    """
    tracks = []
    f = open(path)
    line = f.readline()
    
    def init_track():
        track_record = {
            'current_track': [],
            'current_Date': [],
            'current_YearMonth': [],
            'current_Month': [],
            'current_Record_identifier': [],
            'current_Strength': [],
            'current_Max_wind': [],
            'current_Min_pressure': [],
            'current_kt34_radius_ne': [],
            'current_kt34_radius_se': [],
            'current_kt34_radius_sw': [],
            'current_kt34_radius_nw': [],
            'current_kt50_radius_ne': [],
            'current_kt50_radius_se': [],
            'current_kt50_radius_sw': [],
            'current_kt50_radius_nw': [],
            'current_kt64_radius_ne': [],
            'current_kt64_radius_se': [],
            'current_kt64_radius_sw': [],
            'current_kt64_radius_nw': []
        }
        return track_record

    track_record = init_track()

    while line != '':
        words = line.split(',')
        if len(words) <= 4:
            if len(track_record['current_track']) > select_bound:
                tracks.append(track_record)
            track_record = init_track()
            line = f.readline()
            continue
        else:
            # Process track data
            track_record['current_track'].append((float(words[5][:-1]), float(words[4][:-1])))
            track_record['current_Date'].append(words[0])
            track_record['current_YearMonth'].append(words[0][:-2])
            track_record['current_Month'].append(words[0][-4:-2])
            track_record['current_Record_identifier'].append(words[2])
            track_record['current_Strength'].append(words[3])
            track_record['current_Max_wind'].append(float(words[6]))
            track_record['current_Min_pressure'].append(float(words[7]))
            
            # Process radius data
            for i, prefix in enumerate(['kt34', 'kt50', 'kt64']):
                for j, suffix in enumerate(['ne', 'se', 'sw', 'nw']):
                    key = f'current_{prefix}_radius_{suffix}'
                    track_record[key].append(float(words[8 + i*4 + j]))
            
            line = f.readline()

    if only_track:
        return [track['current_track'] for track in tracks]
    return tracks

def linear_interpolation(array, sample_rate):
    """
    Perform linear interpolation on array.
    
    Args:
        array (np.array): Input array
        sample_rate (int): Number of points to sample
        
    Returns:
        np.array: Interpolated array
    """
    len_track = len(array)
    x = range(len_track)
    f = interpolate.interp1d(x, array, fill_value="extrapolate")
    gap = len_track / sample_rate
    xnew = np.arange(0, len_track, gap)[:sample_rate]
    
    return np.array(f(xnew), dtype=float)

def track_standardization(track, sample_rate=100, ex_channels=['current_Min_pressure']):
    """
    Standardize track data using linear interpolation.
    
    Args:
        track (dict): Hurricane track data
        sample_rate (int): Number of points to sample
        ex_channels (list): Additional channels to process
        
    Returns:
        np.array: Standardized track data
    """
    track_position = track['current_track']
    track_position = np.transpose(track_position)
    x = track_position[0]
    y = track_position[1]
    
    x_new = linear_interpolation(x, sample_rate=sample_rate)
    y_new = linear_interpolation(y, sample_rate=sample_rate)
    m = np.array([x_new, y_new])

    for channel in ex_channels:
        c = track[channel]
        if min(c) < 0:
            return [-1]
        c = linear_interpolation(c, sample_rate=sample_rate)
        c = np.reshape(c, [1, sample_rate])
        m = np.r_[m, c]

    return np.array(np.transpose(m), dtype=float)

def create_land_sea_dataset(test_x, z, sample_rate, max_x, max_y):
    """
    Create dataset with land/sea information.
    
    Args:
        test_x (np.array): Test data
        z (np.array): Latent representation
        sample_rate (int): Number of points per track
        max_x (float): Maximum x value for normalization
        max_y (float): Maximum y value for normalization
        
    Returns:
        list: Dataset with land/sea information
    """
    data_line = []
    for line in test_x:
        x = line[:sample_rate]
        y = line[sample_rate:sample_rate*2]
        land_sea_line = [1 if globe.is_ocean(y[i]*max_y, x[i]*max_x) else 0 
                        for i in range(sample_rate)]
        data_line.extend(test_x.tolist()[0])
        data_line.extend(land_sea_line)
        data_line.extend(z.tolist()[0])
    return data_line

def track_standradization(track,sample_rate = 100, ex_channels = ['current_Min_pressure']):
    """
    Standardize a single hurricane track.
    
    Args:
        track (dict): Hurricane track data
        sample_rate (int): Number of points per track
        ex_channels (list): Additional channels to process
        
    Returns:
        np.array: Standardized track data
    """
    hurricane_without_min_pressure = []
    hurricane_with_min_pressure = []
    
    track_position = track['current_track']
    track_position = np.transpose(track_position)
    x = track_position[0]
    y = track_position[1]
    x_new = linear_interpolation(x , sample_rate=sample_rate)
    y_new = linear_interpolation(y, sample_rate=sample_rate)
    m = np.array([x_new, y_new])

    for channel in ex_channels:
        c = track[channel]
        if min(c)<0:
            hurricane_without_min_pressure.append([track['current_track'][0][0], track['current_track'][0][1], track['current_Date'][0], "False"])
            return [-1]
        else:
            hurricane_with_min_pressure.append([track['current_track'][0][0],track['current_track'][0][1],track['current_Date'][0],"True"])
        c = linear_interpolation(c, sample_rate=sample_rate)
        c = np.reshape(c, [1, sample_rate])
        m = np.r_[m,c]
    new_track = np.transpose(m)

    return np.array(new_track,dtype=float)

def tracks_batch_standardization(tracks,sample_rate = 100, ex_channels = ['current_Min_pressure']):
    """
    Standardize a batch of hurricane tracks.
    
    Args:
        tracks (list): List of hurricane track dictionaries
        sample_rate (int): Number of points per track
        ex_channels (list): Additional channels to process
        
    Returns:
        tuple: Standardized tracks and start times
    """
    output = []
    hur_start_time = []
    for track in tracks:
        new_track = track_standradization(track,sample_rate = sample_rate,ex_channels= ex_channels)
        
        if len(new_track) == 1:
            continue
        
        output.append(new_track)
        hur_start_time.append(track['current_Date'][0])
        
        if len(new_track) != sample_rate:
            raise ValueError(f"Track {track['current_Date'][0]} has length {len(new_track)} instead of {sample_rate}")

    return np.array(output,dtype=float), hur_start_time

def prediction_data_generate(output_data=1000):
    # sample by cluster
    ori_tracks = pd.read_csv('run/oriwind_cluster.csv', index_col=0)
    label = ori_tracks['Class'].to_numpy()
    ori_tracks = ori_tracks.drop(['Date','Month','Class'], axis=1)
    ori_tracks = ori_tracks.to_numpy()
    hurrNumAll = ori_tracks.shape[0]
    class_label = cc.defaultdict(list)
    class_count = cc.defaultdict(int)

    # for track in ori_tracks.iterrows():
    for ind, _ in enumerate(ori_tracks):
        class_label[label[ind]].append([ind])
        class_count[label[ind]] += 1

    orderedHurr = cc.OrderedDict(sorted(class_count.items(), key=lambda kv: (kv[1])))
    hurNumList = list(map(lambda kv: kv[1], orderedHurr.items()))
    hurNumInd = list(map(lambda kv: kv[0], orderedHurr.items()))

    generated_seeds_id = []
    # hurrRandomArea = random.randint(1, hurrNumAll)
    for n in range(output_data):
        hurrRandomArea = random.randint(1, hurrNumAll)
        sumNum = 0
        for i, e in enumerate(hurNumList):
            sumNum += e
            if hurrRandomArea <= sumNum:
                selected = hurrRandomArea - (sumNum - e) - 1
                break
        selected_info = class_label[hurNumInd[i]][selected]
        selected_id = selected_info[0]
        generated_seeds_id.append(selected_id)
    generated_seeds_data = ori_tracks[generated_seeds_id]
    generated_seeds_data = generated_seeds_data.astype(float)

    return generated_seeds_data

def output_prediction(model, epoch, opath, test_loader, device, max_x, max_y, max_z, save_figure=False, figure_name=None):
    """
    Save model predictions to file.
    
    Args:
        model (CVAE): Trained model
        epoch (int): Current epoch
        opath (str): Output path
        test_loader (torch.utils.data.DataLoader): Test data loader
        device (str): Device to use for computation
        max_x (float): Maximum x value for normalization
        max_y (float): Maximum y value for normalization
        max_z (float): Maximum z value for normalization
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tracks_predicted = get_predict_tracks(model, epoch, test_loader, predict_time=1, device=device, max_x=max_x, max_y=max_y, max_z=max_z)
    
    # Convert to PyTorch tensor and move to device
    tracks_predicted = torch.tensor(tracks_predicted, device=device)
    
    # Scale predictions
    tracks_predicted_new = scale_predictions(tracks_predicted[0], max_x, max_y, max_z)
    
    # Convert to numpy and save
    tracks_predicted_new = tracks_predicted_new.cpu().numpy()
    print(f"Shape of predicted tracks: {tracks_predicted_new.shape}")
    
    # Save to CSV
    tracks_predicted_df = pd.DataFrame(tracks_predicted_new)
    tracks_predicted_df.to_csv(f'{opath}/sampled_track_{model.sample_rate}_{epoch}.csv', index=False)
    
    if save_figure:
        draw_track_with_wind(f'{opath}/sampled_track_{model.sample_rate}_{epoch}.csv', save=True, index=0, sample_rate=model.sample_rate, figure_name=figure_name)

def get_predict_tracks(model, epoch, test_loader, predict_time, device, max_x, max_y, max_z):
    """
    Generate predicted tracks.
    
    Args:
        model (CVAE): Trained model
        epoch (int): Current epoch
        test_loader (torch.utils.data.DataLoader): Test data loader
        predict_time (int): Number of predictions per input
        device (str): Device to use for computation
        
    Returns:
        list: Generated tracks
    """
    model.eval()
    tracks_pred = []
    
    data = next(iter(test_loader))
    
    with torch.no_grad():
        for test_x in data:
            test_x = test_x.to(device)
            tracks_pred_part = []
            for _ in range(predict_time):
                z = model.encode(test_x)
                data_line = create_land_sea_dataset(test_x.cpu().numpy(), z.cpu().numpy(), model.sample_rate, max_x, max_y)
                predictions = model.sample(z)
                tracks_pred_part.append(predictions.cpu().numpy())
            tracks_pred.extend(tracks_pred_part)
 
    return tracks_pred