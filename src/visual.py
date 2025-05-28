"""
Visualization functions for hurricane tracks.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

def read_track_csv(path, sample_rate=20, channels=3):
    """
    Read track data from CSV file.
    
    Args:
        path (str): Path to CSV file
        sample_rate (int): Number of points per track
        channels (int): Number of channels in data
        
    Returns:
        list: Processed track data
    """
    tracks = np.array(pd.read_csv(path), dtype=float)
    tracks = tracks.reshape(-1, sample_rate, channels)
    return tracks
    output = []
    for track in tracks:
        data = []
        for c in range(channels):
            current_data = track[sample_rate*c:sample_rate*(c+1)]
            if len(data) == 0:
                data = current_data
            else:
                data = np.c_[data, current_data]
        output.append(data)
    return output

def draw_track_with_wind(path, save=False, index=0, sample_rate=20, figure_name=None):
    """
    Draw hurricane tracks with wind speed coloring.
    
    Args:
        path (str): Path to track data
        save (bool): Whether to save the plot
        index (int): Index for saving multiple plots
        sample_rate (int): Number of points per track
    """
    tracks = read_track_csv(path, sample_rate, 3) if isinstance(path, str) else path
    w = np.array(tracks)
    w = w[:,:,2]
    wind_max = np.max(w)
    wind_min = np.min(w)
    if wind_min < 0:
        wind_min = 0
        
    norm = plt.Normalize(vmax=5, vmin=0)
    fig = plt.figure()
    axs = fig.add_subplot()
    
    for t in tracks:
        x = t[:,0]
        y = t[:,1]
        p = t[:,2]
        dydx = p.copy()
        
        # Categorize pressure levels
        dydx[np.where(p>1000)] = 0
        dydx[np.where((p > 979) & (p <= 1000))] = 1
        dydx[np.where((p > 965) & (p <= 979))] = 2
        dydx[np.where((p > 945) & (p <= 965))] = 3
        dydx[np.where((p > 920) & (p <= 945))] = 4
        dydx[np.where(p <= 920)] = 5

        if min(dydx) < 0:
            continue
            
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap='rainbow', norm=norm)
        lc.set_array(dydx)
        lc.set_linewidth(2)
        axs.add_collection(lc)

    # Add legend
    la = ['>1000', '980-1000', '966-979', '946-965', '921-945', '<921']
    for color in range(6):
        plt.plot([], [], c=plt.get_cmap('rainbow', 6)(color), label=la[color])
    plt.legend(frameon=False)

    # Set plot properties
    plt.xlabel('Longitude', size=14)
    plt.ylabel('Latitude', size=14)
    plt.tick_params(labelsize=14)
    plt.grid(linestyle='--')
    
    x_ticks = [20, 40, 60, 80, 100, 120]
    x_ts = [f'{x}$^\circ$' for x in x_ticks]
    y_ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    y_ts = [f'{x}$^\circ$' for x in y_ticks]
    plt.xticks(x_ticks, x_ts)
    plt.yticks(y_ticks, y_ts)
    
    axs.set_xlim(0, 120)
    axs.set_ylim(0, 80)
    
    if save:
        figure_name = f'tracks.pdf' if figure_name is None else figure_name
        plt.savefig(f'run/{figure_name}.pdf')
    else:
        plt.show()
        plt.close()

def draw_heatmap(x, y, bins=100):
    """
    Draw heatmap of track density.
    
    Args:
        x (list): Longitude values
        y (list): Latitude values
        bins (int): Number of bins for histogram
    """
    plt.hist2d(np.multiply(x,1), y, bins=bins, cmap='Blues')
    
    plt.xlabel('Longitude', size=14)
    plt.ylabel('Latitude', size=14)
    plt.tick_params(labelsize=14)
    
    x_ticks = [20, 40, 60, 80]
    x_ts = [f'{x}$^\circ$' for x in x_ticks]
    y_ticks = [10, 20, 30, 40, 50, 60]
    y_ts = [f'{x}$^\circ$' for x in y_ticks]
    plt.xticks(x_ticks, x_ts)
    plt.yticks(y_ticks, y_ts)
    
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.show()

def draw_clustered_tracks(cluster_paths, sample_rate=20):
    """
    Draw tracks colored by cluster.
    
    Args:
        cluster_paths (list): List of paths to cluster data
        sample_rate (int): Number of points per track
    """
    norm = plt.Normalize(vmax=5, vmin=0)
    fig = plt.figure()
    axs = fig.add_subplot()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    
    for i, path in enumerate(cluster_paths):
        tracks = read_track_csv(path, sample_rate, 3)
        first_track = True
        
        for t in tracks:
            x = t[:,0]
            y = t[:,1]
            if first_track:
                plt.plot(x, y, color=colors[i%8], label=f'cluster{i+1}')
                first_track = False
            else:
                plt.plot(x, y, color=colors[i%8])

    plt.xlabel('Longitude', size=14)
    plt.ylabel('Latitude', size=14)
    plt.tick_params(labelsize=14)
    plt.legend()
    plt.grid(linestyle='--')
    
    x_ticks = [20, 40, 60, 80, 100, 120]
    x_ts = [f'{x}$^\circ$' for x in x_ticks]
    y_ticks = [0, 10, 20, 30, 40, 50, 60]
    y_ts = [f'{x}$^\circ$' for x in y_ticks]
    plt.xticks(x_ticks, x_ts)
    plt.yticks(y_ticks, y_ts)
    
    axs.set_xlim(20, 120)
    axs.set_ylim(0, 60)
    plt.show()

def draw_track_with_radius(path, save=False, opath='', index=0):
    """
    Draw tracks with radius visualization.
    
    Args:
        path (str): Path to track data
        save (bool): Whether to save the plot
        opath (str): Output path for saving
        index (int): Index for saving multiple plots
    """
    tracks = read_track_csv(path)
    w = np.array(tracks)
    w = w[:,:,2]
    wind_max = np.max(w)
    wind_min = np.min(w)
    if wind_min < 0:
        wind_min = 0
        
    norm = plt.Normalize(vmax=wind_max, vmin=wind_min)
    fig = plt.figure()
    axs = fig.add_subplot()
    
    for t in tracks:
        x = t[:,0]
        y = t[:,1]
        dydx = t[:,2]
        if min(dydx) < 0:
            continue
        axs.scatter(x, y, s=dydx, alpha=0.5)

    axs.set_xlim(0, 120)
    axs.set_ylim(0, 80)
    
    if save:
        plt.savefig(f'{opath}/{index}.png')
    else:
        plt.show()
    plt.close() 
    