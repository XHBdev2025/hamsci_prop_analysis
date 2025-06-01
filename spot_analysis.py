#!/usr/bin/env python3

"""
- Loads Madrigal data via HDF5PolarsLoader
-

Author: [Diego F. Sanchez]
"""

import os
from datetime import date
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np


# Internal modules
from scripts.regions import REGIONS
from scripts.df_gen import HDF5PolarsLoader
from scripts.utils_plot import *

def plot_freq_snr_heatmap(df, save_path="freq_snr_heatmap.png"):
    fig, ax = plt.subplots(figsize=(8, 6))

    freq = np.array(df['freq'])
    snr = np.array(df['snr'])

    h = ax.hist2d(freq, snr, bins=100, cmap='viridis')
    cbar = plt.colorbar(h[3], ax=ax)
    cbar.set_label('Count')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('SNR (dB)')
    ax.set_title('Frequency vs SNR Heatmap')

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_paths_on_us_map(df, save_path="snr_paths_us_map.png"):
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([-130, -65, 24, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.STATES, edgecolor='gray')

    tx_lat = np.array(df['tx_lat'])
    tx_lon = np.array(df['tx_long'])
    rx_lat = np.array(df['rx_lat'])
    rx_lon = np.array(df['rx_long'])
    snr    = np.array(df['snr'])

    norm = plt.Normalize(vmin=np.nanmin(snr), vmax=np.nanmax(snr))
    cmap = plt.cm.viridis

    for i in range(len(tx_lat)):
        ax.plot(
            [tx_lon[i], rx_lon[i]],
            [tx_lat[i], rx_lat[i]],
            color=cmap(norm(snr[i])),
            linewidth=0.4,         # thinner lines
            alpha=0.4,             # more transparent
            transform=ccrs.PlateCarree()
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='SNR (dB)')

    ax.set_title('TX → RX Paths Colored by SNR')

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

if __name__ == "__main__":

    ########## Parameters ##########

    # Region to analyze (from REGIONS dictionary)
    region_name = 'New York - California'
    region = REGIONS[region_name]

    # Altitude in kilometers (0 km = ground level)
    altitude = 0

    # Date for which to generate the plots
    sDate = datetime(2017, 7, 1, 0, 0, 0)
    eDate = datetime(2017, 7, 1, 23, 59, 59)

    ################################

    # Initialize data loader and load filtered ionospheric data
    loader = HDF5PolarsLoader(
        data_dir="data/madrigal",
        sDate=sDate,
        eDate=eDate,
        region_name=region_name,
        freq_range={'min_freq': 1_000_000, 'max_freq': 40_000_000},
        distance_range={'min_dist': 0, 'max_dist': 20_000},
        altitudes=altitude,
        use_cache=False
    )
    df = loader.get_dataframe()
    print(df)
    print(df.columns)
    
    plot_freq_snr_heatmap(df, save_path="freq_vs_snr_heatmap.png")
    plot_paths_on_us_map(df, save_path="tx_rx_paths_map.png")


    # Output directory (named by date)
    #output_folder = f"output/{plot_date.isoformat()}_hamspot_analysis"

    # Generate plot

