#!/usr/bin/env python3

import shutil
import dask.dataframe as dd
import polars as pl
import pandas as pd
from datetime import datetime
import apexpy
import logging
from pathlib import Path
from dask.diagnostics import ProgressBar
from scripts.utils import *
from scripts.utils_geo import *
from scripts.regions import REGIONS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HDF5PolarsLoader:
    def __init__(self, 
                 data_dir: str, 
                 sDate: datetime, 
                 eDate: datetime, 
                 cache_dir: str = "../cache/df_gen", 
                 use_cache: bool = True, 
                 region_name: str = None, 
                 freq_range: dict = None, 
                 distance_range: dict = None, 
                 chunk_size: int = 100000, 
                 altitudes: list = None):
        """
        :param data_dir: Directory where the HDF5 files are stored.
        :param date_str: The date in the format 'YYYY-MM-DD' to construct the file name.
        :param cache_dir: Directory to store cached files.
        :param use_cache: Whether to load from cache if available.
        :param region: Region filter for data, containing 'lat_lim' and 'lon_lim'.
        :param freq_range: Frequency range filter for data, containing 'min_freq' and 'max_freq'.
        :param chunk_size: Chunk size for reading the HDF5 file.
        :param altitudes: List of altitudes to calculate geomagnetic coordinates.
        """

        if region_name not in REGIONS:
            raise ValueError(f"Region '{region_name}' is not defined.")
        
        self.data_dir       = Path(data_dir)
        self.sDate          = sDate
        self.eDate          = eDate
        self.cache_dir      = Path(cache_dir)
        self.use_cache      = use_cache
        self.region         = REGIONS[region_name]
        self.freq_range     = freq_range
        self.distance_range = distance_range
        self.chunk_size     = chunk_size
        self.altitudes      = altitudes 

        # Extract datetimes
        

        # Construct dynamic cache path
        region_str = region_name if region_name else "full_region"
        
        if self.freq_range:
            min_freq_mhz = self.freq_range['min_freq'] / 1_000_000
            max_freq_mhz = self.freq_range['max_freq'] / 1_000_000
            freq_str = f"{min_freq_mhz:.2f}MHz_{max_freq_mhz:.2f}MHz"
        else:
            freq_str = "full_freq_range"
    
        if self.distance_range:
            min_dist = self.distance_range['min_dist']
            max_dist = self.distance_range['max_dist']
            distance_str = f"dist{min_dist}_{max_dist}km"
        else:
            distance_str = "full_distance_range"
            
        altitudes = self.altitudes if isinstance(self.altitudes, (list, tuple)) else [self.altitudes]
        altitudes_str = '_'.join(str(alt) for alt in altitudes)
    
        # cache path
        self.cache_path = self.cache_dir / f"{sDate}_{eDate}_{freq_str}_{region_str}_{distance_str}_{altitudes_str}km.parquet"
        self.df = None
        self.log = logging.getLogger(__name__)
    
        self.cache_dir.mkdir(parents=True, exist_ok=True)


    def get_file_path(self, date_str):
        """Construct the file path based on the date string."""
        file_name = f"rsd{date_str}.01.hdf5"
        return self.data_dir / file_name

    def load_data(self):
        """Split, load, and concat hdf5 files in required range of dates."""
        datetime_split = split_datetime_range_by_day(self.sDate, self.eDate)
        all_dfs = []
        for sDate, eDate, date_str in datetime_split:
            file_name = f"rsd{date_str}.01.hdf5"
            file_path = self.data_dir / file_name
            self.log.info(f"Loading data from {sDate} - {eDate}...") 
            self.log.info(f"Loading data from HDF5 file {file_path}...")        

            try:
                # Load only the required columns from the HDF5 file
                dask_df = dd.read_hdf(file_path, key="Data/Table Layout", chunksize=self.chunk_size)
            except (FileNotFoundError, OSError) as e:
                self.log.warning(f"Could not load file: {file_path} - Skipping. ({e})")
                continue
    
            required_columns = [
                'year', 'month', 'day', 'hour', 'min', 'sec',
                'pthlen', 'rxlat', 'rxlon', 'txlat', 'txlon',
                'tfreq', 'latcen', 'loncen', 'ssrc', 'sn'
            ]
            
            dask_df = dask_df[required_columns]
            
            # Downcast columns to lower precision where appropriate
            dask_df = dask_df.astype({
                'rxlat': 'float32',  # Downcast latitude to float32
                'rxlon': 'float32',  # Downcast longitude to float32
                'txlat': 'float32',  # Downcast latitude to float32
                'txlon': 'float32',  # Downcast longitude to float32
                'latcen': 'float32',
                'loncen': 'float32',
                'ssrc': 'category',
                'tfreq': 'float32',  # Downcast frequency to float32
                'pthlen': 'int32',    # Downcast distance to int32 (as you're only interested in distances under 8000 km)
                'year': 'int16',
                'month': 'int8',
                'day': 'int8',
                'hour': 'int8',
                'min': 'int8',
                'sec': 'int8'
            })
            
            # Apply filters to the data (if any) before proceeding with the rest of the steps
            if sDate and eDate:
                dask_df = dask_df.map_partitions(self.apply_datetime_filter, sDate, eDate)
            if self.region:
                dask_df = dask_df.map_partitions(self.apply_region_filter)
            if self.freq_range:
                dask_df = dask_df.map_partitions(self.apply_freq_filter)
            if self.distance_range:
                dask_df = dask_df.map_partitions(self.apply_distance_filter)

            dask_df = dask_df.dropna(subset=['sn'])
            # Continue processing and converting to pandas (or directly to Polars)
            with ProgressBar():
                df = dask_df.compute()
                all_dfs.append(df)
                
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
        else:
            final_df = pd.DataFrame()

        self.log.info(f"Loaded and merged data from {self.sDate} - {self.eDate}...") 
        return final_df

    def process_data(self):
        """Load the dataset from cache or process the HDF5 file using Dask and convert to polars df."""
        if self.use_cache and self.cache_path.exists():
            self.log.info(f"Loading data from cache for {self.sDate} - {self.eDate}...")
            self.df = pl.read_parquet(self.cache_path)
            return self.df

        df = self.load_data()
        
        # Renaming and processing the dataframe as before
#        df['occurred'] = pd.to_datetime(df['year'] + '-' + df['month'] + '-' + df['day'] + ' ' + df['hour'] + ':' + df['min'] + ':' + df['sec'])
#        df.drop(['year', 'month', 'day', 'hour', 'min', 'sec'], axis=1, inplace=True)
        df['occurred'] = pd.to_datetime({
            'year': df['year'],
            'month': df['month'],
            'day': df['day'],
            'hour': df['hour'],
            'minute': df['min'],
            'second': df['sec'],
        })
        df.drop(['year', 'month', 'day', 'hour', 'min', 'sec'], axis=1, inplace=True)
        
        df = df.rename(columns={"occurred": "date",
                                "pthlen": "dist_Km", 
                                "rxlat": "rx_lat", 
                                "rxlon": "rx_long", 
                                "txlat": "tx_lat", 
                                "txlon": "tx_long",
                                "tfreq": "freq",
                                "ssrc": "source",
                                "latcen": "mid_lat",
                                "loncen": "mid_long",
                                "sn": "snr"})

        #move to polars df
        df['band'] = df['freq'].apply(self.get_band).astype('int8')
        
        df = df[['date', 'freq', 'band', 'dist_Km', 'source', 'mid_lat', 'mid_long', 'rx_lat', 'tx_lat', 'rx_long', 'tx_long', 'snr']]
        
        # Convert to Polars
        df_polars = pl.from_pandas(df)
        
        # Apply geomagnetic conversion
        df_polars = add_geomagnetic_columns(df_polars, altitudes=self.altitudes)
        
        # Save to cache
        df_polars.write_parquet(self.cache_path, compression='snappy')
        self.df = df_polars
        return self.df

    def apply_datetime_filter(self, df, sDate, eDate):
        """Apply the datetime filter to the Dask DataFrame based on start and end datetime."""
        if sDate and eDate:
            # Split start datetime
            sy, smo, sd, sh, smin, ssec = (self.sDate.year, self.sDate.month, self.sDate.day,
                                           self.sDate.hour, self.sDate.minute, self.sDate.second)
            # Split end datetime
            ey, emo, ed, eh, emin, esec = (self.eDate.year, self.eDate.month, self.eDate.day,
                                           self.eDate.hour, self.eDate.minute, self.eDate.second)
    
            # Apply the datetime filter
            df = df[
                (
                    (df['year'] > sy) |
                    ((df['year'] == sy) & (df['month'] > smo)) |
                    ((df['year'] == sy) & (df['month'] == smo) & (df['day'] > sd)) |
                    ((df['year'] == sy) & (df['month'] == smo) & (df['day'] == sd) & (df['hour'] > sh)) |
                    ((df['year'] == sy) & (df['month'] == smo) & (df['day'] == sd) & (df['hour'] == sh) & (df['min'] > smin)) |
                    ((df['year'] == sy) & (df['month'] == smo) & (df['day'] == sd) & (df['hour'] == sh) & (df['min'] == smin) & (df['sec'] >= ssec))
                ) &
                (
                    (df['year'] < ey) |
                    ((df['year'] == ey) & (df['month'] < emo)) |
                    ((df['year'] == ey) & (df['month'] == emo) & (df['day'] < ed)) |
                    ((df['year'] == ey) & (df['month'] == emo) & (df['day'] == ed) & (df['hour'] < eh)) |
                    ((df['year'] == ey) & (df['month'] == emo) & (df['day'] == ed) & (df['hour'] == eh) & (df['min'] < emin)) |
                    ((df['year'] == ey) & (df['month'] == emo) & (df['day'] == ed) & (df['hour'] == eh) & (df['min'] == emin) & (df['sec'] <= esec))
                )]
        return df
    

    def apply_region_filter(self, df):
        if self.region:
            tx_box = self.region.get('tx_box', None)
            rx_box = self.region.get('rx_box', None)
    
            if tx_box:
                tx_lat_min = tx_box['lat_min']
                tx_lat_max = tx_box['lat_max']
                tx_lon_min = tx_box['lon_min']
                tx_lon_max = tx_box['lon_max']
    
                df = df[(df['txlat'] >= tx_lat_min) & (df['txlat'] <= tx_lat_max)]
                df = df[(df['txlon'] >= tx_lon_min) & (df['txlon'] <= tx_lon_max)]
    
            if rx_box:
                rx_lat_min = rx_box['lat_min']
                rx_lat_max = rx_box['lat_max']
                rx_lon_min = rx_box['lon_min']
                rx_lon_max = rx_box['lon_max']
    
                df = df[(df['rxlat'] >= rx_lat_min) & (df['rxlat'] <= rx_lat_max)]
                df = df[(df['rxlon'] >= rx_lon_min) & (df['rxlon'] <= rx_lon_max)]
    
        return df

    def apply_freq_filter(self, df):
        """Apply the frequency filter to the Dask DataFrame using the 'tfreq' column."""
        if self.freq_range:
            min_freq, max_freq = self.freq_range['min_freq'], self.freq_range['max_freq']
            df = df[(df['tfreq'] >= min_freq) & (df['tfreq'] <= max_freq)]
        return df

    def apply_distance_filter(self, df):
        """Apply the distance filter to the Dask DataFrame using the 'pthlen' column."""
        if self.distance_range:
            min_dist, max_dist = self.distance_range['min_dist'], self.distance_range['max_dist']
            df = df[(df['pthlen'] >= min_dist) & (df['pthlen'] <= max_dist)]
        return df

    def get_band(self, frequency):
        """Assign a frequency band based on the value."""
        if 137 <= frequency < 2000000:          # 160 meters band (0.137 - 2 MHz)
            return 160
        elif 2000 <= frequency < 4000000:       # 80 meters band (2 - 4 MHz)
            return 80
        elif 4000 <= frequency < 7000000:       # 40 meters band (4 - 7 MHz)
            return 40
        elif 7000 <= frequency < 14000000:      # 20 meters band (7 - 14 MHz)
            return 20
        elif 14000 <= frequency < 21000000:     # 15 meters band (14 - 21 MHz)
            return 15
        elif 21000 <= frequency < 30000000:     # 10 meters band (21 - 30 MHz)
            return 10
        else:
            return 0  

    def get_dataframe(self):
        """Return the loaded Polars DataFrame."""
        if self.df is None:
            self.process_data()
        return self.df

    def clear_cache(self):
        """Delete all files in the cache directory."""
        if self.cache_dir.exists() and self.cache_dir.is_dir():
            for cache_file in self.cache_dir.iterdir():
                if cache_file.is_file() and cache_file.name.startswith(f"{self.sDate}_{self.eDate}_"):
                    cache_file.unlink()
                    self.log.info(f"Cache file removed: {cache_file}")
        else:
            self.log.info(f"Cache directory not found: {self.cache_dir}")


if __name__ == "__main__":

    #For testing only!
    
    freq_range = {
        'min_freq': 6000000,  # Example minimum frequency (0 MHz)
        'max_freq': 8000000  # Example maximum frequency (30 MHz)
    }

    distance_range = {
        'min_dist': 0,    # Example minimum distance in kilometers
        'max_dist': 20000  # Example maximum distance in kilometers
    }

    # Define the altitudes for conversion
    altitudes = [0,100,300]  # Altitudes in km

    # Define the date range you want to process (e.g., a list of dates)
    sDate = datetime(2017, 7, 1, 21, 8, 0)
    eDate = datetime(2017, 7, 2, 4, 8, 0)

    region = 'Equatorial America'

    loader = HDF5PolarsLoader(
        data_dir="data/madrigal", 
        sDate=sDate,
        eDate=eDate,
        region_name=region, 
        freq_range=freq_range,
        distance_range=distance_range,
        altitudes=altitudes,  # Altitudes passed here
        use_cache=True
    )

    # Clear cache and load the dataframe (it will use the cache if available and `use_cache=True`)
    loader.clear_cache()  # Clear cache first
    df = loader.get_dataframe()  # Load the data

    # Print the processed data
    print(f"Finished df_gen test run for {sDate} - {eDate}:")
    print(df)
