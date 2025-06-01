#!/usr/bin/env python3

import polars as pl
from datetime import datetime
import apexpy

def add_geomagnetic_columns(df: pl.DataFrame, altitudes: list) -> pl.DataFrame:
    if isinstance(altitudes, int):
        altitudes = [altitudes]
    # Convert 'mid_lat' and 'mid_long' to lists
    geocen_lats = df['mid_lat'].to_list()
    geocen_lons = df['mid_long'].to_list()

    ### Placeholder!!!! get datetime from first row
    date_str = df.select(pl.col("date").cast(str)).to_series()[0][:10]

    # Prepare to store results for each altitude
    for alt in altitudes:
        # Convert 'mid_lat' and 'mid_long' to geomagnetic coordinates for each altitude
        geomagnetic_lat, geomagnetic_lon = convert_geocentric_to_geomagnetic(geocen_lats, geocen_lons, alt, date_str)
        
        # Add new columns for geomagnetic latitude and longitude
        df = df.with_columns([
            pl.Series(f'geomag_lat_{alt}', geomagnetic_lat).cast(pl.Float32),
            pl.Series(f'geomag_lon_{alt}', geomagnetic_lon).cast(pl.Float32)
        ])

    return df  # Add return statement

def convert_geocentric_to_geomagnetic(geocen_lats: list, geocen_lons: list, alt: float, date_str: str) -> tuple[list, list]:
    # Convert date string to decimal year
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    decimal_year = date_obj.year + (date_obj.timetuple().tm_yday - 1) / 365.25
    
    # Initialize the Apex object with the decimal year
    apex_out = apexpy.Apex(date=decimal_year)

    # Convert geographic latitudes and longitudes to geomagnetic coordinates
    geomagnetic_lat, geomagnetic_lon = apex_out.convert(geocen_lats, geocen_lons, 'geo', 'apex', height=alt)
    
    return geomagnetic_lat, geomagnetic_lon  # Return as tuple

def convert_geomagnetic_to_geocentric(geomag_lats: list, geomag_lons: list, alt: float, date_str: str) -> tuple[list, list]:
    """Convert geomagnetic (apex) coordinates to geocentric (geographic) coordinates."""

    # Convert date string to decimal year
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    decimal_year = date_obj.year + (date_obj.timetuple().tm_yday - 1) / 365.25

    # Initialize the Apex object
    apex_out = apexpy.Apex(date=decimal_year)

    # Convert geomagnetic (apex) to geographic (geocentric) coordinates
    geocen_lat, geocen_lon = apex_out.convert(geomag_lats, geomag_lons, 'apex', 'geo', height=alt)

    return geocen_lat, geocen_lon
