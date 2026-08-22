"""
Robust OMNI 1-minute geomagnetic event analysis using CDAWeb (cdasws), pandas, and matplotlib.

What this script does
---------------------
1. Downloads OMNI 1-minute data for a chosen time range from CDAWeb.
2. Uses chunked requests plus retry logic to reduce RemoteDisconnected/network failures.
3. Converts the returned data to a pandas DataFrame.
4. Plots key storm parameters in aligned panels.
5. Marks the minimum SYM-H time if available.
6. Saves the figure to disk.

Install requirements
--------------------
python -m pip install cdasws pandas matplotlib numpy requests urllib3

Notes
-----
- Dataset used here: OMNI_HRO2_1MIN
- Variable availability can vary slightly; the script handles missing variables gracefully.
- Times are treated as UTC.
- If CDAWeb is temporarily unstable, just rerun later; this script already retries automatically.
"""

from __future__ import annotations

from cdasws import CdasWs
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path
from datetime import timedelta
import time


# =========================
# User settings
# =========================
START = '2003-10-22T00:00:00Z'
END = '2003-11-03T00:00:00Z'
DATASET = 'OMNI2_H0_MRG1HR'
OUTPUT_FIG = 'omni_event_plot.png'

# Variables commonly useful for storm analysis
REQUESTED_VARS = [
    'BX_GSE1800',
    'BY_GSE1800',
    'BZ_GSE1800',
    'ABS_B1800',
    'KP1800',
    'AP_INDEX1800',
    'DST1800',
    'AE1800'
  #  'flow_speed',
 #   'Vx',
 #   'Vy',
 #   'Vz',
  #  'SYM_H',
]

# Networking robustness settings
MAX_RETRIES = 4
RETRY_WAIT_SECONDS = 4
CHUNK_HOURS = 24  # Split the full interval into chunks of this many hours for each CDAWeb request


# =========================
# Helper functions
# =========================
def _to_datetime_index(epoch_values) -> pd.DatetimeIndex:
    """Convert CDAWeb epoch-like values to a UTC pandas DatetimeIndex."""
    return pd.to_datetime(epoch_values, utc=True)


def _make_series_if_possible(data_dict: dict, name: str, index: pd.DatetimeIndex) -> pd.Series | None:
    """
    Convert a returned CDAWeb variable to a 1D pandas Series when possible.

    Returns None for missing or non-1D variables.
    """
    if name not in data_dict:
        return None

    values = np.asarray(data_dict[name])

    if values.ndim != 1:
        return None

    values = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy()
    
    values = np.where(np.abs(values) > 9999, np.nan, values)

    if len(values) != len(index):
        return None

    return pd.Series(values, index=index, name=name)


def _extract_data_dict(result):
    """Handle a few likely cdasws return structures."""
    if isinstance(result, tuple) and len(result) >= 2:
        return result[1] 


def _fetch_single_chunk(cdas: CdasWs, start: str, end: str, dataset: str, variables: list[str]) -> pd.DataFrame:
    """Fetch one chunk from CDAWeb and return a DataFrame indexed by UTC time."""
    result = cdas.get_data(dataset, variables, start, end)
    print("Raw result type:", type(result), "with", len(result), "items")
    print("Raw result:", result)
    data = _extract_data_dict(result)
    print("Extracted data type:", type(data))
    print(data.keys())

    if not isinstance(data, dict):
        raise RuntimeError(f'Unexpected data structure returned from CDAWeb: {type(data)}')

    time_key = None
    for candidate in ['Epoch', 'EPOCH', 'epoch', 'Time']:
        if candidate in data:
            time_key = candidate
            break

    if time_key is None:
        raise KeyError('Could not find a time variable in CDAWeb response (expected something like Epoch).')

    time_index = _to_datetime_index(data[time_key])

    series_list = []
    found = []
    missing = []

    for var in variables:
        s = _make_series_if_possible(data, var, time_index)
        if s is None:
            missing.append(var)
        else:
            series_list.append(s)
            found.append(var)

    if not series_list:
        raise RuntimeError('No requested variables could be converted into 1D time series in this chunk.')

    df = pd.concat(series_list, axis=1).sort_index()

    print(f'Loaded chunk {start} -> {end}')
    print('  Found variables:', found)
    if missing:
        print('  Missing/skipped:', missing)

    return df


def _fetch_single_chunk_with_retries(cdas: CdasWs, start: str, end: str, dataset: str, variables: list[str]) -> pd.DataFrame:
    """Fetch one chunk with retry logic for transient network/server failures."""
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f'Attempt {attempt}/{MAX_RETRIES} for chunk {start} -> {end}')
            return _fetch_single_chunk(cdas, start, end, dataset, variables)
        except Exception as exc:
            last_error = exc
            print(f'  Chunk request failed: {type(exc).__name__}: {exc}')
            if attempt < MAX_RETRIES:
                print(f'  Waiting {RETRY_WAIT_SECONDS} s before retrying...')
                time.sleep(RETRY_WAIT_SECONDS)

    raise RuntimeError(f'Failed to fetch chunk after {MAX_RETRIES} attempts: {start} -> {end}') from last_error


def _build_chunk_ranges(start: str, end: str, chunk_hours: int) -> list[tuple[str, str]]:
    """Split the full interval into smaller UTC chunks."""
    start_ts = pd.Timestamp(start, tz='UTC')
    end_ts = pd.Timestamp(end, tz='UTC')

    ranges = []
    current = start_ts
    delta = pd.Timedelta(hours=chunk_hours)

    while current < end_ts:
        next_time = min(current + delta, end_ts)
        ranges.append((current.isoformat().replace('+00:00', 'Z'), next_time.isoformat().replace('+00:00', 'Z')))
        current = next_time

    return ranges


def fetch_omni_dataframe(start: str, end: str, dataset: str, variables: list[str], chunk_hours: int = CHUNK_HOURS) -> pd.DataFrame:
    """Fetch OMNI data from CDAWeb in chunks and return a combined DataFrame indexed by UTC time."""
    cdas = CdasWs()
    ranges = _build_chunk_ranges(start, end, chunk_hours)

    print(f'Fetching data in {len(ranges)} chunk(s) of up to {chunk_hours} hour(s) each...')

    dfs = []
    for chunk_start, chunk_end in ranges:
        df_chunk = _fetch_single_chunk_with_retries(cdas, chunk_start, chunk_end, dataset, variables)
        dfs.append(df_chunk)

    if not dfs:
        raise RuntimeError('No data chunks were downloaded successfully.')

    df = pd.concat(dfs, axis=0)
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    df = df.dropna(how='all')
   # print(df.columns)

    return df


def plot_omni_event(df: pd.DataFrame, output_path: str) -> None:
    """Create and save a multi-panel OMNI event plot."""
    panel_specs = [
        ('KP1800', 'Kp index', 'tab:orange'),
        ('AP_INDEX1800', 'Ap index', 'tab:green'),
        ('DST1800', 'Dst [nT]', 'tab:blue'),
        ('AE1800', 'Ae [nT]', 'tab:red'),
                                           
        
    #    ('flow_speed', 'Flow speed [km/s]', 'tab:red'),
    #    ('SYM_H', 'SYM-H [nT]', 'black'),
    ]

    available_panels = [(col, label, color) for col, label, color in panel_specs if col in df.columns]
    print(available_panels, type(available_panels))

    if not available_panels:
        raise RuntimeError('No plottable columns are available in the DataFrame.')

    fig, axes = plt.subplots(len(available_panels), 1, figsize=(11, 2.2 * len(available_panels)), sharex=True)

    if len(available_panels) == 1:
        axes = [axes]

    #symh_min_time = None
    #if 'SYM_H' in df.columns and df['SYM_H'].notna().any():
    #    symh_min_time = df['SYM_H'].idxmin()
    #    symh_min_value = df.loc[symh_min_time, 'SYM_H']
    #    print(f'\nSYM-H minimum: {symh_min_value:.1f} nT at {symh_min_time}')

    for ax, (col, ylabel, color) in zip(axes, available_panels):
        ax.plot(df.index, df[col], color=color, linewidth=1.0)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        if col == 'BZ_GSE1800':
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.8) # horizontal line for 0, B_z <0 gives 
    

    #    if symh_min_time is not None:
    #        ax.axvline(symh_min_time, color='gray', linestyle='--', linewidth=0.8)

    axes[0].set_title(f'OMNI event overview: {df.index.min()} to {df.index.max()}')
    axes[-1].set_xlabel('Time [UTC]')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M', tz=df.index.tz))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.show()

    print(f'\nFigure saved to: {Path(output_path).resolve()}')

def plot_B_field(df: pd.DataFrame, output_path: str) -> None:
    """Create a single plot for B field variations"""
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['BX_GSE1800'], label='Bx GSE', color='purple')
    plt.plot(df.index, df['BY_GSE1800'], label='By GSE', color='red')
    plt.plot(df.index, df['BZ_GSE1800'], label='Bz GSE', color='blue')
    plt.plot(df.index, df['ABS_B1800'], label='|B|', color='black')
    plt.axhline(0, color='gray', linestyle='--', alpha = 0.5, linewidth=0.8) #useful for checking whether B_z environment favours reconnection
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))

    ax.set_xlabel('Date [UTC]')


    plt.title(f'B Field Variations: {df.index.min()} to {df.index.max()}')
    #plt.xlabel('Time [UTC]')
    plt.ylabel('Magnetic Field [nT]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.show()

    print(f'\nB field figure saved to: {Path(output_path).resolve()}')
    print(df.index)

def plot_disturbance(df: pd.DataFrame, output_path: str) -> None:
    """Create a single plot for geomagnetic disturbance indices"""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Date [UTC]')
    ax1.set_ylabel('Index Value')
    ax1.plot(df.index, df[KP1800], label = 'Kp index', color = 'black')
    ax1.tick_params(axis = 'y', labelcolor = 'black')

    ax2 = ax1.twinx()  # Plot Ap index on the same x-axis but different y-axis, Kp is the logarithmic version of Ap index
    ax2.set_ylabel('ap [nT]', color = 'red')

    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d'))


    plt.title(f'Geomagnetic Disturbance Indices: {df.index.min()} to {df.index.max()}')
    plt.xlabel('Date [UTC]')
    plt.ylabel('Index Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.show()

    print(f'\nDisturbance indices figure saved to: {Path(output_path).resolve()}')
# =========================
# Main script
# =========================
def main() -> None:
    print(f'Fetching {DATASET} data from {START} to {END}...')
    print(f'Requested variables: {REQUESTED_VARS}')
    df = fetch_omni_dataframe(START, END, DATASET, REQUESTED_VARS)

    print('\nData preview:')
    print(df.head())

    print('\nSummary statistics:')
    print(df.describe())

    plot_omni_event(df, OUTPUT_FIG)
    plot_B_field(df, 'omni_B_field_plot.png')
    plot_disturbance(df, 'omni_disturbance_plot.png')
    

if __name__ == '__main__':
    main()
