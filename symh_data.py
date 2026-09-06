from __future__ import annotations
from operator import index

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
DATASET = 'OMNI_HRO_5MIN' #5 minute resolution data from OMNI
OUTPUT_FIG = 'omni_symh_plot.png'

REQUESTED_VARS = ['SYM_H'] #List of variables to request from OMNI dataset. SYM-H is the main variable of interest for this project.

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

def plot_symh(df: pd.DataFrame, output_file: str):
    """Plot the SYM-H index from the DataFrame and save to a file."""
    if 'SYM_H' not in df.columns:
        raise KeyError('SYM_H variable not found in DataFrame.')\

    dsymh = df['SYM_H'].diff()
    cond_threshold = df['SYM_H'] < -50
    cond_decreasing = (
        (dsymh < -5) & (dsymh.shift(-1) < -5) & (dsymh.shift(-2) < -5)
    )

    cond_date = (df.index >= pd.Timestamp('2003-10-28T00:00:00Z'))

    storm_mask = df.shift(2).index[cond_threshold & cond_decreasing & cond_date]
    print(storm_mask)
    print(len(storm_mask))
    if len(storm_mask) > 0:
        storm_start = storm_mask[0]
        storm_end = storm_mask[-1]
        print(f'Storm period detected from {storm_start} to {storm_end}')
    else:
        storm_start = None
        storm_end = None
        print('No storm period detected.')

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['SYM_H'], label='SYM-H', color='blue')
    plt.title('SYM-H Index from OMNI Data')
    plt.axvline(storm_start, color='black', linestyle='--', label='Storm Start')
    plt.axvline(storm_end, color='black', linestyle='--', label='Storm End')
    plt.axvspan(storm_start, storm_end, color='gray', alpha=0.3, label='Storm Period')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval = 1))
    plt.xlabel('Time (UTC)')
    plt.ylabel('SYM-H (nT)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.savefig(output_file)
    plt.show()
    #print(f'Plot saved to {output_file}')

def main():
    """Main function to fetch OMNI data and plot SYM-H index."""
    df = fetch_omni_dataframe(START, END, DATASET, REQUESTED_VARS)
    print(f'Data fetched with {len(df)} rows.')
    plot_symh(df, OUTPUT_FIG)

if __name__ == '__main__':
    print(f'Fetching {DATASET} from {START} to {END}')
    df = fetch_omni_dataframe(START, END, DATASET, REQUESTED_VARS)
    print(df.head())
    print(df.describe())
    plot_symh(df, OUTPUT_FIG)