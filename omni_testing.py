"""
OMNI 1-minute geomagnetic event analysis using CDAWeb (cdasws), pandas, and matplotlib.

What this script does
---------------------
1. Downloads OMNI 1-minute data for a chosen time range from CDAWeb.
2. Converts the returned data to a pandas DataFrame.
3. Plots key storm parameters in aligned panels.
4. Marks the minimum SYM-H time if available.
5. Saves the figure to disk.

Install requirements
--------------------
python -m pip install cdasws pandas matplotlib numpy

Notes
-----
- Dataset used here: OMNI_HRO2_1MIN
- Variable availability can vary slightly; the script handles missing variables gracefully.
- Times are treated as UTC.
"""

from __future__ import annotations

from cdasws import CdasWs
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path


# =========================
# User settings
# =========================
START = '2013-11-05T00:00:00Z'
END = '2013-11-06T00:00:00Z'
DATASET = 'OMNI_HRO2_1MIN'
OUTPUT_FIG = 'omni_event_plot.png'

# Variables commonly useful for storm analysis
REQUESTED_VARS = [
    'BZ_GSM',
    'flow_speed',
    'Vx',
    'Vy',
    'Vz',
    'proton_density',
    'Pressure',
    'SYM_H',
]


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

    # Expect 1D time series for this workflow.
    if values.ndim != 1:
        return None

    # Replace obvious fill values with NaN where possible.
    # OMNI fill values are often large/special numbers, but these can vary.
    values = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy()
    values = np.where(np.abs(values) > 1e4, np.nan, values)

    if len(values) != len(index):
        return None

    return pd.Series(values, index=index, name=name)


def fetch_omni_dataframe(start: str, end: str, dataset: str, variables: list[str]) -> pd.DataFrame:
    """Fetch OMNI data from CDAWeb and return a DataFrame indexed by UTC time."""
    cdas = CdasWs()

    result = cdas.get_data(dataset, variables, start, end)

    # cdasws typically returns a tuple-like structure: (status, data)
    # but this can vary slightly by version, so handle a few cases.
    if isinstance(result, tuple) and len(result) >= 2:
        data = result[1]
    else:
        data = result

    if not isinstance(data, dict):
        raise RuntimeError(f'Unexpected data structure returned from CDAWeb: {type(data)}')

    # Find the time variable. CDAWeb commonly uses Epoch.
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
        raise RuntimeError('No requested variables could be converted into 1D time series.')

    df = pd.concat(series_list, axis=1)
    df = df.sort_index()

    print('\nVariables successfully loaded:')
    print(found)

    if missing:
        print('\nVariables missing or skipped:')
        print(missing)

    return df


def plot_omni_event(df: pd.DataFrame, output_path: str) -> None:
    """Create and save a multi-panel OMNI event plot."""
    panel_specs = [
        ('BZ_GSM', 'Bz GSM [nT]', 'tab:blue'),
        ('flow_speed', 'Flow speed [km/s]', 'tab:red'),
        ('Vx', 'Vx [km/s]', 'tab:orange'),
        ('Vy', 'Vy [km/s]', 'tab:purple'),
        ('Vz', 'Vz [km/s]', 'tab:brown'),
        ('proton_density', 'Proton density [cm$^{-3}$]', 'tab:green'),
        ('Pressure', 'Dynamic pressure [nPa]', 'tab:pink'),
        ('SYM_H', 'SYM-H [nT]', 'black'),
    ]

    available_panels = [(col, label, color) for col, label, color in panel_specs if col in df.columns]

    if not available_panels:
        raise RuntimeError('No plottable columns are available in the DataFrame.')

    fig, axes = plt.subplots(len(available_panels), 1, figsize=(11, 2.2 * len(available_panels)), sharex=True)

    if len(available_panels) == 1:
        axes = [axes]

    symh_min_time = None
    if 'SYM_H' in df.columns and df['SYM_H'].notna().any():
        symh_min_time = df['SYM_H'].idxmin()
        symh_min_value = df.loc[symh_min_time, 'SYM_H']
        print(f'\nSYM-H minimum: {symh_min_value:.1f} nT at {symh_min_time}')

    for ax, (col, ylabel, color) in zip(axes, available_panels):
        ax.plot(df.index, df[col], color=color, linewidth=1.0)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        if col == 'BZ_GSM':
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)

        if symh_min_time is not None:
            ax.axvline(symh_min_time, color='gray', linestyle='--', linewidth=0.8)

    axes[0].set_title(f'OMNI event overview: {df.index.min()} to {df.index.max()}')
    axes[-1].set_xlabel('Time [UTC]')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M', tz=df.index.tz))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.show()

    print(f'\nFigure saved to: {Path(output_path).resolve()}')


# =========================
# Main script
# =========================
def main() -> None:
    print(f'Fetching {DATASET} data from {START} to {END}...')
    df = fetch_omni_dataframe(START, END, DATASET, REQUESTED_VARS)

    print('\nData preview:')
    print(df.head())

    print('\nSummary statistics:')
    print(df.describe())

    plot_omni_event(df, OUTPUT_FIG)


if __name__ == '__main__':
    main()
