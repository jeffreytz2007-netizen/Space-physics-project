"""
Integrated southward IMF forcing versus geomagnetic-storm intensity.

Expected CSV columns:
    time, Bz, Vsw, SYMH

Units:
    time: UTC timestamps
    Bz: nT, preferably GSM Bz
    Vsw: km/s
    SYMH: nT

This script uses event windows supplied by you. Define onset and end times
using a documented storm catalogue or an explicit, reproducible criterion.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# User settings
# ----------------------------
DATA_FILE = Path("omni_clean.csv")

# Replace these example events with your selected storms.
# For each event, use the same definition of onset and end.
EVENTS = [
    {
        "event": "Halloween 2003",
        "onset": "2003-10-29 00:00",
        "end": "2003-11-01 00:00",
    },
    # {"event": "Storm 2", "onset": "YYYY-MM-DD HH:MM", "end": "YYYY-MM-DD HH:MM"},
]

# Optional threshold: only count southward IMF stronger than this value.
# Set to 0.0 to count all southward Bz.
BZ_THRESHOLD_NT = 0.0


def prepare_data(path: Path) -> pd.DataFrame:
    """Read, clean, sort, and calculate time intervals."""
    df = pd.read_csv(path, parse_dates=["time"])
    required = {"time", "Bz", "Vsw", "SYMH"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df[["time", "Bz", "Vsw", "SYMH"]].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["time", "Bz", "Vsw", "SYMH"])
    df = df.sort_values("time").drop_duplicates("time")

    # Duration of each sample in hours. The final row has no forward interval.
    df["dt_h"] = df["time"].diff().dt.total_seconds().div(3600.0).shift(-1)
    df["dt_h"] = df["dt_h"].where((df["dt_h"] > 0) & (df["dt_h"] < 6.0))

    # Southward magnetic-field magnitude: Bs = max(0, -Bz - threshold).
    # For Bz threshold = 0, northward Bz contributes zero.
    df["Bs"] = (-df["Bz"] - BZ_THRESHOLD_NT).clip(lower=0.0)

    # Solar-wind electric-field proxy in mV/m:
    # Ey_proxy = Vsw * Bs * 1e-3, with Vsw in km/s and Bs in nT.
    df["Ey_mVm"] = df["Vsw"] * df["Bs"] * 1e-3
    return df


def integrate_event(event_df: pd.DataFrame) -> dict:
    """Calculate forcing measures and storm intensity for one event."""
    valid = event_df.dropna(subset=["dt_h"])
    if valid.empty:
        raise ValueError("Event contains no valid time intervals")

    # Rectangular integration. For higher-quality work, compare with a
    # trapezoidal integration and report that the conclusion is unchanged.
    integrated_bs = (valid["Bs"] * valid["dt_h"]).sum()       # nT h
    integrated_ey = (valid["Ey_mVm"] * valid["dt_h"]).sum()  # mV/m h

    minimum_symh = event_df["SYMH"].min()
    minimum_time = event_df.loc[event_df["SYMH"].idxmin(), "time"]

    duration = (event_df["time"].iloc[-1] - event_df["time"].iloc[0]).total_seconds() / 3600.0
    return {
        "duration_h": duration,
        "integrated_Bs_nT_h": integrated_bs,
        "integrated_Ey_mVm_h": integrated_ey,
        "minimum_SYMH_nT": minimum_symh,
        "minimum_SYMH_time": minimum_time,
        "mean_Bs_nT": valid["Bs"].mean(),
        "maximum_Bs_nT": valid["Bs"].max(),
        "maximum_Ey_mVm": valid["Ey_mVm"].max(),
    }


def analyse_events(df: pd.DataFrame, events: list[dict]) -> pd.DataFrame:
    rows = []
    for event in events:
        onset = pd.Timestamp(event["onset"], tz="UTC")
        end = pd.Timestamp(event["end"], tz="UTC")
        mask = (df["time"] >= onset) & (df["time"] <= end)
        event_df = df.loc[mask].copy()
        if event_df.empty:
            print(f"Skipping {event['event']}: no data in selected interval")
            continue
        result = integrate_event(event_df)
        result["event"] = event["event"]
        result["onset"] = onset
        result["end"] = end
        rows.append(result)
    return pd.DataFrame(rows)


def plot_event(df: pd.DataFrame, event: dict) -> None:
    """Plot the quantities used for one event."""
    onset = pd.Timestamp(event["onset"], tz="UTC")
    end = pd.Timestamp(event["end"], tz="UTC")
    sub = df[(df["time"] >= onset) & (df["time"] <= end)]

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(sub["time"], sub["Bz"], color="tab:blue")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("Bz [nT]")

    axes[1].plot(sub["time"], sub["Bs"], color="tab:orange")
    axes[1].set_ylabel("Bs [nT]")

    axes[2].plot(sub["time"], sub["Ey_mVm"], color="tab:green")
    axes[2].set_ylabel("Ey [mV/m]")

    axes[3].plot(sub["time"], sub["SYMH"], color="black")
    axes[3].set_ylabel("SYM-H [nT]")
    axes[3].set_xlabel("UTC")

    fig.suptitle(event["event"])
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = prepare_data(DATA_FILE)
    summary = analyse_events(data, EVENTS)

    if summary.empty:
        raise RuntimeError("No events were successfully analysed")

    print(summary.to_string(index=False))

    # Storm intensity is represented by the positive magnitude of the minimum
    # SYM-H value, so stronger storms have larger values here.
    summary["storm_intensity_nT"] = -summary["minimum_SYMH_nT"]

    # Correlation is meaningful only when several events are available.
    if len(summary) >= 3:
        print("\nEvent-level correlations:")
        print(summary[["integrated_Bs_nT_h", "integrated_Ey_mVm_h", "storm_intensity_nT"]].corr())

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(summary["integrated_Ey_mVm_h"], summary["storm_intensity_nT"])
        for _, row in summary.iterrows():
            ax.annotate(row["event"], (row["integrated_Ey_mVm_h"], row["storm_intensity_nT"]))
        ax.set_xlabel("Integrated Ey proxy [mV/m h]")
        ax.set_ylabel("Storm intensity, -minimum SYM-H [nT]")
        ax.set_title("Integrated southward forcing versus storm intensity")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()
    else:
        print("\nAdd at least three events before interpreting event-level correlations.")

    plot_event(data, EVENTS[0])
