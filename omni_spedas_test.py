import os

# Set where PySPEDAS should cache/download data
os.environ["SPEDAS_DATA_DIR"] = r"C:\spedas_data"
os.makedirs(os.environ["SPEDAS_DATA_DIR"], exist_ok=True)

print("SPEDAS_DATA_DIR =", os.environ.get("SPEDAS_DATA_DIR"))
print("Contents before load =", os.listdir(os.environ["SPEDAS_DATA_DIR"]))

import pyspedas
from pytplot import tplot_names


def main():
    # Load OMNI data for the 2015 St. Patrick's Day storm
    trange = ["2015-03-16", "2015-03-19"]

    omni_vars = pyspedas.omni.data(trange=trange, datatype="1min")

    print("\nLoaded OMNI variables:")
    print(omni_vars)

    print("\nCurrent tplot variables:")
    print(tplot_names())

    print("\nContents after load =", os.listdir(os.environ["SPEDAS_DATA_DIR"]))
    print("\nWalk of SPEDAS_DATA_DIR:")
    for root, dirs, files in os.walk(os.environ["SPEDAS_DATA_DIR"]):
        print(root)
        for f in files[:10]:
            print("   ", f)


if __name__ == "__main__":
    main()
