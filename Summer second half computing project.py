import os

# Set SPEDAS data directory
os.environ["SPEDAS_DATA_DIR"] = r"C:\spedas_data"
os.makedirs(r"C:\spedas_data", exist_ok=True)

print("SPEDAS_DATA_DIR =", os.environ.get("SPEDAS_DATA_DIR"))
print("Folder exists:", os.path.exists(r"C:\spedas_data"))
print("Contents before download:", os.listdir(r"C:\spedas_data"))

import pyspedas

print("Imported pyspedas successfully")

# Try loading OMNI data
try:
    omni_vars = pyspedas.omni.data(
        trange=["2015-03-17", "2015-03-18"],
        datatype="1min"
    )
    print("OMNI load returned:", omni_vars)
except Exception as e:
    print("Error while loading OMNI:", e)

print("Contents after download:", os.listdir(r"C:\spedas_data"))