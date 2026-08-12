#import os

# Set where PySPEDAS should cache/download data
#os.environ["SPEDAS_DATA_DIR"] = r"C:\spedas_data"
#os.makedirs(os.environ["SPEDAS_DATA_DIR"], exist_ok=True)

#print("SPEDAS_DATA_DIR =", os.environ.get("SPEDAS_DATA_DIR"))
#print("Contents before load =", os.listdir(os.environ["SPEDAS_DATA_DIR"]))

import pyspedas
#from pyspedas import tplot
omni_vars = pyspedas.projects.omni.data(trange=['2013-11-5', '2013-11-6'])
print(omni_vars)
#tplot(['BX_GSE', 'BY_GSE', 'BZ_GSE', 'flow_speed', 'Vx', 'Vy', 'Vz', 'SYM_H'])