import pyspedas
from pytplot import get_data
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Load OMNI data
omni_vars = pyspedas.projects.omni.data(trange=['2013-11-05', '2013-11-06'])

print("Loaded variables:")
print(omni_vars)

# Helper function to convert pytplot variable to Python arrays
def extract_var(name):
    data = get_data(name)
    if data is None:
        print(f"Variable {name} not found")
        return None, None
    times = [datetime.utcfromtimestamp(t) for t in data.times]
    values = data.y
    return times, values

# Variables to plot
vars_to_plot = ['BZ_GSM', 'flow_speed', 'SYM_H']

fig, axes = plt.subplots(len(vars_to_plot), 1, figsize=(10, 8), sharex=True)

for ax, var in zip(axes, vars_to_plot):
    times, values = extract_var(var)
    if times is not None:
        ax.plot(times, values, label=var)
        ax.set_ylabel(var)
        ax.legend(loc='upper right')
        ax.grid(True)

axes[-1].set_xlabel('Time (UTC)')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%Y-%m-%d'))

plt.tight_layout()
plt.show()