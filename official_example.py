#  Install these prerequisites once before executing the example code:
#  PyPI spacepy Option:
#    pip install -U spacepy cdasws
#  PyPI xarray Option:
#    pip install -U xarray cdflib cdasws
#  conda-forge xarray Option:
#    conda install conda-forge::xarray conda-forge::cdflib conda-forge::cdasws

from cdasws.cdasws import CdasWs
cdas = CdasWs()

DATASET = 'OMNI2_H0_MRG1HR'
# Edit the following var_names and example_interval
# variables to suit your needs.
var_names = cdas.get_variable_names(DATASET)
print('Variable names:', var_names)
example_interval = cdas.get_example_time_interval(DATASET)
print('Example time interval:', example_interval)
status, data = cdas.get_data(DATASET, var_names, example_interval)

if 'spacepy' in str(type(data)):
    #  see https://spacepy.github.io/datamodel.html
    print(var_names[0], '=', data[var_names[0]])
    print(data[var_names[0]].attrs)
else:
    #  see https://github.com/MAVENSDC/cdflib
    print(var_names[0], '=', data.data_vars[var_names[0]].values)
    print(data.data_vars[var_names[0]].attrs)

print(data)
# ...

