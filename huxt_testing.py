#%%
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import datetime
import os
import sys
print(sys.executable)
import huxt.huxt as H
import huxt.huxt_analysis as HA
import huxt.huxt_inputs as Hin
#%%
'''Example 1: A 1D run with user-specified BCs'''
# Form longitudinal boundary conditions - background wind of 400 km/s with two fast streams.
v_boundary = np.ones(128) * 400 * (u.km/u.s)
v_boundary[30:50] = 600 * (u.km/u.s)
v_boundary[95:125] = 700 * (u.km/u.s)
#piecewise, one end has 400, middle has 600, other end has 700.
# This boundary condition looks like
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(v_boundary,'k-')
ax.set_xlabel('Longitude bin')
ax.set_ylabel('Input Wind Speed (km/s)')

# Setup HUXt to do a 5-day simulation, with model output every 4 timesteps (roughly half and hour time step), looking at 0 longitude
model = H.HUXt(v_boundary=v_boundary, lon_out=0.0*u.deg, simtime=10*u.day, dt_scale=4)

# Solve these conditions, with no ConeCMEs added.
cme_list = []
model.solve(cme_list)

# Plot the radial profile of the ambient wind profile at a fixed time (in days). 
t = 1.5*u.day
HA.plot_radial(model, t, lon=0.0)

# Plot the time series of the ambient wind profile at a fixed radius. 
r = 1.0*u.AU
HA.plot_timeseries(model, r, lon=0.0)
#%%
'''Example 2: A 1D run with a single ConeCME added to the background wind'''
# Set up a ConeCME that launches half a day after the simulation begins, at 0 longitude, 30 degree width, speed 850km/s and thickness=5 solar radii
cme = H.ConeCME(t_launch=0.5*u.day, longitude=0.0*u.deg, width=30*u.deg, v=850*(u.km/u.s), thickness=5*u.solRad)
cme_list = [cme]

# Setup HUXt to do a 5-day simulation, with model output every 4 timesteps (roughly half and hour time step), looking at 0 longitude
# Form longitudinal boundary conditions - background wind of 400 km/s with two fast streams.
v_boundary = np.ones(128) * 400 * (u.km/u.s) 
v_boundary[30:50] = 600 * (u.km/u.s)
v_boundary[95:125] = 700 * (u.km/u.s)
#Original boundary conditions is 400 km/s uniform. Now I will add two fast streams just like the previous example
model = H.HUXt(v_boundary=v_boundary, lon_out=0.0*u.deg, simtime=5*u.day, dt_scale=4)

# Run the model, and this time save the results to file.
model.solve(cme_list, save=True, tag='1d_conecme_test')

# Plot the radial profile and time series of both the ambient and ConeCME solutions at a fixed time (in days). 
# Save both to file as well. These are saved in HUXt>figures>HUXt1D
t = 2*u.day
HA.plot_radial(model, t, lon=0.0*u.deg, save=True)

r = 1.0*u.AU
HA.plot_timeseries(model, r, lon=0.0*u.deg, tag='1d_cone_test_radial')
#%%
'''Example 3: A 2D run with a single ConeCME added to the background wind'''
# Form boundary conditions - background wind of 400 km/s with two fast streams.
v_boundary = np.ones(360) * 400 * (u.km/u.s)
v_boundary[30:50] = 600 * (u.km/u.s)
v_boundary[95:125] = 500 * (u.km/u.s)

# Add a CME
cme = H.ConeCME(t_launch=1*u.day, longitude=360*u.deg, latitude = 30*u.deg, width=70*u.deg, v=1200*(u.km/u.s), thickness=1*u.solRad)
cme_list = [cme]

# Setup HUXt to do a 5-day simulation, with model output every 4 timesteps (roughly half and hour time step)
model = H.HUXt(v_boundary=v_boundary, latitude = 0*u.deg, simtime=6*u.day, dt_scale=4)

model.solve(cme_list, tag='cone_cme_test')

# Plot this out
t_interest = 3*u.day
fig, ax = HA.plot(model, t_interest)

# %%
'''Example 4: 2D run with actual near-sun observations from MAS'''
# HUXt can be easily initiated MAS, by specifying a carrington rotation number. Data are downloaded from the Pred Sci Inc archive on demand
#from sunpy.coordinates import sun
#from astropy.time import Time

#t = Time('2024-05-09T00:00:00')
#print(sun.carrington_rotation_number(t))

cr = 2254
v_mas = Hin.get_MAS_long_profile(cr, 0.0*u.deg)

# MAS solutions are at 30 rS
model = H.HUXt(v_boundary=v_mas, cr_num=cr, simtime=5*u.day, dt_scale=4, r_min=30*u.solRad)
model.solve([])

# Plot the solution 
fig, ax = HA.plot(model, 0.2*u.day)
fig.suptitle('MAS/HUXt')

# Read in the data
cr = 2285

demo_dir = H._setup_dirs_()['example_inputs']
print(H._setup_dirs_())

wsafilepath = os.path.join(demo_dir, '2003-05-09T22Z.wsa.gong.fits') #getting data slightly before the 2024 May storms
#pfssfilepath = os.path.join(demo_dir, 'windbound_b_pfss20220224.22.nc')
#cortomfilepath = os.path.join(demo_dir, 'tomo_sta_cor2_20240224153428_8-0.dat')
#dumfricfilepath = os.path.join(demo_dir, 'windbound_b20220224.12.nc')

v_wsa = Hin.get_WSA_long_profile(wsafilepath, lat=0.0 * u.deg)
#v_pfss = Hin.get_PFSS_long_profile(pfssfilepath, lat=0.0 * u.deg)
#v_dumfric = Hin.get_PFSS_long_profile(dumfricfilepath, lat=0.0 * u.deg) #DUMFRIC uses the PFSS reader too
#v_cortom = Hin.get_CorTom_long_profile(cortomfilepath, lat=0.0 * u.deg)

# set up and run the models. WSA and PFSS maps are at 21.5 rS, CorTom at 8 rS
model = H.HUXt(v_boundary=v_wsa, cr_num=cr, simtime=5*u.day, dt_scale=4, r_min=21.5*u.solRad)
model.solve([])
fig, ax = HA.plot(model, 1.5*u.day)
fig.suptitle('WSA/HUXt')

#model = H.HUXt(v_boundary=v_pfss, cr_num=cr, simtime=5*u.day, dt_scale=4, r_min=21.5*u.solRad)
#model.solve([])
#fig, ax = HA.plot(model, 1.5*u.day)
#fig.suptitle('PFSS/HUXt')

#model = H.HUXt(v_boundary=v_dumfric, cr_num=cr, simtime=27*u.day, dt_scale=4, r_min=21.5*u.solRad)
#model.solve([])
#fig, ax = HA.plot(model, 1.5*u.day)
#fig.suptitle('DUMFRIC/HUXt')

#model = H.HUXt(v_boundary=v_cortom, cr_num=cr, simtime=27*u.day, dt_scale=4, r_min=8*u.solRad)
#model.solve([])
#fig, ax = HA.plot(model, 1.5*u.day)
#fig.suptitle('CorTom/HUXt')
# %%

'''Example 5: Animating HUXt simulation output'''
# Set up and run previous example of idealised solar wind with single CME
v_boundary = np.ones(128) * 400 * (u.km/u.s)
v_boundary[30:50] = 600 * (u.km/u.s)
v_boundary[95:125] = 500 * (u.km/u.s)

# Add a CME
cme = H.ConeCME(t_launch=1*u.day, longitude=360*u.deg, latitude=0*u.deg, width=30*u.deg, v=1000*(u.km/u.s), thickness=5*u.solRad)

# Setup HUXt to do a 5-day simulation, with model output every 4 timesteps (roughly half and hour time step)
model = H.HUXt(v_boundary=v_boundary, latitude=0*u.deg, simtime=6*u.day, dt_scale=4)
model.solve([cme], tag='cone_cme_test')

# Create the animation
#==========================
HA.animate(model, tag='cone_cme_test') # This takes about two minutes.

# %%
