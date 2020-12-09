# Script from: https://github.com/timlichtenberg/2stage_scripts_data
# Part of the combined repository: https://osf.io/e2kfv/
import matplotlib
import os
import sys
import glob
import struct
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker, VPacker
import pickle
import math
import random
import datetime
import time
import operator
from matplotlib import colors
import matplotlib.ticker as ticker
from scipy import stats
from scipy import signal
import matplotlib.pyplot as plt
import astropy
from astropy import stats as astrostats
import pandas as pd
import scipy
from scipy.interpolate import griddata
import csv
import datetime
import seaborn as sns 
from natsort import natsorted

def save_obj(obj, name):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
plt.rcParams.update(params)

### WORKING DIRECTORIES
working_dir  = os.getcwd()
dat_dir      = working_dir+"/data/planetesimal_evolution/"
csv_dir      = dat_dir+"/sfd_data/"
image_dir    = working_dir+"/figures/"
fig_dir      = image_dir

## Constant definitions
T_sol       = 1416.193      # K
T_liq       = 1973.000      # K
H_L         = 400000        # J kg-1
c_p         = 1000          # J kg-1 K-1
G0          = 6.67408e-11   # m3 kg-1 s-2
rho_vol_ice = 1000          # kg m-3
rho_sil_sol = 3500          # kg m-3
rho_sil_liq = 2900          # kg m-3
rho_sil_ice = 2750          # kg m-3, 70% silicates, 30% water ice, 3500*0.7+1000*0.3
M_ceres     = 9.393e+20     # kg
M_earth     = 5.97237e24    # kg

### FUNCTION DEFINITIONS

def normalise_dict(d):
    factor=1.0/math.fsum(d.itervalues())
    for k in d:
        d[k] = d[k]*factor
    key_for_max = max(d.iteritems(), key=operator.itemgetter(1))[0]
    diff = 1.0 - math.fsum(d.itervalues())
    d[key_for_max] += diff

def round10(x):
    return int(round(x / 10.0) * 10)

def generate_radius_powerlaw(random_number, slope, r_min):
    
    radius_planetesimal = r_min * math.pow( ( 1 - random_number ), ( - 1. / ( slope - 1. ) ) )

    return round(radius_planetesimal,2)

def generate_mass_powerlaw(random_number, slope, m_min):
    
    m_plts = m_min * math.pow( ( 1 - random_number ), ( - 1. / ( slope - 1. ) ) )

    return round(m_plts,2)

def calc_mass(rad_plts):
    
    mass_planetesimal = rho_sil_ice * (4./3.) * np.pi *  ((rad_plts*1000.) ** 3.)

    return round(mass_planetesimal,2) # kg

def calc_radius(mass_plts):
    
    rad_plts = ((mass_plts / (rho_sil_ice * (4./3.) * np.pi))  ** (1./3.)) / 1000.

    return round(rad_plts,2) # km

def calc_volume(rad_plts):
    
    vol_plts = (4./3.) * np.pi * ( (rad_plts * 1000.) ** 3.)

    return round(vol_plts,2) # m^3

mass_func   = np.vectorize(calc_mass)
rad_func    = np.vectorize(calc_radius)
vol_func    = np.vectorize(calc_volume)
gen_mass    = np.vectorize(generate_mass_powerlaw) 

def calc_mass_bins(rad_plts, no_plts):
    
    mass_planetesimal = rho_sil_ice * (4./3.) * np.pi *  ((rad_plts*1000.) ** 3.)
    mass_planetesimals = mass_planetesimal * no_plts

    return mass_planetesimals # kg
mass_bin_func = np.vectorize(calc_mass_bins)

def normalise_dict(d):
    factor=1.0/math.fsum(d.itervalues())
    for k in d:
        d[k] = d[k]*factor
    key_for_max = max(d.iteritems(), key=operator.itemgetter(1))[0]
    diff = 1.0 - math.fsum(d.itervalues())
    d[key_for_max] += diff

def bilinear_1d( radius_x, radius_a, radius_b, quantity_a, quantity_b ):

    # Calculate 1D weight
    weight_ab = ( ( radius_x - radius_a ) / ( radius_b - radius_a ) )

    # Interpolate quantity ( e.g., chondrules, matrix or forbidden )
    quantity_x = quantity_a * ( 1. - weight_ab ) + quantity_b * weight_ab

    # print radius_x, radius_a, radius_b, weight_ab, quantity_a, quantity_b, quantity_x

    return quantity_x

def bilinear_2d( r_a, r_b, t_a, t_b, quantity_a, quantity_b, quantity_c, quantity_d, r_x, t_x ):

    vol_a_tot = ( 4. / 3. ) * math.pi * math.pow( r_a, 3. )
    vol_b_tot = ( 4. / 3. ) * math.pi * math.pow( r_b, 3. )
    vol_x_tot = ( 4. / 3. ) * math.pi * math.pow( r_x, 3. )

    dv_m    = vol_x_tot - vol_a_tot
    dv      = vol_b_tot - vol_a_tot
    dv_norm = dv_m / dv

    dt_m    = t_x - t_a
    dt      = t_b - t_a

    if dt > 0.0:
        dt_norm = dt_m / dt
        quantity_interpolated = quantity_a * ( 1. - dv_norm ) * ( 1. - dt_norm ) + quantity_b * dv_norm * ( 1. - dt_norm ) + quantity_c * ( 1. - dv_norm ) * dt_norm + quantity_d * dv_norm * dt_norm
    else:
        quantity_x = quantity_a * ( 1. - dv_norm ) + quantity_b * dv_norm

    return quantity_x

def generate_planetesimal_family(random_reals_mass, imf_slope, mass_min, mass_max, mass_mean, total_mass, preoverflow_powerlaw, preoverflow_gaussian):
    
    # Generate random masses with given slope
    mass_family_powerlaw = []

    # Prevent negative masses from overflow mass
    if total_mass + preoverflow_powerlaw < 0.:

        mass_family_powerlaw = [ 0. ]
        overflow_powerlaw = total_mass + preoverflow_powerlaw
    else:

        # Add overflow from last timestep
        total_mass_powerlaw = total_mass + preoverflow_powerlaw

        cut_idx     = 0
        mra_cumsum  = []
        mass_family_powerlaw = []
        leftover_mass_powerlaw = total_mass_powerlaw

        # Make sure that random number array contains enough total mass
        while leftover_mass_powerlaw > calc_mass(100):

            # Generate random numbers and masses
            random_reals_mass       = np.random.rand(np.size(random_reals_mass),1)
            mass_randoms_powerlaw   = gen_mass(random_reals_mass, imf_slope, mass_min)
            mass_randoms_powerlaw   = mass_randoms_powerlaw[mass_randoms_powerlaw >= mass_min]
            mass_randoms_powerlaw   = mass_randoms_powerlaw[mass_randoms_powerlaw <= mass_max]

            # Fill mass array
            if leftover_mass_powerlaw >= np.sum(mass_randoms_powerlaw):
                mass_family_powerlaw.extend(mass_randoms_powerlaw)
                leftover_mass_powerlaw = leftover_mass_powerlaw - np.sum(mass_randoms_powerlaw)
            else:
                mra_cumsum              = np.cumsum(mass_randoms_powerlaw)
                cut_idx                 = (np.abs(mra_cumsum - leftover_mass_powerlaw)).argmin()
                mass_family_powerlaw.extend(mass_randoms_powerlaw[:cut_idx])
                leftover_mass_powerlaw = leftover_mass_powerlaw - np.sum(mass_randoms_powerlaw[:cut_idx])

        # Update overflow mass
        overflow_powerlaw = total_mass_powerlaw - np.sum(mass_family_powerlaw)

    # Prevent negative masses from overflow mass
    if total_mass + preoverflow_gaussian < 0:
        
        mass_family_gaussian = [ 0. ]
        overflow_gaussian = total_mass + preoverflow_gaussian
    
    else:

        # Add overflow from last timestep
        total_mass_gaussian = total_mass + preoverflow_gaussian
        leftover_mass_gaussian = total_mass_gaussian

        mass_family_gaussian = []

        while leftover_mass_gaussian > calc_mass(100):

            mass_randoms_gaussian    = np.random.normal(loc=mass_mean, scale=mass_mean*gaussian_width, size=np.size(random_reals_mass))

            # Limit mass family to total mass generated (per time step)
            mass_randoms_gaussian = mass_randoms_gaussian[mass_randoms_gaussian > mass_min]
            mass_randoms_gaussian = mass_randoms_gaussian[mass_randoms_gaussian < mass_max]

            if leftover_mass_gaussian >= np.sum(mass_randoms_gaussian):
                mass_family_gaussian.extend(mass_randoms_gaussian)
                leftover_mass_gaussian = leftover_mass_gaussian - np.sum(mass_randoms_gaussian)
            else:
                mfg_cumsum           = np.cumsum(mass_randoms_gaussian)
                cut_idx              = (np.abs(mfg_cumsum - leftover_mass_gaussian)).argmin()
                mass_family_gaussian.extend(mass_randoms_gaussian[:cut_idx])
                leftover_mass_gaussian = leftover_mass_gaussian - np.sum(mass_randoms_gaussian[:cut_idx])

        # Update overflow mass
        overflow_gaussian = total_mass_gaussian - np.sum(mass_family_gaussian)  

    # Define masses/radii using constant density
    if np.size(mass_family_powerlaw) == 0: # problems with vectorize for small masses
        mass_family_powerlaw = [ 0 ]
    if np.size(mass_family_gaussian) == 0: # problems with vectorize for small masses
        mass_family_gaussian = [ 0 ]
    rad_family_powerlaw = rad_func(mass_family_powerlaw)
    rad_family_gaussian = rad_func(mass_family_gaussian)

    rad_family_powerlaw     = np.asarray(rad_family_powerlaw)
    rad_family_gaussian     = np.asarray(rad_family_gaussian)
    mass_family_powerlaw    = np.asarray(mass_family_powerlaw)
    mass_family_gaussian    = np.asarray(mass_family_gaussian)

    return rad_family_powerlaw, mass_family_powerlaw, overflow_powerlaw, rad_family_gaussian, mass_family_gaussian, overflow_gaussian

# Input parameters
sfd_slope        = 2.8                      # Johansen+ 15 / Simon+ 16,17
imf_slope        = 1.6                      # Johansen+ 15 / Simon+ 16,17 / Abod 19
mass_min         = calc_mass(1)             # kg
mass_max         = calc_mass(300)           # kg
mass_mean        = calc_mass(100)           # kg
mass_cut         = calc_mass(100)           # kg
rad_min          = calc_radius(mass_min)    # km
rad_max          = calc_radius(mass_max)    # km
rad_mean         = calc_radius(mass_mean)   # km
sample_size      = 100000                   # No of distinct planetesimals
gaussian_centre  = 100                      # centre of the gaussian sfd
gaussian_width   = 0.30                     # relative width/spread of the gaussian sfd
total_mass       = 3.*M_ceres               # 3*M_ceres = approx. mass of asteroid belt

# Generate planetesimal family at time t0
random_reals_mass    = np.random.rand(10000000,1)

# Number of planetesimal bins
n_bins = 100