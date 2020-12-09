# Script from: https://github.com/timlichtenberg/2stage_scripts_data
# Part of the combined repository: https://osf.io/e2kfv/
from jd_natconst import *
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib
import os, math, sys, glob, struct
import numpy as np 
from natsort import natsorted
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker, VPacker
import pickle
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors
from numpy.random import uniform, seed
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
from shutil import copyfile
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from matplotlib.ticker import NullFormatter
import astropy
from astropy import stats as astrostats
import pandas as pd
import scipy
from scipy.interpolate import griddata
import csv
import matplotlib.patheffects as path_effects
from natsort import natsorted
from cycler import cycler
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import interpolate

working_dir         = os.getcwd()
dat_dir             = working_dir+"/data/planetesimal_evolution/"
fig_dir             = working_dir+"/figures/"
dat_dir_migration   = working_dir+"/data/migration/"
dat_dir_conduction  = working_dir+"/data/surface_conduction/"

qgray       = "#768E95"
qblue       = "#4283A9"
qgreen      = "#62B4A9"
qred        = "#E6767A"
qturq       = "#2EC0D1"
qmagenta    = "#9A607F"
qyellow     = "#EBB434"
qgray_dark  = "#465559"
qblue_dark  = "#274e65"
qgreen_dark = "#3a6c65"
qred_dark   = "#b85e61"
qturq_dark  = "#2499a7"
qmagenta_dark = "#4d303f"
qyellow_dark  = "#a47d24"
qgray_light  = "#acbbbf"
qblue_light  = "#8db4cb"
qgreen_light = "#a0d2cb"
qred_light   = "#eb9194"
qturq_light  = "#57ccda"
qmagenta_light = "#c29fb2"
qyellow_light = "#f1ca70"
greys   = sns.color_palette("Greys", 7)
blues   = sns.color_palette("Blues", 7)
reds    = sns.color_palette("Reds", 7)
greens  = sns.color_palette("Greens", 7)
purples = sns.color_palette("Purples", 7)

params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
plt.rcParams.update(params)

def save_obj(obj, name):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

def plot(f,x=None,oplot=None):
    if oplot is None:
        fig = plt.figure()
        ax  = fig.add_subplot(111)
    else:
        fig = None
        ax  = None
    if x is None:
        nx = f.shape[0]
        x  = np.linspace(0,nx-1,nx)
    plt.plot(x,f)
    return fig,ax

def oplot(f,x=None):
    plot(f,x=x,oplot=True)

def surface(f,x=None,y=None,rstride=None,cstride=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    nx = f.shape[0]
    ny = f.shape[1]
    if x is None:
        x = np.linspace(0,nx-1,nx)
    if y is None:
        y = np.linspace(0,ny-1,ny)
    if rstride is None:
        rstride = 1
    if cstride is None:
        cstride = 1
    xx, yy = np.meshgrid(x, y,indexing='ij')
    ax.plot_wireframe(xx, yy, f, rstride=rstride, cstride=cstride)
    return fig,ax

def find_nearest(array, value):
    array   = np.asarray(array)
    idx     = (np.abs(array - value)).argmin()
    return array[idx], idx

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def image(f,x=None,y=None,aspect='auto',cmap=None,range=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if x is None:
        nx = f.shape[1]
        x  = np.array((0,nx-1))
    if y is None:
        ny = f.shape[0]
        y  = np.array((0,ny-1))
    extent = (x[0],x[len(x)-1],y[0],y[len(y)-1])
    if range is None:
        ff = f.copy()
    else:
        ff = f.copy()
        mi = range[0]
        ma = range[1]
        if mi > ma:
            ma = range[0]
            mi = range[1]
        ff[ff>ma] = ma
        ff[ff<mi] = mi
    plt.imshow(ff,origin='lower',aspect=aspect,extent=extent,cmap=cmap)
    return fig,ax
