from jd_natconst import *
from jd_readdata import *
from jd_plot import *

working_dir         = os.getcwd()
dat_dir_migration   = working_dir+"/data/migration/"
fig_dir             = working_dir+"/figures/"

import subprocess

q = readdata()

df_gapmap = pd.read_csv(dat_dir_migration+"gapmap.dat")

def find_nearest(array, value):
    array   = np.asarray(array)
    idx     = (np.abs(array - value)).argmin()
    return array[idx], idx


# Plot settings
lw      = 2.0
fscale  = 2.0
fsize   = 15
fsize_label = fsize + 7

plt.rcParams.update({'font.size': 18})
plt.rcParams['axes.linewidth'] = 1.5 

fig = plt.figure(tight_layout=True, constrained_layout=False, figsize=[10, 11])
gs = fig.add_gridspec(nrows=5, ncols=1, wspace=0.15, hspace=0.15, left=0.055, right=0.98, top=0.99, bottom=0.08)
sns.set(style="ticks")
ax0 = fig.add_subplot(gs[0, 0])
sns.set(style="ticks")
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[2, 0])
ax3 = fig.add_subplot(gs[3, 0])
ax4 = fig.add_subplot(gs[4, 0])

print(np.shape(q.sigma), np.shape(q.r/au))

times = [ 2e+5*year, 3e+5*year, 7e+5*year, 2.5e+6*year, 5e+6*year ]

# Change default color cycle for all new axes
cmap = cm.get_cmap('BuGn', len(times)+1)
cmap_hex = []
for i in range(1, cmap.N):
    rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
    cmap_hex.append(mpl.colors.rgb2hex(rgb))

print(cmap_hex)

mpl.rcParams['axes.prop_cycle'] = cycler(color=cmap_hex)
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

## Pebble flux unit
norm=1e6*year/Mea

print(times)
for i, time in enumerate(times):

    color = colors[i]
    
    time_num, time_idx = find_nearest(q.time, time)

    print(i, time/(1e+6*year), time, time_num, time_idx)

    # Total flux
    ax0.plot(q.r/au, np.abs(q.mflux[time_idx,:])*norm, color=color, lw=lw, label=str(time/(1e+6*year)))
    # Inward flux
    ax1.plot(q.r/au,-1*q.mflux[time_idx,:]*norm, color=color, lw=lw, label=str(time/(1e+6*year)))
    # Outward flux
    ax2.plot(q.r/au, q.mflux[time_idx,:]*norm, color=color, lw=lw, label=str(time/(1e+6*year)))

    ax3.plot(q.r/au, q.dustsize[time_idx,:], color=color, lw=lw, label=str(time/(1e+6*year)))

    ax4.plot(q.r/au, q.stokesnr[time_idx,:], color=color, lw=lw, label=str(time/(1e+6*year)))

sns.set_style("whitegrid")

sns.despine(ax=ax0, bottom=False, top=True, left=False, right=True)
sns.despine(ax=ax1, bottom=False, top=True, left=False, right=True)    

sns.set_style("ticks", {"ax" : ax0, "xtick.major.size":0, "ytick.major.size":0, "xtick.minor.size":0, "ytick.minor.size":0})

xlab_A = 0.02
ylab_A = 0.1
ax0.text(xlab_A, ylab_A, 'A', color="k", rotation=0, ha="left", va="center", fontsize=fsize_label, transform=ax0.transAxes, bbox=dict(fc='white', ec="white", alpha=0.5, pad=0.05, boxstyle='round'))
ax1.text(xlab_A, ylab_A, 'B', color="k", rotation=0, ha="left", va="center", fontsize=fsize_label, transform=ax1.transAxes, bbox=dict(fc='white', ec="white", alpha=0.5, pad=0.05, boxstyle='round'))
ax2.text(xlab_A, ylab_A, 'C', color="k", rotation=0, ha="left", va="center", fontsize=fsize_label, transform=ax2.transAxes, bbox=dict(fc='white', ec="white", alpha=0.5, pad=0.05, boxstyle='round'))
ax3.text(xlab_A, ylab_A, 'D', color="k", rotation=0, ha="left", va="center", fontsize=fsize_label, transform=ax3.transAxes, bbox=dict(fc='white', ec="white", alpha=0.5, pad=0.05, boxstyle='round'))
ax4.text(xlab_A, ylab_A, 'E', color="k", rotation=0, ha="left", va="center", fontsize=fsize_label, transform=ax4.transAxes, bbox=dict(fc='white', ec="white", alpha=0.5, pad=0.05, boxstyle='round'))

xlab_txt = xlab_A+0.03
ax0.text(xlab_txt, ylab_A, 'Total dust flux', color="k", rotation=0, ha="left", va="center", fontsize=fsize, transform=ax0.transAxes, bbox=dict(fc='white', ec="white", alpha=0.5, pad=0.1, boxstyle='round'))
ax1.text(xlab_txt, ylab_A, 'Inward dust flux', color="k", rotation=0, ha="left", va="center", fontsize=fsize, transform=ax1.transAxes, bbox=dict(fc='white', ec="white", alpha=0.5, pad=0.1, boxstyle='round'))
ax2.text(xlab_txt, ylab_A, r'Outward dust flux', color="k", rotation=0, ha="left", va="center", fontsize=fsize, transform=ax2.transAxes, bbox=dict(fc='white', ec="white", alpha=0.5, pad=0.1, boxstyle='round'))
ax3.text(xlab_txt, ylab_A, r'Dust grain size', color="k", rotation=0, ha="left", va="center", fontsize=fsize, transform=ax3.transAxes, bbox=dict(fc='white', ec="white", alpha=0.5, pad=0.1, boxstyle='round'))
ax4.text(xlab_txt, ylab_A, r'Stokes number', color="k", rotation=0, ha="left", va="center", fontsize=fsize, transform=ax4.transAxes, bbox=dict(fc='white', ec="white", alpha=0.5, pad=0.1, boxstyle='round'))

xleft  = 0.5
xright = 300

for ax in [ ax0, ax1, ax2, ax3, ax4 ]:
    ax.set_xlim(left=xleft, right=xright)
    ax.set_xscale("log")
    ax.set_yscale("log")
    sns.despine(ax=ax, bottom=False, left=False, top=True, right=True)
    ax.tick_params(axis='both', which='major', labelsize=fsize-2)
    ax.tick_params(axis='both', which='minor', labelsize=fsize-2)

ax0.set_ylim(bottom=1e-4)
ax1.set_ylim(bottom=1e-4)
ax2.set_ylim(bottom=1e-4)
ax3.set_ylim(bottom=1e-4, top=1e2)
ax4.set_ylim(bottom=1e-4, top=1e0)

for ax in [ ax0, ax1, ax2, ax3 ]:
    ax.set_xticklabels([])

xticks = [0.5, 1, 3, 10, 30, 100, 300]
xticklabels = [ str(i) for i in xticks ]
ax4.set_xticks(xticks)
ax4.set_xticklabels(xticklabels)

ax0.set_ylabel(r'$\.m_\mathrm{tot}$ ($M_\mathrm{Earth}$ Myr$^{-1}$)', fontsize=fsize)
ax1.set_ylabel(r'$\.m_\mathrm{in}$ ($M_\mathrm{Earth}$ Myr$^{-1}$)', fontsize=fsize)
ax2.set_ylabel(r'$\.m_\mathrm{out}$ ($M_\mathrm{Earth}$ Myr$^{-1}$)', fontsize=fsize)
ax3.set_ylabel(r'$d_\mathrm{dust}$ (cm)', fontsize=fsize)
ax4.set_ylabel(r'St (non-dim.)', fontsize=fsize)

ax4.set_xlabel(r'Orbital distance, $r$ (au)', fontsize=fsize)

time_legend = ax1.legend(loc=4, title="Time (Myr)", ncol=1, fontsize=fsize, fancybox=True, framealpha=0.9)
plt.setp(time_legend.get_title(), fontsize=fsize)

plt.savefig(fig_dir+'fig_s3.pdf', bbox_inches='tight')
