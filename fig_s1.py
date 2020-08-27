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

fig = plt.figure(tight_layout=True, constrained_layout=False, figsize=[9, 7])
gs = fig.add_gridspec(nrows=1, ncols=1, wspace=0.15, hspace=0.15, left=0.055, right=0.98, top=0.99, bottom=0.08)
sns.set(style="ticks")
ax0 = fig.add_subplot(gs[0, 0])

print(np.shape(q.sigma), np.shape(q.r/au))

times = [ 1e+4*year, 1e+5*year, 3e+5*year, 1e+6*year, 3e+6*year, 5e+6*year ] # , 5e+5*year, 

# Change default color cycle for all new axes
cmap = cm.get_cmap('PuBu', len(times)+1)
cmap_hex = []
for i in range(1, cmap.N):
    rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
    cmap_hex.append(mpl.colors.rgb2hex(rgb))

print(cmap_hex)

mpl.rcParams['axes.prop_cycle'] = cycler(color=cmap_hex)
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

ax0.plot(q.time/(1e+6*year), q.mstar/np.max(q.mstar), color=qyellow, lw=lw, label="Protosun")
ax0.plot(q.time/(1e+6*year), q.mdisk/np.max(q.mstar), color=qgreen, lw=lw, label="H/He gas")
ax0.plot(q.time/(1e+6*year), q.mvap/np.max(q.mstar), color=qblue, lw=lw, label=r"H$_2$O gas")
ax0.plot(q.time/(1e+6*year), q.mdust/np.max(q.mstar), color=qred, lw=lw, label="Dust")
ax0.plot(q.time/(1e+6*year), q.mplts/np.max(q.mstar), color=qmagenta, lw=lw, label="Planetesimals, total", zorder=10)
ax0.plot(q.time/(1e+6*year), q.mplts_resI/np.max(q.mstar), color=qmagenta_light, lw=lw, ls="--", label="Reservoir I")
ax0.plot(q.time/(1e+6*year), q.mplts_resII/np.max(q.mstar), color=qmagenta_dark, lw=lw, ls=":", label="Reservoir II")

sns.set_style("whitegrid")

sns.despine(ax=ax0, bottom=False, top=True, left=False, right=True)

ax0.set_xscale("log")
ax0.set_yscale("log")

ax0.set_xlim([1e-2, 5])
ax0.set_ylim([1e-8, 1])

ax0.set_ylabel(r'Mass, $M$ ($M_{\odot}$)', fontsize=fsize)
ax0.set_xlabel(r'Time, $t$ (Myr)', fontsize=fsize)

time_legend = ax0.legend(loc=3, ncol=1, fontsize=fsize)

xticks = [1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 5]
xticklabels = [ str(round(float(i),2)) for i in xticks ]
ax0.set_xticks(xticks)
ax0.set_xticklabels(xticklabels)

ax0.axhline(Mju/np.max(q.mstar), color=qgray_light, ls=':', lw=1)
ax0.axhline(Mea/np.max(q.mstar), color=qgray_light, ls=':', lw=1)
ax0.axhline(Mma/np.max(q.mstar), color=qgray_light, ls=':', lw=1)
ax0.axhline(Mmo/np.max(q.mstar), color=qgray_light, ls=':', lw=1)
ax0.axhline(Mce/np.max(q.mstar), color=qgray_light, ls=':', lw=1)
ax0.text(0.1, Mju/np.max(q.mstar), 'Jupiter', color=qgray_light, size=fsize-4, va="bottom")
ax0.text(0.1, Mea/np.max(q.mstar), 'Earth', color=qgray_light, size=fsize-4, va="bottom")
ax0.text(0.1, Mma/np.max(q.mstar), 'Mars', color=qgray_light, size=fsize-4, va="bottom")
ax0.text(0.1, Mmo/np.max(q.mstar), 'Moon', color=qgray_light, size=fsize-4, va="bottom")

ax0.set_yticks([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0])

ax0.tick_params(axis='both', which='major', labelsize=fsize)
ax0.tick_params(axis='both', which='minor', labelsize=fsize)

plt.savefig(fig_dir+'fig_s1.pdf', bbox_inches='tight')
