from sfd_functions import *
from jd_natconst import *
from jd_plot import *

df_evolution = pd.read_csv(dat_dir+"sfd_evolution.csv")

# Plot settings
lw      = 2.5
fscale  = 2.0
fsize   = 20

sns.set_style("whitegrid")
sns.set(style="ticks", font_scale=fscale)

imf = "powerlaw"
ls  = "-"

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(13,9))
plt.tight_layout(pad=0.2, w_pad=1, h_pad=0.1)
axes = plt.gca()

ax1 = plt.subplot(111)
axes = plt.gca()

df_current = df_evolution.loc[(df_evolution["imf"] == imf) & (df_evolution["res"] == "resI")]
plt.plot(df_current.time, df_current.Tmean, ls='-', lw=2.5, c=qred, label=r'$T_{\mathrm{mean}}$')
plt.plot(df_current.time, df_current.Tmax, ls='--', lw=2.5, c=qred, label=r'$T_{\mathrm{max}}$')

df_current = df_evolution.loc[(df_evolution["imf"] == imf) & (df_evolution["res"] == "resII")]
plt.plot(df_current.time, df_current.Tmean, ls='-', lw=2.5, c=qblue, label=r'$T_{\mathrm{mean}}$')
plt.plot(df_current.time, df_current.Tmax, ls='--', lw=2.5, c=qblue, label=r'$T_{\mathrm{max}}$')

ax1.set_xscale("log")
ax1.set_yscale("log")

ax1.set_xlim(left=0.1, right=5)
ax1.set_xticks( [0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 3, 5] )
ax1.set_xticklabels( ["0.1", "0.2", "0.3", "0.5", "0.7", "1", "2", "3", "5"], fontsize=fsize-2)

ax1.set_ylim(top=1700, bottom=150)
yticks = [ 150, 273, 573, 673, 1223, 1273, 1639 ]
yticklabels = [ "150", "273", "573", "673", "\n1223", "1273\n", "1639" ]
ax1.set_yticks(yticks )
ax1.set_yticklabels( yticklabels , fontsize=fsize-2)

ax1.set_xlabel(r"Time after CAIs, $\Delta t_\mathrm{CAI}$ (Myr)", fontsize=fsize+3)
ax1.set_ylabel(r"Planetesimal swarm temperature, $<T_{\mathrm{plts}}>$ (K)", fontsize=fsize+2)

t_sol   = 1416.193
t_liq   = 1973.000
t_mo    = 0.40*(t_liq-t_sol)+t_sol
t_cocl  = 0.10*(t_liq-t_sol)+t_sol

xloc = 0.103
ax1.axhline(273.15, color=qgreen_dark, ls='--', lw=1)
ax1.text(xloc, 273+3, r'Water ice melting, $T_\mathrm{hydr}$', color=qgreen_dark, size=fsize-4, va="bottom", ha="left")

ax1.axhline(1223.15, color=qgreen_dark, ls='--', lw=1)
ax1.text(xloc, 1223.15-20, r'Amphibolite decomposition, $T_\mathrm{decomp}$', color=qgreen_dark, size=fsize-4, va="top", ha="left")

ax1.axhline(1273.15, color=qgreen_dark, ls='--', lw=1)
ax1.text(xloc, 1273.15+10, r'Percolative core formation, $T_\mathrm{perc}$', color=qgreen_dark, size=fsize-4, va="bottom", ha="left")

ax1.axhline(t_mo, color=qgreen_dark, ls='--', lw=1)
ax1.text(xloc,t_mo-20, r"Metal rain-out, $T(\varphi_\mathrm{rain})$", color=qgreen_dark, size=fsize-4, va="top", ha="left")

ax1.axhline(573, color=qgray_light, ls=':', lw=1)
ax1.text(xloc,573+50, r'Serpentine breakdown', color=qgray_light, size=fsize-6, va="center", ha="left")
ax1.axhline(673, color=qgray_light, ls=':', lw=1)

ax1.text(0.32, 400, 'Reservoir I', color=qred, size=fsize-2, va="bottom", ha="right")
ax1.text(2.3, 400, 'Reservoir II', color=qblue, size=fsize-2, va="bottom", ha="right")

ax1.text(0.44, 870, r'$T_{\mathrm{max}}$', color=qred, size=fsize-2, va="center", ha="center")
ax1.text(0.65, 870, r'$T_{\mathrm{mean}}$', color=qred, size=fsize-2, va="center", ha="center")

ax1.text(0.82, 330, r'$T_{\mathrm{max}}$', color=qblue, size=fsize-2, va="center", ha="center")
ax1.text(1.17, 330, r'$T_{\mathrm{mean}}$', color=qblue, size=fsize-2, va="center", ha="center")

sns.despine(ax=ax1, top=True, right=True, left=False, bottom=False)

figure_name="fig_s8.pdf"
plt.savefig(fig_dir+figure_name, bbox_inches='tight')
