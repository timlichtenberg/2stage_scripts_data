from jd_natconst import *
from jd_readdata import *
from jd_plot import *

q = readdata()

# Plot settings
lw      = 1.5
fscale  = 2.0
fsize   = 22

fig = plt.figure(tight_layout=True, constrained_layout=False, figsize=[13, 11])
gs = fig.add_gridspec(nrows=5, ncols=2, wspace=0.05, hspace=0.7, left=0.055, right=0.98, top=0.99, bottom=0.08)
sns.set(style="ticks")
ax0 = fig.add_subplot(gs[2:5, 0])
sns.set(style="ticks")
ax1 = fig.add_subplot(gs[2:5, 1])
sns.set(style="ticks")
ax2 = fig.add_subplot(gs[0:2, 0:2])

depth_max_all = 0
Tdisk_max_all = 0
Tdisk_min_all = 3000

color_range = [ qred, qgreen, qblue, qmagenta, qturq, qyellow ]
for col_idx, orbit in enumerate([ 2, 4, 7, 15 ]):
    orbit_num, orbit_idx = find_nearest(q.r/au, orbit)
    print(orbit, col_idx, orbit_num, orbit_idx)

    ax2.plot(q.time/(1e+6*year), q.tmid[:,orbit_idx], color=color_range[col_idx], lw=lw, label=str(orbit))

    Tdisk_max_all = np.max([Tdisk_max_all, np.max(q.tmid[:,orbit_idx]) ])
    Tdisk_min_all = np.min([Tdisk_min_all, np.min(q.tmid[:,orbit_idx]) ])

legend_ax2 = ax2.legend(loc=1, ncol=2, fontsize=fsize-4, title=r"Orbital distance, $r$ (au)")
plt.setp(legend_ax2.get_title(), fontsize=fsize-4)

### Varying radius, max T

rad_list = [ 'tim21/tim21_T_5000000.txt', 'tim24/tim24_T_5000000.txt', 'tim25/tim25_T_5000000.txt', 'tim22/tim22_T_5000000.txt', 'tim26/tim26_T_5000000.txt' ] # 

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

for i, file_name in enumerate(rad_list):

    color = colors[i]

    depth   = [ ]
    tmp     = [ ]
    tmp_max = [ ]

    with open(dat_dir_conduction+file_name) as f:
        for line in f:
            cols = line.split()
            if float(cols[0]) > 0:
                depth.append(float(cols[0])/1e+3)
                tmp.append(float(cols[1]))
                tmp_max.append(float(cols[2]))

    time = round(float(file_name.split(".")[0].split("_")[-1])/(1e+6),1)
    depth_max_all = np.max([depth_max_all, np.max(depth) ])

    # Convert depth/radius
    depth = [ np.max(depth)-x for x in depth ]

    l1, = ax0.plot(tmp_max, depth, color=color, linestyle='-', lw=lw, zorder=9, label=str(int(round(np.max(depth)))))

### Time evolution 100 km radius planetesimals

file_time_list = [ 300000, 600000, 800000, 1000000, 5000000 ] # 'tim15_T_10000000.txt'

# Change default color cycle for all new axes
cmap = cm.get_cmap('RdBu', len(file_time_list)+1)
cmap_hex = []
for i in range(1, cmap.N): cmap_hex.append(mpl.colors.rgb2hex(cmap(i)[:3]))
mpl.rcParams['axes.prop_cycle'] = cycler(color=cmap_hex)
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

for i, file_time in enumerate(file_time_list):

    color = colors[i]

    file_name = 'tim22/tim22_T_'+str(file_time)+'.txt'

    depth   = [ ]
    tmp     = [ ]
    tmp_max = [ ]

    with open(dat_dir_conduction+file_name) as f:
        for line in f:
            cols = line.split()
            if float(cols[0]) > 0:
                depth.append(float(cols[0])/1e+3)
                tmp.append(float(cols[1]))
                tmp_max.append(float(cols[2]))

    time = round(float(file_name.split(".")[0].split("_")[-1])/(1e+6),1)
    depth_max_all = np.max([depth_max_all, np.max(depth) ])

    # Convert depth/radius
    depth = [ np.max(depth)-x for x in depth ]

    l2, = ax1.plot(tmp, depth, color=color, linestyle='-', lw=lw, zorder=9, label=str(time))

ax0.axvline(1223, color=qgreen, lw=lw, ls=":")
ax1.axvline(1223, color=qgreen, lw=lw, ls=":")

ax0.text(1220, 13.0, r'$T_\mathrm{decomp}$', color=qgreen, rotation=90, ha="right", va="center", fontsize=fsize-2, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
ax1.text(1220, 13.0, r'$T_\mathrm{decomp}$', color=qgreen, rotation=90, ha="right", va="center", fontsize=fsize-2, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))

legend_ax0 = ax0.legend(loc=4, fontsize=fsize-4, title=r"$R_\mathrm{P}$ (km)")
plt.setp(legend_ax0.get_title(), fontsize=fsize-4)

legend_ax1 = ax1.legend(loc=4, fontsize=fsize-4, title=r"Time (Myr)")
plt.setp(legend_ax1.get_title(), fontsize=fsize-4)

ax2.set_xscale("log")
ax2.set_yscale("log")

ax0.set_yscale("symlog", linthreshy=30)
ax1.set_yscale("symlog", linthreshy=30)

ax0.set_ylim(top=0, bottom=depth_max_all)
ax1.set_ylim(top=0, bottom=depth_max_all)
ax2.set_ylim(top=Tdisk_max_all)
ax2.set_xlim(left=0.1, right=5)

xticks_time = [0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 3, 5]
xticklabels_time = [str(round(i,1)) for i in xticks_time]
ax2.set_xlim(left=np.min(xticks_time), right=np.max(xticks_time))
ax2.set_xticks( xticks_time )
ax2.set_xticklabels( xticklabels_time )

yticks_tmp = [ 40, 70, 100, 200, 300, 500, 1000, int(Tdisk_max_all) ]
yticklabels_tmp = [str(round(i)) for i in yticks_tmp]
ax2.set_yticks( yticks_tmp )
ax2.set_yticklabels( yticklabels_tmp )

yticks_tmp = [ 0, 3, 10, 20, 30, 100, int(depth_max_all) ]
yticklabels_tmp = [ str(round(i)) for i in yticks_tmp ]
ax0.set_yticks( yticks_tmp )
ax0.set_yticklabels( yticklabels_tmp )
ax1.set_yticks( yticks_tmp )
ax1.set_yticklabels( yticklabels_tmp )

ax1.set_yticklabels( [] )

sns.despine(ax=ax0, top=True, right=True, left=False, bottom=False)
sns.despine(ax=ax1, top=True, right=True, left=False, bottom=False)
sns.despine(ax=ax2, top=True, right=True, left=False, bottom=False)

ax0.set_xlabel(r"Maximum temperature, $T_\mathrm{max}$ (K)", fontsize=fsize+2)
ax0.set_ylabel(r"Depth inside planetesimal, $d_\mathrm{P}$ (km)", fontsize=fsize+2)

ax1.set_xlabel(r"Temperature, $T_\mathrm{P}(t)$ (K)", fontsize=fsize+2)

ax2.set_xlabel(r"Time in disk model, $t_\mathrm{disk}$ (Myr)", fontsize=fsize+2)
ax2.set_ylabel(r"$T_\mathrm{disk}$ (K)", fontsize=fsize+2)

yloc_center=0.06
xloc_left=0.02
xloc_left2=xloc_left+0.06
ax2.text(xloc_left, 0.93, 'A', color="k", rotation=0, ha="left", va="center", fontsize=fsize+8, transform=ax2.transAxes, bbox=dict(fc='white', ec="white", alpha=0.5, pad=0.02, boxstyle='round'), zorder=10)
ax0.text(xloc_left, yloc_center, 'B', color="k", rotation=0, ha="left", va="center", fontsize=fsize+8, transform=ax0.transAxes, bbox=dict(fc='white', ec="white", alpha=0.5, pad=0.02, boxstyle='round'), zorder=10)
ax1.text(xloc_left, yloc_center, 'C', color="k", rotation=0, ha="left", va="center", fontsize=fsize+8, transform=ax1.transAxes, bbox=dict(fc='white', ec="white", alpha=0.5, pad=0.02, boxstyle='round'), zorder=10)
ax2.text(xloc_left2-0.03, 0.93, 'Midplane temperature', color="k", rotation=0, ha="left", va="center", fontsize=fsize-2, transform=ax2.transAxes, bbox=dict(fc='white', ec="white", alpha=0.5, pad=0.1, boxstyle='round'), zorder=10)
ax0.text(xloc_left2, yloc_center, 'Varying planetesimal radius\n'+r"$r$ = 2 au", color="k", rotation=0, ha="left", va="center", fontsize=fsize-2, transform=ax0.transAxes, bbox=dict(fc='white', ec="white", alpha=0.5, pad=0.1, boxstyle='round'), zorder=10)
ax1.text(xloc_left2, yloc_center, 'Change over time\n'+r'$R_\mathrm{P}$ = 100 km, $r$ = 2 au', color="k", rotation=0, ha="left", va="center", fontsize=fsize-2, transform=ax1.transAxes, bbox=dict(fc='white', ec="white", alpha=0.5, pad=0.1, boxstyle='round'), zorder=10)

ax0.tick_params(axis='both', which='major', labelsize=fsize-2)
ax0.tick_params(axis='both', which='minor', labelsize=fsize-2)
ax1.tick_params(axis='both', which='major', labelsize=fsize-2)
ax1.tick_params(axis='both', which='minor', labelsize=fsize-2)
ax2.tick_params(axis='both', which='major', labelsize=fsize-2)
ax2.tick_params(axis='both', which='minor', labelsize=fsize-2)

plt.savefig(fig_dir+'fig_s10.pdf', bbox_inches='tight')
