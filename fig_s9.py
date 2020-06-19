from jd_natconst import *
from jd_readdata import *
from jd_plot import *

q = readdata()

df_gapmap = pd.read_csv(dat_dir_migration+"gapmap.dat")

# Plot settings
lw      = 1
fscale  = 2.0
fsize   = 20

lw_mig = 1.5
mig_col_resI        = qred
mig_col_resI_lost   = qred_dark
mig_col_resII       = qblue
mig_col_resII_lost  = qblue_dark

fig = plt.figure(tight_layout=True, constrained_layout=False, figsize=[13, 14])
gs = fig.add_gridspec(nrows=3, ncols=1, wspace=0.15, hspace=0.10, left=0.055, right=0.98, top=0.99, bottom=0.08)
sns.set(style="ticks")
ax0 = fig.add_subplot(gs[0, 0])
sns.set(style="ticks")
ax1 = fig.add_subplot(gs[1, 0])
sns.set(style="ticks")
ax2 = fig.add_subplot(gs[2, 0])

for idx, mig_file_name in enumerate([ '750000.0yr1.0Mearth10au', '750000.0yr10.0Mearth10au', '750000.0yr100.0Mearth10au' ]):
    
    migration_track   = np.loadtxt(dat_dir_migration+mig_file_name+'.dat', delimiter=",")
    planet_time  = [float(x/1e+6) for x in migration_track[0]]
    planet_orbit = [float(x) for x in migration_track[1]]

    if idx == 0:
        time_print  = str(float(mig_file_name.split(".")[0])/1e+6)
        mass_print  = str(mig_file_name.split("yr")[1].split("Mearth")[0])
        orbit_print = str(mig_file_name.split("Mearth")[1].split("au")[0])
        print("Time:", time_print)
        print("Mass mars:", mass_print)
        print("au:", orbit_print)
        ax0.scatter(planet_time[0], planet_orbit[0], color=qred_dark, zorder=10, label=time_print+" Myr, "+orbit_print+" au", s=100)

    ax0.plot(planet_time, planet_orbit, color=qred, linestyle='-', lw=lw_mig, zorder=9)

ax0.text(0.68, 1, r'100 $M_\mathrm{Earth}$', color=qred, rotation=0, ha="center", va="center", fontsize=fsize-7, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
ax0.text(0.88, 2, r'10 $M_\mathrm{Earth}$', color=qred, rotation=0, ha="center", va="center", fontsize=fsize-7, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
ax0.text(1, 5.7, r'1 $M_\mathrm{Earth}$', color=qred, rotation=0, ha="center", va="center", fontsize=fsize-7, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))

for idx, mig_file_name in enumerate([ '1300000.0yr10.0Mearth17au', '1300000.0yr100.0Mearth17au', '1300000.0yr1.0Mearth17au' ]):
    
    migration_track   = np.loadtxt(dat_dir_migration+mig_file_name+'.dat', delimiter=",")
    planet_time  = [float(x/1e+6) for x in migration_track[0]]
    planet_orbit = [float(x) for x in migration_track[1]]

    if idx == 0:
        time_print  = str(float(mig_file_name.split(".")[0])/1e+6)
        mass_print  = str(mig_file_name.split("yr")[1].split("Mearth")[0])
        orbit_print = str(mig_file_name.split("Mearth")[1].split("au")[0])
        print("Time:", time_print)
        print("Mass mars:", mass_print)
        print("au:", orbit_print)
        ax0.scatter(planet_time[0], planet_orbit[0], color=qblue_dark, zorder=10, label=time_print+" Myr, "+orbit_print+" au", s=100)
    ax0.plot(planet_time, planet_orbit, color=qblue_light, linestyle='-', lw=lw_mig, zorder=9)

ax0.text(1.17, 0.8, r'100 $M_\mathrm{Earth}$', color=qblue_light, rotation=0, ha="center", va="center", fontsize=fsize-7, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
ax0.text(1.51, 1.5, r'10 $M_\mathrm{Earth}$', color=qblue_light, rotation=0, ha="center", va="center", fontsize=fsize-7, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
ax0.text(1.9, 3, r'1 $M_\mathrm{Earth}$', color=qblue_light, rotation=0, ha="center", va="center", fontsize=fsize-7, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))

for idx, mig_file_name in enumerate([ '2500000.0yr10.0Mearth80au', '2500000.0yr50.0Mearth80au' ]):
    
    migration_track   = np.loadtxt(dat_dir_migration+mig_file_name+'.dat', delimiter=",")
    planet_time  = [float(x/1e+6) for x in migration_track[0]]
    planet_orbit = [float(x) for x in migration_track[1]]

    if idx == 0:
        time_print  = str(float(mig_file_name.split(".")[0])/1e+6)
        mass_print  = str(mig_file_name.split("yr")[1].split("Mearth")[0])
        orbit_print = str(mig_file_name.split("Mearth")[1].split("au")[0])
        print("Time:", time_print)
        print("Mass mars:", mass_print)
        print("au:", orbit_print)
        ax0.scatter(planet_time[0], planet_orbit[0], color=qgreen_dark, zorder=10, label=time_print+" Myr, "+orbit_print+" au", s=100)
    ax0.plot(planet_time, planet_orbit, color=qgreen, linestyle='-', lw=lw_mig, zorder=9)

ax0.text(2.32, 8, r'50 $M_\mathrm{Earth}$', color=qgreen_light, rotation=0, ha="center", va="center", fontsize=fsize-7, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
ax0.text(2.97, 6, r'10 $M_\mathrm{Earth}$', color=qgreen_light, rotation=0, ha="center", va="center", fontsize=fsize-7, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))

for idx, mig_file_name in enumerate([ '2000000.0yr150.0Mearth90au' ]):
    
    migration_track   = np.loadtxt(dat_dir_migration+mig_file_name+'.dat', delimiter=",")
    planet_time  = [float(x/1e+6) for x in migration_track[0]]
    planet_orbit = [float(x) for x in migration_track[1]]

    if idx == 0:
        time_print  = str(float(mig_file_name.split(".")[0])/1e+6)
        mass_print  = str(mig_file_name.split("yr")[1].split("Mearth")[0])
        orbit_print = str(mig_file_name.split("Mearth")[1].split("au")[0])
        print("Time:", time_print)
        print("Mass mars:", mass_print)
        print("au:", orbit_print)
        ax0.scatter(planet_time[0], planet_orbit[0], color=qmagenta_dark, zorder=10, label=time_print+" Myr, "+orbit_print+" au", s=100)
        ax0.scatter(planet_time[-1], planet_orbit[-1]-1.5, marker='s', color=qmagenta, zorder=10)
    ax0.plot(planet_time, planet_orbit, color=qmagenta, linestyle='--', lw=lw_mig, zorder=9)
    
ax0.text(1.8, 70, r'150 $M_\mathrm{Earth}$', color=qmagenta, rotation=0, ha="center", va="center", fontsize=fsize-7, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))

legend_ax0 = ax0.legend(loc=3, fontsize=fsize-4, title="Core starting points")
plt.setp(legend_ax0.get_title(), fontsize=fsize-4)

#### RESERVOIR I

mig_mars_mass = [ '400000.0yr0.1Mearth3.0au', '600000.0yr0.1Mearth7.0au', '2000000.0yr0.1Mearth2.0au'  ]

# Interpolate migration corridors
for idx, track_names in enumerate([ ('400000.0yr0.01Mearth3.0au', '400000.0yr0.1Mearth3.0au'), ('600000.0yr0.1Mearth7.0au', '600000.0yr0.2Mearth7.0au') , ('2000000.0yr0.01Mearth2.0au', '2000000.0yr0.2Mearth2.0au') ]):

    track1_name, track2_name = track_names

    if idx == 0:
        mig_col         = qred
        mig_col_dot     = qred_dark
        mig_col_cone    = qred_light
    if idx == 1:
        mig_col         = qblue
        mig_col_dot     = qblue_dark
        mig_col_cone    = qblue_light
    if idx == 2:
        mig_col         = qgreen
        mig_col_dot     = qgreen_dark
        mig_col_cone    = qgreen_light

    migration_track1  = np.loadtxt(dat_dir_migration+track1_name+'.dat', delimiter=",")
    migration_track2  = np.loadtxt(dat_dir_migration+track2_name+'.dat', delimiter=",")

    planet_time1     = [float(x/1e+6) for x in migration_track1[0]]
    planet_orbit1    = [float(x) for x in migration_track1[1]]
    planet_time2     = [float(x/1e+6) for x in migration_track2[0]]
    planet_orbit2    = [float(x) for x in migration_track2[1]]

    mig1 = interpolate.interp1d(planet_time1, planet_orbit1)
    mig2 = interpolate.interp1d(planet_time2, planet_orbit2)

    time_range_interpolated = np.arange(planet_time1[0], 5, 0.01)
    orbit1_interpolated = mig1(time_range_interpolated)
    orbit2_interpolated = mig2(time_range_interpolated)

    ax1.fill_between(time_range_interpolated, orbit1_interpolated, orbit2_interpolated, color=mig_col, alpha=0.5)

    migration_track_mars_mass  = np.loadtxt(dat_dir_migration+mig_mars_mass[idx]+'.dat', delimiter=",")

    print(idx, mig_mars_mass[idx])
    time_print  = str(float(mig_mars_mass[idx].split(".")[0])/1e+6)
    mass_print  = str(mig_mars_mass[idx].split("yr")[1].split("Mearth")[0])
    mass1_print  = str(track1_name.split("yr")[1].split("Mearth")[0])
    mass2_print  = str(track2_name.split("yr")[1].split("Mearth")[0])
    orbit_print = str(mig_mars_mass[idx].split("Mearth")[1].split("au")[0])
    print("Time:", time_print)
    print("Mass mars:", mass_print)
    print("Mass1:", mass1_print)
    print("Mass2:", mass2_print)
    print("au:", orbit_print)

    mass_min = str(np.min([float(mass1_print), float(mass2_print)]))
    mass_max = str(np.max([float(mass1_print), float(mass2_print)]))

    planet_time     = [float(x/1e+6) for x in migration_track_mars_mass[0]]
    planet_orbit    = [float(x) for x in migration_track_mars_mass[1]]

    ax1.scatter(planet_time[0], planet_orbit[0], color=mig_col_dot, zorder=10, s=50, label=time_print+" Myr, "+orbit_print+" au, "+mass_min+r"$\leq m_\mathrm{embryo}\leq$"+mass_max+r" $M_\mathrm{Earth}$")
    ax1.plot(planet_time, planet_orbit, color=mig_col, linestyle='-', lw=lw_mig, zorder=9)

legend_ax1 = ax1.legend(loc=1, fontsize=fsize-4, title="Terrestrial planet migration corridors")
plt.setp(legend_ax1.get_title(), fontsize=fsize-4)

ax1.plot([0.33, 0.36], [0.6, 0.6], color=qgray, linestyle='-', lw=lw_mig, zorder=9)
ax1.text(0.37, 0.6, r'0.1 $M_\mathrm{Earth}$', color=qgray, rotation=0, ha="left", va="center", fontsize=fsize-7, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))

############ MIGRATION
X           = [ float(x) for x in list(df_gapmap.columns.values)[1:]]
Y           = df_gapmap[df_gapmap.columns[0]].tolist()
Y           = [i/1e+6 for i in Y]
df_gapmap   = df_gapmap.drop(columns="T")
Z           = df_gapmap.values*(MS/Mea)
Z = np.ma.array(Z)

x, y            = np.meshgrid(X, Y)
color_levels    = np.round(np.linspace(np.min(Z), Mju/Mea, 100),0)
color_ticks     = [ 13, 75, 150, 230, 318 ]
contour_levels  = [ 100, 150, 300 ][::-1]

##### GAP OPENING CONTOURS
for i in range(0, 3):
    CS = ax0.contourf(y, x, Z, color_levels, extend='max', alpha=1.0, corner_mask=True, cmap="bone")
CS2 = ax0.contour(CS, levels=contour_levels, zorder=5, linestyles=":", linewidths=1, colors=qgray_light)

# Gap opening
manual_loc =[ [4.2, 8], [4.2, 15], [4.2, 40] ]
ax0.clabel(CS2, CS2.levels, inline=True, fmt=r'%1.0f $M_\mathrm{Earth}$', manual=manual_loc, fontsize=fsize-7, inline_spacing=+10.5, rightside_up=True)

# Colorbar ax0
xloc_colorlegend = -0.03
yloc_colorlegend = +0.00
cax = inset_axes(ax0, width='2%', height='35%', loc=4, bbox_to_anchor=(xloc_colorlegend, yloc_colorlegend, 1, 1), bbox_transform=ax0.transAxes, borderpad=1)
clb = plt.colorbar(CS, cax=cax, ticks=color_ticks) # , orientation="horizontal"
clb.ax.tick_params(labelsize=fsize-7, labelrotation=0, color=qgray_light) # , color="white"
clb.outline.set_edgecolor(qgray_light)
cbytick_obj = plt.getp(clb.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color=qgray_light)

clb.ax.text(-3.5, 0.5, 'Gap opening\nthreshold mass\n'+r'($M_\mathrm{Earth}$)', color=qgray_light, rotation=0, ha="center", va="center", fontsize=fsize-5, bbox=dict(fc='white', ec="white", alpha=0.0, pad=0.0, boxstyle='round'))

ax1.plot(q.time/(1e+6*year), q.rsnow/au, color=qturq, linestyle='--', lw=1.5, zorder=1)
ax1.text(0.77, 9.0, r'Snowline', color=qturq, rotation=-26, ha="left", va="center", fontsize=fsize-5, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))

### VISCOSITY VARIATION

### RESERVOIR II
for idx, mig_file_name in enumerate([ '750000.0yr10.0Mearth10.0au', '1300000.0yr10.0Mearth17.0au', '2500000.0yr10.0Mearth80.0au' ]):

    print(mig_file_name)
    
    migration_track   = np.loadtxt(dat_dir_migration+mig_file_name+'.dat', delimiter=",")
    planet_time  = [float(x/1e+6) for x in migration_track[0]]
    planet_orbit = [float(x) for x in migration_track[1]]

    migration_track2   = np.loadtxt(dat_dir_migration+mig_file_name+'0.0001alpha'+'.dat', delimiter=",")
    planet_time2  = [float(x/1e+6) for x in migration_track2[0]]
    planet_orbit2 = [float(x) for x in migration_track2[1]]

    migration_track3   = np.loadtxt(dat_dir_migration+mig_file_name+'1e-05alpha'+'.dat', delimiter=",")
    planet_time3  = [float(x/1e+6) for x in migration_track3[0]]
    planet_orbit3 = [float(x) for x in migration_track3[1]]

    time_print  = str(float(mig_file_name.split(".")[0])/1e+6)
    mass_print  = str(mig_file_name.split("yr")[1].split("Mearth")[0])
    orbit_print = str(mig_file_name.split("Mearth")[1].split("au")[0])
    print("Time:", time_print)
    print("Mass mars:", mass_print)
    print("au:", orbit_print)
    ax2.scatter(planet_time[0], planet_orbit[0], color=qblue_dark, zorder=20, label=time_print+" Myr, "+orbit_print+" au", s=100)

    ax2.plot(planet_time, planet_orbit, color=qblue, linestyle='-', lw=lw_mig, zorder=9)
    ax2.plot(planet_time2, planet_orbit2, color=qblue_light, linestyle='--', lw=lw_mig, zorder=10)
    ax2.plot(planet_time3, planet_orbit3, color=qblue_dark, linestyle=':', lw=lw_mig, zorder=11)

ax2.text(0.75, 13, r'10 $M_\mathrm{Earth}$', color=qblue_dark, rotation=0, ha="center", va="center", fontsize=fsize-7, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
ax2.text(1.3, 23, r'10 $M_\mathrm{Earth}$', color=qblue_dark, rotation=0, ha="center", va="center", fontsize=fsize-7, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
ax2.text(2.25, 80, r'10 $M_\mathrm{Earth}$', color=qblue_dark, rotation=0, ha="center", va="center", fontsize=fsize-7, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))

### RESERVOIR I
for idx, mig_file_name in enumerate([ '400000.0yr0.1Mearth3.0au', '600000.0yr0.2Mearth7.0au', '2000000.0yr0.2Mearth2.0au']):

    print(mig_file_name)
    
    migration_track   = np.loadtxt(dat_dir_migration+mig_file_name+'.dat', delimiter=",")
    planet_time  = [float(x/1e+6) for x in migration_track[0]]
    planet_orbit = [float(x) for x in migration_track[1]]

    migration_track2   = np.loadtxt(dat_dir_migration+mig_file_name+'0.0001alpha'+'.dat', delimiter=",")
    planet_time2  = [float(x/1e+6) for x in migration_track2[0]]
    planet_orbit2 = [float(x) for x in migration_track2[1]]

    migration_track3   = np.loadtxt(dat_dir_migration+mig_file_name+'1e-05alpha'+'.dat', delimiter=",")
    planet_time3  = [float(x/1e+6) for x in migration_track3[0]]
    planet_orbit3 = [float(x) for x in migration_track3[1]]

    time_print  = str(float(mig_file_name.split(".")[0])/1e+6)
    mass_print  = str(mig_file_name.split("yr")[1].split("Mearth")[0])
    orbit_print = str(mig_file_name.split("Mearth")[1].split("au")[0])
    print("Time:", time_print)
    print("Mass mars:", mass_print)
    print("au:", orbit_print)
    ax2.scatter(planet_time[0], planet_orbit[0], color=qred_dark, zorder=20, label=time_print+" Myr, "+orbit_print+" au", s=50)

    ax2.plot(planet_time, planet_orbit, color=qred, linestyle='-', lw=lw_mig, zorder=9)
    ax2.plot(planet_time2, planet_orbit2, color=qred_light, linestyle='--', lw=lw_mig, zorder=10)
    ax2.plot(planet_time3, planet_orbit3, color=qred_dark, linestyle=':', lw=lw_mig, zorder=11)

ax2.text(0.36, 3, r'0.1 $M_\mathrm{Earth}$', color=qred_dark, rotation=0, ha="center", va="center", fontsize=fsize-7, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
ax2.text(0.54, 7, r'0.2 $M_\mathrm{Earth}$', color=qred_dark, rotation=0, ha="center", va="center", fontsize=fsize-7, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
ax2.text(1.8, 2, r'0.2 $M_\mathrm{Earth}$', color=qred_dark, rotation=0, ha="center", va="center", fontsize=fsize-7, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))

l1, = ax2.plot(0, 0, color=qgray, linestyle='-', lw=lw_mig, zorder=9, label=r"$\alpha_\mathrm{mig}/\alpha_\mathrm{v}$ = 1")
l2, = ax2.plot(0, 0, color=qgray, linestyle='--', lw=lw_mig, zorder=9, label=r"$\alpha_\mathrm{mig}/\alpha_\mathrm{v}$ = 0.1")
l3, = ax2.plot(0, 0, color=qgray, linestyle=':', lw=lw_mig, zorder=9, label=r"$\alpha_\mathrm{mig}/\alpha_\mathrm{v}$ = 0.01")
legend_ax2 = ax2.legend(loc=1, fontsize=fsize-4, handles=[l1,l2,l3], title="Migration viscosity")
plt.setp(legend_ax2.get_title(), fontsize=fsize-4)

ax0.set_xscale("log")
ax0.set_yscale("log")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax2.set_xscale("log")
ax2.set_yscale("log")

xticks = [np.min(y), 0.5, 0.7, 1, 2, 3, 5]
xticklabels = [str(round(i,1)) for i in xticks]
ax0.set_xlim(left=np.min(xticks), right=np.max(xticks))
ax0.set_xticks( xticks )
ax0.set_xticklabels([])
ax1.set_xlim(left=np.min(xticks), right=np.max(xticks))
ax1.set_xticks( xticks )
ax1.set_xticklabels( [] )
ax2.set_xlim(left=np.min(xticks), right=np.max(xticks))
ax2.set_xticks( xticks )
ax2.set_xticklabels( xticklabels , fontsize=fsize-2 )

yticks = [ 0.5, 1, 3, 10, 30, 100 ]
yticklabels = [str(i) for i in yticks]
ax0.set_ylim(bottom=np.min(yticks), top=np.max(yticks))
ax0.set_yticks(yticks )
ax0.set_yticklabels(yticklabels , fontsize=fsize-2)

yticks = [ 0.5, 1, 2, 3, 5, 10, 20 ]
yticklabels = [str(i) for i in yticks]
ax1.set_ylim(bottom=np.min(yticks), top=np.max(yticks))
ax1.set_yticks(yticks )
ax1.set_yticklabels( yticklabels , fontsize=fsize-2)

yticks = [ 0.5, 1, 3, 10, 30, 100 ]
yticklabels = [str(i) for i in yticks]
ax2.set_ylim(bottom=np.min(yticks), top=np.max(yticks))
ax2.set_yticks(yticks )
ax2.set_yticklabels(yticklabels , fontsize=fsize-2)

sns.despine(ax=ax0, top=True, right=True, left=False, bottom=False)
sns.despine(ax=ax1, top=True, right=True, left=False, bottom=False)
sns.despine(ax=ax2, top=True, right=True, left=False, bottom=False)

ax2.set_xlabel(r"Time in disk model, $t_\mathrm{disk}$ (Myr)", fontsize=fsize+2)
ax0.set_ylabel(r"$r$ (au)", fontsize=fsize+2)
ax1.set_ylabel(r"Orbital distance, $r$ (au)", fontsize=fsize+2)
ax2.set_ylabel(r"$r$ (au)", fontsize=fsize+2)

ax1.scatter([5], [1.524], color=qmagenta_dark, s=80, zorder=20)
ax1.scatter([5], [1.0], color=qmagenta_dark, s=130, zorder=20)
ax1.scatter([5], [0.732], color=qmagenta_dark, s=130, zorder=20)
ax1.scatter([5], [0.39], color=qmagenta_dark, s=70, zorder=20)

yloc_center=0.92
xloc_left=0.01
xloc_left2=xloc_left+0.03
ax0.text(xloc_left, yloc_center, 'A', color="k", rotation=0, ha="left", va="center", fontsize=fsize+8, transform=ax0.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
ax1.text(xloc_left, yloc_center, 'B', color="k", rotation=0, ha="left", va="center", fontsize=fsize+8, transform=ax1.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
ax2.text(xloc_left, yloc_center, 'C', color="k", rotation=0, ha="left", va="center", fontsize=fsize+8, transform=ax2.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
ax0.text(xloc_left2, yloc_center, 'Reservoir II, giant planet cores', color=qgray_dark, rotation=0, ha="left", va="center", fontsize=fsize, transform=ax0.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
ax1.text(xloc_left2, yloc_center, 'Reservoir I, rocky protoplanets', color=qgray_dark, rotation=0, ha="left", va="center", fontsize=fsize, transform=ax1.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))
ax2.text(xloc_left2, yloc_center, 'Viscosity ratio sensitivity', color=qgray_dark, rotation=0, ha="left", va="center", fontsize=fsize, transform=ax2.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))

plt.savefig(fig_dir+'fig_s9.pdf', bbox_inches='tight')
plt.savefig(fig_dir+'fig_s9.jpg', bbox_inches='tight')
