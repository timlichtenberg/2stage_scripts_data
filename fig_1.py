# Script from: https://github.com/timlichtenberg/2stage_scripts_data
# Part of the combined repository: https://osf.io/e2kfv/
from jd_natconst import *
from jd_readdata import *
from jd_plot import *

q = readdata()

df_gapmap = pd.read_csv(dat_dir_migration+"gapmap.dat")

# Plot settings
lw      = 1
fscale  = 2.0
fsize   = 20

plt.rcParams.update({'font.size': 18})
plt.rcParams['axes.linewidth'] = 1.5 

fig = plt.figure(tight_layout=True, constrained_layout=False, figsize=[10, 9])
gs = fig.add_gridspec(nrows=15, ncols=13, wspace=0.15, hspace=0.15, left=0.055, right=0.98, top=0.99, bottom=0.08)
sns.set(style="ticks")
ax1 = fig.add_subplot(gs[0:15, 0:15])

ax1.set_xlim([0.5,20.])
ax1.set_ylim([1e+5,5e+6])
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'Orbital distance, $r\ (\rm{au})$', fontsize=fsize+2)
ax1.set_ylabel(r'Time, $t\ (\rm{Myr})$', fontsize=fsize+2)
ax1.set_xticks( [0.5, 1, 2, 3, 5, 7, 10, 14, 20] )
ax1.set_xticklabels( ["0.5", "1", "2", "3", "5", "7", "10", "14", "20"], fontsize=fsize-2 )
ax1.set_yticks( [1e+5, 2e+5, 4e+5, 7e+5, 1e+6, 2e+6, 3e+6, 4e+6, 5e+6])
ax1.set_yticklabels(["0.1", "0.2", "0.4", "0.7", "1.0", "2.0", "3.0", "4.0", "5.0"], fontsize=fsize-2)

# cgs color axis units
no_colors       = 255
v               = np.linspace(-33, -26, no_colors, endpoint=True)
v_legend        = np.linspace(-33, -26, 8, endpoint=True)

# Mearth au-2 Myr-1 units
v_nat           = [ (np.e**(i))*((au*au)*(1e+6*year))/Mea for i in v_legend ]
v_nat_mag       = [ math.floor(np.log(number)) for number in v_nat ]
v_nat_mag_str   = [ str(i) for i in v_nat_mag ]
print("Natural units, order of magnitude:", v_nat_mag)

# Snowline
ax1.plot(q.rsnow[26:]/au,q.time[26:]/year, color="#530971", ls='-', lw=2.5, zorder=1)

for i in range(0, 3):
    cont_resI = ax1.contourf(q.r/au, q.time/year, np.log(q.dSigmadt_resI), v, cmap='Reds', interpolation='nearest',zorder=2)
    cont_resII = ax1.contourf(q.r/au, q.time/year, np.log(q.dSigmadt_resII), v, cmap='Blues', interpolation='nearest',zorder=2)

# Colorbar legends
xloc_colorlegend = -0.03
yloc_colorlegend = +0.12

# Res I
cax = inset_axes(ax1, width='35%', height='2%', loc='lower right', bbox_to_anchor=(xloc_colorlegend, yloc_colorlegend, 1, 1), bbox_transform=ax1.transAxes, borderpad=0)
clb = plt.colorbar(cont_resI, cax=cax, ticks=v_legend, orientation="horizontal")
clb.ax.tick_params(labelsize=fsize-7, labelrotation=45)

# Res II
cbaxes2 = inset_axes(ax1, width="35%", height="2%", loc='lower right', bbox_to_anchor=(xloc_colorlegend, yloc_colorlegend+0.0205, 1, 1), bbox_transform=ax1.transAxes, borderpad=0.0)
clb2 = plt.colorbar(cont_resII, cax=cbaxes2, ticks=v_legend, orientation="horizontal")
clb2.ax.tick_params(labelsize=fsize-7, labelrotation=45, width=0, length=0)
clb2.ax.xaxis.set_tick_params(pad=+50)
clb2.ax.set_xticklabels(v_nat_mag_str)
# cbaxes2.xaxis.set_ticks_position('top')

clb2.ax.text(+0.5, +2.5, 'Planetesimal formation rate, '+r'$\mathrm{d}\Sigma_{\rm{plts}}/\mathrm{d}t$', fontsize=fsize-4, rotation=0, va='center', ha="center") 

clb.ax.text(-0.1, -2.0, r'ln $\mathrm{g}\ \mathrm{cm}^{-2}\ \mathrm{s}^{-1}$', fontsize=fsize-7, rotation=0, va='center', ha="right") 

clb2.ax.text(-0.1, -5.0, r'ln $\mathrm{M}_\mathrm{Earth}\ \mathrm{au}^{-2}\ \mathrm{Myr}^{-1}$', fontsize=fsize-7, rotation=0, va='center', ha="right") 

############ MIGRATION
X           = [ float(x) for x in list(df_gapmap.columns.values)[1:]]
Y           = df_gapmap[df_gapmap.columns[0]].tolist()
df_gapmap   = df_gapmap.drop(columns="T")
Z           = df_gapmap.values*(MS/Mea)

### MASK GAPMAP INSIDE SNOWLINE
Z_mask = np.zeros_like(Z, dtype=bool)

# Mask anything inside snowline
for idx_r, val_r in enumerate(X):
    for idx_t, val_t in enumerate(Y):

        # Find closest time of snowline array
        tsnow, idx_tsnow   = find_nearest(q.time, val_t*year)

        # Compare w/ snowline values
        if val_r < q.rsnow[idx_tsnow]/au:
            # print(val_t, val_r, q.rsnow[idx_tsnow]/au)
            Z_mask[idx_t,idx_r] = True

# Mask anything before 0.7 Myr
for idx_t, val_t in enumerate(Y):
    if val_t < 0.7e+6:
        # print(idx_t, val_t)
        Z_mask[idx_t,:] = True

Z = np.ma.array(Z, mask=Z_mask)

### / MASKING

x,y             = np.meshgrid(X, Y)
color_levels    = np.round(np.linspace(np.min(Z), Mju/Mea, 100),0)
color_ticks     = [ 13, 50, 100, 150, 200, 250, 318 ]
contour_levels  = [ 100, 150, 300 ][::-1]


lw_mig = 1.0


# ########## Add migration tracks originating in Res II
# # Lost 10 Mearth embryo
# migration_track   = np.loadtxt(dat_dir_migration+'1300000.0yr10.0Mearth17au.dat', delimiter=",")
# planet_time_10Me  = [float(x) for x in migration_track[0]]
# planet_orbit_10Me = [float(x) for x in migration_track[1]]
# mig_col_resI        = qred_light
# mig_col_resI_lost   = qred_dark
# ax1.plot(planet_orbit_10Me, planet_time_10Me, color=qblue_dark, linestyle='--', lw=lw_mig, zorder=10)
# ax1.plot(17, 1.3e+6, marker='.', markersize=17, color="k", markerfacecolor=qblue_dark, alpha=0.99)
# ax1.text(17.3, 1.15e+6, r'$10 \, M_{\mathrm{Earth}}$', color=qblue_dark, size=fsize-7, rotation=0, va='center', ha='center')

#### MIGRATION CORRIDORS

# Interpolate migration corridors
for idx, track_names in enumerate([ ('350000.0yr0.1Mearth3.0au', '350000.0yr0.01Mearth3.0au'), ('1000000.0yr0.2Mearth5.0au', '1000000.0yr0.1Mearth5.0au') ]):

    track1_name, track2_name = track_names

    if idx == 0 or idx == 1:
        mig_col         = qgray_dark
        mig_col_dot     = qgray_dark
        mig_col_cone    = qgray_light
    if idx == 2:
        mig_col         = qgreen
        mig_col_dot     = qgreen_dark
        mig_col_cone    = qgreen_light

    migration_track1  = np.loadtxt(dat_dir_migration+track1_name+'.dat', delimiter=",")
    migration_track2  = np.loadtxt(dat_dir_migration+track2_name+'.dat', delimiter=",")

    planet_time1     = [float(x) for x in migration_track1[0]]
    planet_orbit1    = [float(x) for x in migration_track1[1]]
    planet_time2     = [float(x) for x in migration_track2[0]]
    planet_orbit2    = [float(x) for x in migration_track2[1]]

    mig1 = interpolate.interp1d(planet_time1, planet_orbit1)
    mig2 = interpolate.interp1d(planet_time2, planet_orbit2)

    min_time = np.max([planet_time1[0], planet_time2[0]])
    time_range_interpolated = np.arange(min_time, 5e+6, 0.01e+6)
    orbit1_interpolated = mig1(time_range_interpolated)
    orbit2_interpolated = mig2(time_range_interpolated)

    ax1.fill_betweenx(time_range_interpolated, orbit1_interpolated, orbit2_interpolated, color=mig_col_cone, alpha=0.3)

    ax1.plot(planet_orbit1, planet_time1, color=mig_col, linestyle='--', lw=lw_mig, zorder=9)
    ax1.plot(planet_orbit2, planet_time2, color=mig_col, linestyle='--', lw=lw_mig, zorder=9)

color_mig = qgray_dark
ax1.text(0.46, 0.58, 'Terrestrial planet\nmigration corridor', color=color_mig, rotation=0, ha="center", va="center", fontsize=fsize-3, transform=ax1.transAxes)
ax1.text(0.40, 0.938, r'$0.1 \, M_{\mathrm{Earth}}$', color=color_mig, size=fsize-7, rotation=-40, va='center', ha='center', transform=ax1.transAxes)
ax1.text(0.40, 0.787, r'$0.2 \, M_{\mathrm{Earth}}$', color=color_mig, size=fsize-7, rotation=-33, va='center', ha='center', transform=ax1.transAxes)
ax1.text(0.41, 0.430, r'$0.01 \, M_{\mathrm{Earth}}$', color=color_mig, size=fsize-7, rotation=-46, va='center', ha='center', transform=ax1.transAxes)
ax1.text(0.40, 0.37, r'$0.1 \, M_{\mathrm{Earth}}$', color=color_mig, size=fsize-7, rotation=-38, va='center', ha='center', transform=ax1.transAxes)

color_dot = qgray_dark
ax1.plot(3.0, 0.35e+6, marker='.', markersize=15, color="k", markerfacecolor=color_dot, alpha=0.99, zorder=20)
ax1.plot(5.0, 1.0e+6, marker='.', markersize=15, color="k", markerfacecolor=color_dot, alpha=0.99, zorder=20)

lwa=1.5

x_arrows = 0.017
x_text   = x_arrows-0.01
ax1.annotate("", xy=(x_arrows, 1.0), xycoords='axes fraction', xytext=(x_arrows, 0.795), textcoords='axes fraction', arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.2", lw=lwa, connectionstyle="arc3", fc=qgray_light, ec=qgray_light), transform=ax1.transAxes)
ax1.text(x_text, 0.745, 'Class II', color=qgray_light, size=fsize, rotation=90, va='center', ha='left', fontsize=fsize-5, transform=ax1.transAxes)
ax1.annotate("", xy=(x_arrows, 0.695), xycoords='axes fraction', xytext=(x_arrows, 0.480), textcoords='axes fraction', arrowprops=dict(arrowstyle="|-|,widthA=0.12,widthB=0.0", lw=lwa, connectionstyle="arc3", fc=qgray_light, ec=qgray_light), transform=ax1.transAxes)
ax1.annotate("", xy=(x_arrows, 0.47), xycoords='axes fraction', xytext=(x_arrows, 0.280), textcoords='axes fraction', arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.2", lw=lwa, connectionstyle="arc3", fc=qgray_light, ec=qgray_light), transform=ax1.transAxes)
ax1.text(x_text, 0.236, 'Class I', color=qgray_light, size=fsize, rotation=90, va='center', ha='left', fontsize=fsize-5, transform=ax1.transAxes)
ax1.annotate("", xy=(x_arrows, 0.19), xycoords='axes fraction', xytext=(x_arrows, 0.001), textcoords='axes fraction', arrowprops=dict(arrowstyle="|-|,widthA=0.0,widthB=0.0", lw=lwa, connectionstyle="arc3", fc=qgray_light, ec=qgray_light), transform=ax1.transAxes)

# ##### GAP OPENING CONTOURS (cf. Fig. S9A)
# CS              = ax1.contourf(x, y, Z, color_levels, extend='max', alpha=0.0, corner_mask=True)
# contour_colors  = sns.color_palette("Purples_r", len(contour_levels))
# CS2             = ax1.contour(CS, levels=contour_levels, colors=contour_colors, zorder=5)

# ax1.text(0.80, 1.003, 'Disk gap opening thresholds', color=purples[5], rotation=0, ha="center", va="bottom", fontsize=fsize-5, transform=ax1.transAxes)
# manual_loc =[ [4.5,4e6], [5.5,1.5e6], [10.7,0.9e6] ]
# ax1.clabel(CS2, CS2.levels, inline=True, fmt=r'%1.0f $M_\mathrm{Earth}$', manual=manual_loc, fontsize=fsize-5)


########## RESERVOIRS
ax1.text(0.88, 0.17e+6, r'Water'+'\nsnow line', color="#530971", size=fsize, rotation=0, va='center', ha='center', fontsize=fsize-3)

resI_txt = ax1.text(3.65, 3.8e+5, 'Reservoir I', color=reds[4], rotation=0, ha="left", fontsize=fsize+4)
resII_txt = ax1.text(8.2, 2.0e+6, 'Reservoir II', color=blues[6], rotation=0, ha="center", fontsize=fsize+4)

for text in [ resI_txt, resII_txt ]:
    text.set_path_effects([path_effects.Stroke(linewidth=1.0, foreground='white'),
                       path_effects.Normal()])

sns.set_style("whitegrid")
sns.despine(ax=ax1, bottom=False, top=True, left=False, right=True)

for figure_extension in [ ".pdf", ".jpg" ]:
    figure_name="fig_1" + figure_extension
    plt.savefig(image_dir+figure_name, bbox_inches='tight', dpi=300)

