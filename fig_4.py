from sfd_functions import *
from jd_natconst import *
from jd_plot import *

df_evolution = pd.read_csv(dat_dir+"sfd_evolution.csv")

# Plot settings
lw      = 2.5
fscale  = 2.0
fsize   = 22
sns.set_style("whitegrid")
sns.set(style="ticks", font_scale=fscale)

imf = "powerlaw"
ls  = "-"

fig  = plt.subplots(figsize=(11,11))
grid = plt.GridSpec(4, 1, wspace=0.0, hspace=0.1) #
ax3  = plt.subplot(grid[0:1, 0])
ax1  = plt.subplot(grid[1:4, 0], sharex=ax3)

# Reservoirs
df_resI   = df_evolution.loc[(df_evolution["imf"] == imf) & (df_evolution["res"] == "resI")]
df_resII  = df_evolution.loc[(df_evolution["imf"] == imf) & (df_evolution["res"] == "resII")]

resI_perco_frac = df_resI.perco_frac.tolist()
resI_melt2_frac = df_resI.melt2_frac.tolist()
resI_m_plts_tot = df_resI.m_plts_tot.tolist()
resII_perco_frac = df_resII.perco_frac.tolist()
resII_melt2_frac = df_resII.melt2_frac.tolist()
resII_m_plts_tot = df_resII.m_plts_tot.tolist()

# Normalize by final reservoir mass
M_ResI  = 1.222764    # M_Earth, at 5.0 Myr
M_ResII = 770.832781  # M_Earth, at 5.0 Myr

for i in range(0, len(df_resI)):
   m_plts_tot = resI_m_plts_tot[i]
   m_frac     = m_plts_tot/M_ResI
   resI_perco_frac[i]   = resI_perco_frac[i]*m_frac
   resI_melt2_frac[i]   = resI_melt2_frac[i]*m_frac

for i in range(0, len(df_resII)):
   m_plts_tot = resII_m_plts_tot[i]
   m_frac     = m_plts_tot/M_ResII
   resII_perco_frac[i]   = resII_perco_frac[i]*m_frac
   resII_melt2_frac[i]   = resII_melt2_frac[i]*m_frac

# Metal-silicate
N = 25
rolling_mean_timeI   = np.convolve(df_resI.time, np.ones((N,))/N, mode='valid')
rolling_mean_timeII  = np.convolve(df_resII.time, np.ones((N,))/N, mode='valid')
resI_perco           = np.convolve(resI_perco_frac, np.ones((N,))/N, mode='valid')
resI_melt2           = np.convolve(resI_melt2_frac, np.ones((N,))/N, mode='valid')
resII_perco          = np.convolve(resII_perco_frac, np.ones((N,))/N, mode='valid')
resII_melt2          = np.convolve(resII_melt2_frac, np.ones((N,))/N, mode='valid')

ax1.plot(rolling_mean_timeI, resI_perco, ls="--", c=qred, lw=lw)
ax1.plot(rolling_mean_timeI, resI_melt2, ls="--", c=qred_dark, lw=lw)
ax1.plot(rolling_mean_timeII, resII_perco, ls="--", c=qblue, lw=lw)
ax1.plot(rolling_mean_timeII, resII_melt2, ls="--", c=qblue_dark, lw=lw)

ax1.plot(rolling_mean_timeI, np.gradient(resI_perco), ls="-", c=qred, lw=lw)
ax1.plot(rolling_mean_timeI, np.gradient(resI_melt2), ls="-", c=qred_dark, lw=lw)
ax1.plot(rolling_mean_timeII, np.gradient(resII_perco), ls="-", c=qblue, lw=lw)
ax1.plot(rolling_mean_timeII, np.gradient(resII_melt2), ls="-", c=qblue_dark, lw=lw)

## Fillings
ax1.fill_between(rolling_mean_timeI, 0, np.gradient(resI_perco), color=qred, alpha=0.5)
ax1.fill_between(rolling_mean_timeI, 0, np.gradient(resI_melt2), color=qred_dark, alpha=0.5)
ax1.fill_between(rolling_mean_timeII, 0, np.gradient(resII_perco), color=qblue, alpha=0.5)
ax1.fill_between(rolling_mean_timeII, 0, np.gradient(resII_melt2), color=qblue_dark, alpha=0.5)

N = 3
rolling_mean_timeI 	= np.convolve(df_resI.time, np.ones((N,))/N, mode='valid')
resI_primitive_frac  = np.convolve(df_resI.primitive_frac, np.ones((N,))/N, mode='valid')
resI_h2o_frac        = np.convolve(df_resI.h2o_frac, np.ones((N,))/N, mode='valid')
resI_hydrous_frac    = np.convolve(df_resI.hydrous_frac, np.ones((N,))/N, mode='valid')

N = 5
rolling_mean_timeII  = np.convolve(df_resII.time, np.ones((N,))/N, mode='valid')
resII_primitive_frac = np.convolve(df_resII.primitive_frac, np.ones((N,))/N, mode='valid')
resII_h2o_frac       = np.convolve(df_resII.h2o_frac, np.ones((N,))/N, mode='valid')
resII_hydrous_frac   = np.convolve(df_resII.hydrous_frac, np.ones((N,))/N, mode='valid')

###### ANNOTATIONS

ax3.text(0.01, 0.85, 'A', color="k", rotation=0, ha="left", va="center", fontsize=fsize+3, transform=ax3.transAxes)
ax3.text(0.05, 0.85, 'Metal-silicate segregation times in meteorites', color=qgray_dark, rotation=0, ha="left", va="center", fontsize=fsize-6, transform=ax3.transAxes)
ax1.text(0.01, 0.96, 'B', color="k", rotation=0, ha="left", va="center", fontsize=fsize+3, transform=ax1.transAxes)
ax1.text(0.05, 0.96, 'Simulation', color=qgray_dark, rotation=0, ha="left", va="center", fontsize=fsize-6, transform=ax1.transAxes)

ax1.annotate('Reservoir I\npercolation', xy=(0.24, 0.06), va="center", ha="left", fontsize=fsize-2, color=qred)
ax1.annotate('Reservoir I\nrain-out', xy=(0.8, 0.06), va="center", ha="left", fontsize=fsize-2, color=qred_dark)
# 
ax1.annotate('Res II\npercolation', xy=(5.8, 0.18), va="center", ha="right", fontsize=fsize-2, color=qblue)
ax1.annotate('Res II\nrain-out', xy=(5.8, 0.012), va="center", ha="right", fontsize=fsize-2, color=qblue_dark)

ax1.plot([0], [0], ls="--", c="black", lw=lw, label="Total, "+r"$f_\mathrm{diff}$")
ax1.plot([0], [0], ls="-", c="black", lw=lw, label="Rate, "+r"$\mathrm{d}f_\mathrm{diff}/\mathrm{d}t$")
diff_legend = ax1.legend(fontsize=fsize-4, loc=[0.02, 0.22], title="Differentiated material")
plt.setp(diff_legend.get_title(), fontsize=fsize-4)

ax3.annotate(r'Carbonaceous reservoir (CC)', xy=(1.0, 0.40), va="center", ha="right", fontsize=fsize-6, color=qblue, transform=ax3.transAxes)
ax3.annotate(r'Non-carbonaceous reservoir (NC)', xy=(0.65, 0.25), va="center", ha="right", fontsize=fsize-6, color=qred, transform=ax3.transAxes)

############# CORE FORMATION AGES AND HYDROTHERMAL ACTIVITY

## Calc error propagated mean age
import uncertainties
from uncertainties import ufloat

NC_irons = np.array([ufloat(0.3, 0.5), 
                     ufloat(0.8, 0.5), 
                     ufloat(1.2, 0.5), 
                     ufloat(1.8, 0.7), 
                     ufloat(1.5, 0.6)])#,
                     # ufloat(6.0, 0.8)]) # IAB

CC_irons = np.array([ufloat(2.6, 1.3), 
                        ufloat(2.3, 1.2), 
                        ufloat(2.3, 0.6), 
                        ufloat(2.5, 0.7), 
                        ufloat(2.2, 1.1), 
                        ufloat(2.8, 0.7)])

NC_hydr = np.array([ufloat(2.4, 1.8)])

CC_hydr = np.array( [ufloat(4.2, 0.8), 
                        ufloat(5.1, 0.5), 
                        ufloat(4.8, 3.0), 
                        ufloat(4.9, 0.7), 
                        ufloat(4.8, 0.5), 
                        ufloat(4.7, 1.3)])

# Calculate mean and associated error:
NC_iron_mean = np.mean(NC_irons)
CC_iron_mean = np.mean(CC_irons)
NC_hydr_mean = np.mean(NC_hydr)
CC_hydr_mean = np.mean(CC_hydr)

print("NC_iron_mean:", NC_iron_mean)
print("CC_iron_mean:", CC_iron_mean)
print("NC_hydr_mean:", NC_hydr_mean)
print("CC_hydr_mean:", CC_hydr_mean)

### Annotate ages as error bars
lwa            = 1.5
ages_y_base    = 0.1
ages_y_base_cc = 0.1
nc_color_ages  = qred
cc_color_ages  = qblue

# NC IRONS

# IC: 0.3 ± 0.5, -0.2 - 0.8
ax3.annotate("", xy=(0.8, ages_y_base+0.0), xycoords='data', xytext=(0.1, ages_y_base+0.0), textcoords='data', arrowprops=dict(arrowstyle="|-|,widthA=0.0,widthB=0.0", lw=lwa, connectionstyle="arc3", fc=nc_color_ages, ec=nc_color_ages))
ax3.plot(0.3, ages_y_base+0.00, color=nc_color_ages, marker='D', mec="white", zorder=20)
ax3.annotate(r'IC', xy=(0.82, ages_y_base+0.00), va="center", ha="left", fontsize=fsize-8, color=nc_color_ages, transform=ax3.transAxes)

# IIAB: 0.8 ± 0.5, 0.3 - 1.3
ax3.annotate("", xy=(1.3, ages_y_base+0.05), xycoords='data', xytext=(0.3, ages_y_base+0.05), textcoords='data', arrowprops=dict(arrowstyle="|-|,widthA=0.,widthB=0.", lw=lwa, connectionstyle="arc3", fc=nc_color_ages, ec=nc_color_ages))
ax3.plot(0.8, ages_y_base+0.05, color=nc_color_ages, marker='D', mec="white", zorder=20)
ax3.annotate(r'IIAB', xy=(1.35, ages_y_base+0.05), va="center", ha="left", fontsize=fsize-8, color=nc_color_ages, transform=ax3.transAxes)

# IIIAB: 1.2 ± 0.5, 0.7 - 1.7
ax3.annotate("", xy=(1.7, ages_y_base+0.10), xycoords='data', xytext=(0.7, ages_y_base+0.10), textcoords='data', arrowprops=dict(arrowstyle="|-|,widthA=0.,widthB=0.", lw=lwa, connectionstyle="arc3", fc=nc_color_ages, ec=nc_color_ages))
ax3.plot(1.2, ages_y_base+0.10, color=nc_color_ages, marker='D', mec="white", zorder=20)
ax3.annotate(r'IIIAB', xy=(1.77, ages_y_base+0.10), va="center", ha="left", fontsize=fsize-8, color=nc_color_ages, transform=ax3.transAxes)

# IVA: 1.5 ± 0.6, 0.9 - 2.1
ax3.annotate("", xy=(2.1, ages_y_base+0.15), xycoords='data', xytext=(0.9, ages_y_base+0.15), textcoords='data', arrowprops=dict(arrowstyle="|-|,widthA=0.,widthB=0.", lw=lwa, connectionstyle="arc3", fc=nc_color_ages, ec=nc_color_ages))
ax3.plot(1.5, ages_y_base+0.15, color=nc_color_ages, marker='D', mec="white", zorder=20)
ax3.annotate(r'IVA', xy=(2.2, ages_y_base+0.15), va="center", ha="left", fontsize=fsize-8, color=nc_color_ages, transform=ax3.transAxes)

# IIIE: 1.8 ± 0.7, 1.1 - 2.5
ax3.annotate("", xy=(2.5, ages_y_base+0.20), xycoords='data', xytext=(1.1, ages_y_base+0.20), textcoords='data', arrowprops=dict(arrowstyle="|-|,widthA=0.,widthB=0.", lw=lwa, connectionstyle="arc3", fc=nc_color_ages, ec=nc_color_ages))
ax3.plot(1.8, ages_y_base+0.20, color=nc_color_ages, marker='D', mec="white", zorder=20)
ax3.annotate(r'IIIE', xy=(2.6, ages_y_base+0.20), va="center", ha="left", fontsize=fsize-8, color=nc_color_ages, transform=ax3.transAxes)

# # IAB: 6.0 ± 0.8, 5.2 - 6.8
ax3.annotate("", xy=(6.0, ages_y_base+0.25), xycoords='data', xytext=(5.2, ages_y_base+0.25), textcoords='data', arrowprops=dict(arrowstyle="|-|,widthA=0.,widthB=0.", lw=lwa, connectionstyle="arc3", fc=nc_color_ages, ec=nc_color_ages))
ax3.plot(6.0, ages_y_base+0.25, color=nc_color_ages, marker='D', mec="white", zorder=20)
ax3.annotate(r'IAB', xy=(5.1, ages_y_base+0.25), va="center", ha="right", fontsize=fsize-8, color=nc_color_ages, transform=ax3.transAxes)

# CC IRONS

# IIIF: 2.2 ± 1.1, 1.1 - 3.3
ax3.annotate("", xy=(3.3, ages_y_base+0.30), xycoords='data', xytext=(1.1, ages_y_base+0.30), textcoords='data', arrowprops=dict(arrowstyle="|-|,widthA=0.,widthB=0.", lw=lwa, connectionstyle="arc3", fc=cc_color_ages, ec=cc_color_ages))
ax3.plot(2.2, ages_y_base+0.30, color=cc_color_ages, marker='s', mec="white", zorder=20)
ax3.annotate(r'IIIF', xy=(3.4, ages_y_base+0.30), va="center", ha="left", fontsize=fsize-8, color=cc_color_ages, transform=ax3.transAxes)

# IID: 2.3 ± 0.6, 1.7 - 2.9
ax3.annotate("", xy=(2.9, ages_y_base+0.35), xycoords='data', xytext=(1.7, ages_y_base+0.35), textcoords='data', arrowprops=dict(arrowstyle="|-|,widthA=0.,widthB=0.", lw=lwa, connectionstyle="arc3", fc=cc_color_ages, ec=cc_color_ages))
ax3.plot(2.3, ages_y_base+0.35, color=cc_color_ages, marker='s', mec="white", zorder=20)
ax3.annotate(r'IID', xy=(3.0, ages_y_base+0.35), va="center", ha="left", fontsize=fsize-8, color=cc_color_ages, transform=ax3.transAxes)

# IIF: 2.5 ± 0.7, 1.8 - 3.2
ax3.annotate("", xy=(3.2, ages_y_base+0.40), xycoords='data', xytext=(1.8, ages_y_base+0.40), textcoords='data', arrowprops=dict(arrowstyle="|-|,widthA=0.,widthB=0.", lw=lwa, connectionstyle="arc3", fc=cc_color_ages, ec=cc_color_ages))
ax3.plot(2.5, ages_y_base+0.40, color=cc_color_ages, marker='s', mec="white", zorder=20)
ax3.annotate(r'IIF', xy=(3.3, ages_y_base+0.40), va="center", ha="left", fontsize=fsize-8, color=cc_color_ages, transform=ax3.transAxes)

# IIC: 2.6 ± 1.3, 1.3 - 3.9
ax3.annotate("", xy=(3.9, ages_y_base+0.45), xycoords='data', xytext=(1.3, ages_y_base+0.45), textcoords='data', arrowprops=dict(arrowstyle="|-|,widthA=0.,widthB=0.", lw=lwa, connectionstyle="arc3", fc=cc_color_ages, ec=cc_color_ages))
ax3.plot(2.6, ages_y_base+0.45, color=cc_color_ages, marker='s', mec="white", zorder=20)
ax3.annotate(r'IIC', xy=(4.0, ages_y_base+0.45), va="center", ha="left", fontsize=fsize-8, color=cc_color_ages, transform=ax3.transAxes)

# IVB: 2.8 ± 0.7, 2.1 - 3.5
ax3.annotate("", xy=(3.5, ages_y_base+0.50), xycoords='data', xytext=(2.1, ages_y_base+0.50), textcoords='data', arrowprops=dict(arrowstyle="|-|,widthA=0.,widthB=0.", lw=lwa, connectionstyle="arc3", fc=cc_color_ages, ec=cc_color_ages))
ax3.plot(2.8, ages_y_base+0.50, color=cc_color_ages, marker='s', mec="white", zorder=20)
ax3.annotate(r'IVB', xy=(3.6, ages_y_base+0.50), va="center", ha="left", fontsize=fsize-8, color=cc_color_ages, transform=ax3.transAxes)

# Axes settings

time_ticks  = [0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 3, 4, 5, 6]
time_labels = [ str(n) for n in time_ticks ]

ax1.set_xscale("log")
ax1.set_yscale("log")

ax1.set_xlim([0.1, 6])
ax1.set_xticks(time_ticks)
ax1.set_xticklabels(time_labels)

ax1.set_ylim([0.0001, 1])
ax1.set_yticks([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1])
ax1.set_yticklabels(["0.01", "0.03", "0.1", "0.3", "1", "3", "10", "30", "100"])

ax1.tick_params(axis="both", labelsize=fsize-2)

ax3.set_ylim([0.05, 0.65])
ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, colors='white')
ax3.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False, colors='white')
ax3.spines['top'].set_color('white')
ax3.spines['bottom'].set_color('white')
ax3.spines['left'].set_color('white')
ax3.spines['right'].set_color('white')
ax3.set_facecolor("#f7f7f7")

ax1.set_xlabel(r"Time after CAI formation, $\Delta t_\mathrm{CAI}$ (Myr)", fontsize=fsize+2)
ax1.set_ylabel(r"Fraction of final planetesimal population (vol%)", fontsize=fsize+2)

sns.despine(ax=ax1, bottom=False, top=True, left=False, right=True)

plt.savefig(image_dir+"fig_4"+".pdf", bbox_inches='tight')
plt.savefig(image_dir+"fig_4"+".jpg", bbox_inches='tight', dpi=300)
