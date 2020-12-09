# Script from: https://github.com/timlichtenberg/2stage_scripts_data
# Part of the combined repository: https://osf.io/e2kfv/
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
grid = plt.GridSpec(4, 1, wspace=0.0, hspace=0.1) #, wspace=0.4, hspace=0.3
ax2  = plt.subplot(grid[1:4, 0])
ax4  = plt.subplot(grid[0:1, 0], sharex=ax2)

# Reservoirs
df_resI   = df_evolution.loc[(df_evolution["imf"] == imf) & (df_evolution["res"] == "resI")]
df_resII  = df_evolution.loc[(df_evolution["imf"] == imf) & (df_evolution["res"] == "resII")]

# Metal-silicate

N = 25
rolling_mean_timeI   = np.convolve(df_resI.time, np.ones((N,))/N, mode='valid')
rolling_mean_timeII  = np.convolve(df_resII.time, np.ones((N,))/N, mode='valid')
resI_perco           = np.convolve(df_resI.perco_frac, np.ones((N,))/N, mode='valid')
resI_melt2           = np.convolve(df_resI.melt2_frac, np.ones((N,))/N, mode='valid')
resII_perco          = np.convolve(df_resII.perco_frac, np.ones((N,))/N, mode='valid')
resII_melt2          = np.convolve(df_resII.melt2_frac, np.ones((N,))/N, mode='valid')

### Normalize by final reservoir mass

M_ResI  = 1.222764    # M_Earth, at 5.0 Myr
M_ResII = 770.832781  # M_Earth, at 5.0 Myr

resI_perco_frac      = df_resI.perco_frac.tolist()
resI_melt2_frac      = df_resI.melt2_frac.tolist()
resI_primitive_frac  = df_resI.primitive_frac.tolist()
resI_h2o_frac        = df_resI.h2o_frac.tolist()
resI_hydrous_frac    = df_resI.hydrous_frac.tolist()
resI_m_plts_tot      = df_resI.m_plts_tot.tolist()
resII_perco_frac     = df_resII.perco_frac.tolist()
resII_melt2_frac     = df_resII.melt2_frac.tolist()
resII_primitive_frac = df_resII.primitive_frac.tolist()
resII_h2o_frac       = df_resII.h2o_frac.tolist()
resII_hydrous_frac   = df_resII.hydrous_frac.tolist()
resII_m_plts_tot     = df_resII.m_plts_tot.tolist()

for i in range(0, len(df_resI)):
   m_plts_tot = resI_m_plts_tot[i]
   m_frac     = m_plts_tot/M_ResI
   resI_perco_frac[i]     = resI_perco_frac[i]*m_frac
   resI_melt2_frac[i]     = resI_melt2_frac[i]*m_frac
   resI_primitive_frac[i] = resI_primitive_frac[i]*m_frac
   resI_h2o_frac[i]       = resI_h2o_frac[i]*m_frac
   resI_hydrous_frac[i]   = resI_hydrous_frac[i]*m_frac

for i in range(0, len(df_resII)):
   m_plts_tot = resII_m_plts_tot[i]
   m_frac     = m_plts_tot/M_ResII
   resII_perco_frac[i]     = resII_perco_frac[i]*m_frac
   resII_melt2_frac[i]     = resII_melt2_frac[i]*m_frac
   resII_primitive_frac[i] = resII_primitive_frac[i]*m_frac
   resII_h2o_frac[i]       = resII_h2o_frac[i]*m_frac
   resII_hydrous_frac[i]   = resII_hydrous_frac[i]*m_frac

N = 3
rolling_mean_timeI 	= np.convolve(df_resI.time, np.ones((N,))/N, mode='valid')
resI_primitive_frac  = np.convolve(resI_primitive_frac, np.ones((N,))/N, mode='valid')
resI_h2o_frac        = np.convolve(resI_h2o_frac, np.ones((N,))/N, mode='valid')
resI_hydrous_frac    = np.convolve(resI_hydrous_frac, np.ones((N,))/N, mode='valid')

N = 5
rolling_mean_timeII  = np.convolve(df_resII.time, np.ones((N,))/N, mode='valid')
resII_primitive_frac = np.convolve(resII_primitive_frac, np.ones((N,))/N, mode='valid')
resII_h2o_frac       = np.convolve(resII_h2o_frac, np.ones((N,))/N, mode='valid')
resII_hydrous_frac   = np.convolve(resII_hydrous_frac, np.ones((N,))/N, mode='valid')

## Hydrous activity
ax2.plot(rolling_mean_timeI,  resI_hydrous_frac, ls="-", c=qred_dark, lw=lw)
ax2.plot(rolling_mean_timeII, resII_hydrous_frac, ls="-", c=qblue_dark, lw=lw)

## Fillings
ax2.fill_between(rolling_mean_timeI, 0, resI_hydrous_frac, color=qred, alpha=0.5)
ax2.fill_between(rolling_mean_timeII, 0, resII_hydrous_frac, color=qblue, alpha=0.5)

## Decomposition fields
ax2.plot(rolling_mean_timeI,  resI_primitive_frac, ls="--", c=qred, lw=lw)
ax2.plot(rolling_mean_timeII, resII_primitive_frac, ls="--", c=qblue, lw=lw)

###### ANNOTATIONS

ax4.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, colors='white')
ax4.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False, colors='white')
ax4.spines['top'].set_color('white')
ax4.spines['bottom'].set_color('white')
ax4.spines['left'].set_color('white')
ax4.spines['right'].set_color('white')
ax4.set_facecolor("#f7f7f7")

ax2.annotate(r'Reservoir I', xy=(0.15, 0.45), va="center", ha="center", fontsize=fsize-2, color=qred, xycoords="axes fraction") #, xycoords="axes fraction"
ax2.annotate(r'Reservoir II', xy=(0.75, 0.45), va="center", ha="center", fontsize=fsize-2, color=qblue, xycoords="axes fraction") #, xycoords="axes fraction", transform=ax2.transAxes

ax4.text(0.01, 0.85, 'A', color="k", rotation=0, ha="left", va="center", fontsize=fsize+3, transform=ax4.transAxes)
ax4.text(0.05, 0.85, 'Aqueous alteration times in meteorites', color=qgray_dark, rotation=0, ha="left", va="center", fontsize=fsize-6, transform=ax4.transAxes)
ax2.text(0.01, 0.96, 'B', color="k", rotation=0, ha="left", va="center", fontsize=fsize+3, transform=ax2.transAxes)
ax2.text(0.05, 0.96, 'Simulation', color=qgray_dark, rotation=0, ha="left", va="center", fontsize=fsize-6, transform=ax2.transAxes, bbox=dict(fc='white', ec="white", alpha=0.8, pad=0.1, boxstyle='round'))

ax4.annotate(r'Carbonaceous reservoir (CC)', xy=(0.40, 0.25), va="center", ha="left", fontsize=fsize-6, color=qblue, transform=ax4.transAxes)
ax4.annotate(r'Non-carbonaceous reservoir (NC)', xy=(0.12, 0.09), va="center", ha="left", fontsize=fsize-6, color=qred, transform=ax4.transAxes)

ax2.plot([0], [0], ls="-", c="black", lw=lw, label=r"Hydrous rock, $T_\mathrm{hydr} \leq T_\mathrm{max} < T_\mathrm{decomp}$")
ax2.plot([0], [0], ls="--", c="black", lw=lw, label=r"Water ice, $T_\mathrm{max} < T_\mathrm{hydr}$")

legend1 = ax2.legend(fontsize=fsize-4, loc=1, title="Water phase in planetesimals")
plt.setp(legend1.get_title(), fontsize=fsize-4)

############# HYDROTHERMAL ACTIVITY

# ### Annotate ages as error bars
lwa            = 1.5
ages_y_base    = 0.1
ages_y_base_cc = 0.1
nc_color_ages  = qred
cc_color_ages  = qblue

## HYDROTHERMAL ACTIVITY
ages_y_base    = 0.19
ages_y_base_cc = 0.09

# CAI offset
CAI_offset = 0.0

# OC: 2.4 (-1.3/+1.8), 1.1 - 4.2
y_offset = 0.00
ax4.annotate("", xy=(4.2+CAI_offset, ages_y_base_cc+y_offset), xycoords='data', xytext=(1.1+CAI_offset, ages_y_base_cc+y_offset), textcoords='data', arrowprops=dict(arrowstyle="|-|,widthA=0.0,widthB=0.0", lw=lwa, connectionstyle="arc3", fc=nc_color_ages, ec=nc_color_ages))
ax4.plot(2.4+CAI_offset, ages_y_base_cc+y_offset, color=nc_color_ages, marker='D', mec="white", zorder=20)
ax4.annotate('OC', xy=(1.05, ages_y_base_cc+y_offset), va="center", ha="right", fontsize=fsize-8, color=nc_color_ages, transform=ax4.transAxes)

# TL: 4.1 (-1.1/+1.3), 3.0 - 5.4
y_offset = 0.05
ax4.annotate("", xy=(5.4+CAI_offset, ages_y_base_cc+y_offset), xycoords='data', xytext=(3.0+CAI_offset, ages_y_base_cc+y_offset), textcoords='data', arrowprops=dict(arrowstyle="|-|,widthA=0.0,widthB=0.0", lw=lwa, connectionstyle="arc3", fc=cc_color_ages, ec=cc_color_ages))
ax4.plot(4.1+CAI_offset, ages_y_base_cc+y_offset, color=cc_color_ages, marker='s', mec="white", zorder=20)
ax4.annotate('Tagish Lake', xy=(2.9, ages_y_base_cc+y_offset), va="center", ha="right", fontsize=fsize-9, color=cc_color_ages, transform=ax4.transAxes)

# CM: 4.1 (-0.4/+0.5), 3.7 - 4.6
y_offset = 0.10
ax4.annotate("", xy=(4.6+CAI_offset, ages_y_base_cc+y_offset), xycoords='data', xytext=(3.7+CAI_offset, ages_y_base_cc+y_offset), textcoords='data', arrowprops=dict(arrowstyle="|-|,widthA=0.0,widthB=0.0", lw=lwa, connectionstyle="arc3", fc=cc_color_ages, ec=cc_color_ages))
ax4.plot(4.1+CAI_offset, ages_y_base_cc+y_offset, color=cc_color_ages, marker='s', mec="white", zorder=20)
ax4.annotate('CM', xy=(3.6, ages_y_base_cc+y_offset), va="center", ha="right", fontsize=fsize-8, color=cc_color_ages, transform=ax4.transAxes)

# CV: 4.2 (-0.7/+0.8), 3.5 - 5.0
y_offset = 0.15
ax4.annotate("", xy=(5.0+CAI_offset, ages_y_base_cc+y_offset), xycoords='data', xytext=(3.5+CAI_offset, ages_y_base_cc+y_offset), textcoords='data', arrowprops=dict(arrowstyle="|-|,widthA=0.0,widthB=0.0", lw=lwa, connectionstyle="arc3", fc=cc_color_ages, ec=cc_color_ages))
ax4.plot(4.2+CAI_offset, ages_y_base_cc+y_offset, color=cc_color_ages, marker='s', mec="white", zorder=20)
ax4.annotate('CV', xy=(3.4, ages_y_base_cc+y_offset), va="center", ha="right", fontsize=fsize-8, color=cc_color_ages, transform=ax4.transAxes)

# CR: 4.2 (-1.3/+1.8), 2.9 - 6.0
y_offset = 0.20
ax4.annotate("", xy=(6.0, ages_y_base_cc+y_offset), xycoords='data', xytext=(2.9+CAI_offset, ages_y_base_cc+y_offset), textcoords='data', arrowprops=dict(arrowstyle="|-|,widthA=0.0,widthB=0.0", lw=lwa, connectionstyle="arc3", fc=cc_color_ages, ec=cc_color_ages))
ax4.plot(4.2+CAI_offset, ages_y_base_cc+y_offset, color=cc_color_ages, marker='s', mec="white", zorder=20)
ax4.annotate('CR', xy=(2.8, ages_y_base_cc+y_offset), va="center", ha="right", fontsize=fsize-8, color=cc_color_ages, transform=ax4.transAxes)

# CI: 4.4 (-0.5/+0.6), 3.9 - 5.0
y_offset = 0.25
ax4.annotate("", xy=(5.0+CAI_offset, ages_y_base_cc+y_offset), xycoords='data', xytext=(3.9+CAI_offset, ages_y_base_cc+y_offset), textcoords='data', arrowprops=dict(arrowstyle="|-|,widthA=0.0,widthB=0.0", lw=lwa, connectionstyle="arc3", fc=cc_color_ages, ec=cc_color_ages))
ax4.plot(4.4+CAI_offset, ages_y_base_cc+y_offset, color=cc_color_ages, marker='s', mec="white", zorder=20)
ax4.annotate('CI', xy=(3.8, ages_y_base_cc+y_offset), va="center", ha="right", fontsize=fsize-8, color=cc_color_ages, transform=ax4.transAxes)

# CO: 5.1 (-0.4/+0.5), 4.7 - 5.6
y_offset = 0.30
ax4.annotate("", xy=(5.6, ages_y_base_cc+y_offset), xycoords='data', xytext=(4.7+CAI_offset, ages_y_base_cc+y_offset), textcoords='data', arrowprops=dict(arrowstyle="|-|,widthA=0.0,widthB=0.0", lw=lwa, connectionstyle="arc3", fc=cc_color_ages, ec=cc_color_ages))
ax4.plot(5.1+CAI_offset, ages_y_base_cc+y_offset, color=cc_color_ages, marker='s', mec="white", zorder=20)
ax4.annotate('CO', xy=(4.6, ages_y_base_cc+y_offset), va="center", ha="right", fontsize=fsize-8, color=cc_color_ages, transform=ax4.transAxes)

# Axes settings

time_ticks  = [0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 3, 4, 5, 6]
time_labels = [ str(n) for n in time_ticks ]

ax2.set_xscale("log")

ax2.set_xlim([0.1, 6])
ax2.set_xticks(time_ticks)
ax2.set_xticklabels(time_labels)
ax2.set_ylim([0.0, 1])
ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_yticklabels(["0", "20", "40", "60", "80", "100"])
ax2.tick_params(axis="both", labelsize=fsize-2)
ax4.set_ylim([0.05, 0.42])
ax2.set_ylim([0.0, 1])

sns.despine(ax=ax2, bottom=False, top=True, left=False, right=True)

ax2.set_xlabel(r"Time after CAI formation, $\Delta t_\mathrm{CAI}$ (Myr)", fontsize=fsize+2)

ax2.set_ylabel(r"Fraction of final planetesimal population (vol%)", fontsize=fsize+2)

plt.savefig(image_dir+"fig_5"+".pdf", bbox_inches='tight')
plt.savefig(image_dir+"fig_5"+".jpg", bbox_inches='tight', dpi=300)

