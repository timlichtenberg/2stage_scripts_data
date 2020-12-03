from sfd_functions import *
from jd_natconst import *
from jd_plot import *

# Plot settings
lw      = 2.5
fscale  = 2.0
fsize   = 23
sns.set_style("whitegrid")
sns.set(style="ticks", font_scale=fscale)

### Annotate ages as error bars
lwa            = 1.5
zorder 		   = 10
markersize 	   = 12

fig  = plt.subplots(figsize=(17,7))
grid = plt.GridSpec(1, 2)
ax1  = plt.subplot(grid[0, 0])
ax2  = plt.subplot(grid[0, 1])

# NEUTRON-RICH SUPERNOVAE MATERIAL

# CC chondrites
Cr54 	 = [ 0.48, 0.77, 0.87, 1.01, 1.21, 1.31, 1.56 ]
Cr54_err = [ 0.30, 0.30, 0.10, 0.08, 0.10, 0.10, 0.05 ]
Ti50 	 = [ 3.63, 3.78, 3.50, 3.02, 2.05, 2.76, 1.85 ]
Ti50_err = [ 0.40, 0.40, 0.20, 0.00, 0.00, 0.90, 0.00 ]

ax1.errorbar(Cr54, Ti50, xerr=Cr54_err, yerr=Ti50_err, fmt='v', color=qblue, ms=markersize, mec="white", zorder=zorder, label="CC chondrites")

# CC achondrites
Cr54 	 = [ 1.43, 1.44, 1.56 ]
Cr54_err = [ 0.05, 0.05, 0.10 ]
Ti50 	 = [ 3.05, 2.00, 2.07 ]
Ti50_err = [ 0.00, 0.00, 0.00 ]

ax1.errorbar(Cr54, Ti50, xerr=Cr54_err, yerr=Ti50_err, fmt='^', color=qblue, ms=markersize, mec="white", zorder=zorder, label="CC achondrites")

# Planets
Cr54 	 = [ 0.00, 0.15, -0.18 ]
Cr54_err = [ 0.00, 0.10, 0.00 ]
Ti50 	 = [ 0.00, -0.01, -0.41 ]
Ti50_err = [ 0.00, 0.00, 0.00 ]

ax1.errorbar(Cr54, Ti50, xerr=Cr54_err, yerr=Ti50_err, fmt='o', color=qmagenta, ms=markersize, mec="white", zorder=zorder, label="Earth, Moon, Mars")

# NC chondrites
Cr54 	 = [ -0.42, -0.40, -0.36, 0.04, 0.04 ]
Cr54_err = [ 0.05, 0.05, 0.05, 0.05, 0.05 ]
Ti50 	 = [ -0.66, -0.67, -0.63, -0.27, -0.12 ]
Ti50_err = [ 0.00, 0.00, 0.00, 0.00, 0.00 ]

ax1.errorbar(Cr54, Ti50, xerr=Cr54_err, yerr=Ti50_err, fmt='v', color=qred, ms=markersize, mec="white", zorder=zorder, label="NC chondrites")

# NC achondrites
Cr54 	 = [ -0.16, -0.37, -0.43, -0.67, -0.69, -0.75, -0.92 ]
Cr54_err = [ 0.30, 0.15, 0.15, 0.30, 0.00, 0.00, 0.00 ]
Ti50 	 = [ -0.05, -1.01, -1.17, -1.24, -1.25, -1.29, -1.84 ]
Ti50_err = [ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.20 ]

ax1.errorbar(Cr54, Ti50, xerr=Cr54_err, yerr=Ti50_err, fmt='^', color=qred, ms=markersize, mec="white", zorder=zorder, label="NC achondrites")

ax1.text(0.05, 0.46, 'Non-carbonaceous\nreservoir (NC)', color=qred, rotation=0, ha="left", va="top", fontsize=fsize-7, transform=ax1.transAxes)
ax1.text(0.42, 0.74, 'Carbonaceous\nreservoir (CC)', color=qblue, rotation=0, ha="left", va="top", fontsize=fsize-7, transform=ax1.transAxes)

fs_small = fsize-11

ax1.text(0.665, 0.88, 'CO', color=qblue, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax1.transAxes)
ax1.text(0.495, 0.805, 'CK', color=qblue, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax1.transAxes)
ax1.text(0.725, 0.815, 'CV', color=qblue, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax1.transAxes)
ax1.text(0.845, 0.78, 'NWA 2976', color=qblue, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax1.transAxes)
ax1.text(0.69, 0.705, 'CM', color=qblue, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax1.transAxes)
ax1.text(0.785, 0.67, 'CR', color=qblue, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax1.transAxes)
ax1.text(0.865, 0.635, 'NWA 6704', color=qblue, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax1.transAxes)
ax1.text(0.72, 0.61, 'CB', color=qblue, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax1.transAxes)
ax1.text(0.755, 0.56, 'Tafassasset', color=qblue, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax1.transAxes)
ax1.text(0.93, 0.55, 'CI', color=qblue, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax1.transAxes)

ax1.text(0.24, 0.345, 'Aubrites', color=qred, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax1.transAxes)
ax1.text(0.42, 0.285, 'ECs', color=qred, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax1.transAxes)
ax1.text(0.22, 0.27, 'OCs', color=qred, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax1.transAxes)
ax1.text(0.12, 0.205, 'HEDs', color=qred, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax1.transAxes)
ax1.text(0.31, 0.185, 'Angrites', color=qred, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax1.transAxes)
ax1.text(0.02, 0.18, 'Mesosiderites', color=qred, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax1.transAxes)
ax1.text(0.22, 0.13, 'Pallasites', color=qred, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax1.transAxes)
ax1.text(0.01, 0.12, 'Acapulcoites', color=qred, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax1.transAxes)
ax1.text(0.08, 0.065, 'Ureilites', color=qred, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax1.transAxes)

ax1.text(0.35, 0.365, 'Earth', color=qmagenta, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax1.transAxes)
ax1.text(0.45, 0.345, 'Moon', color=qmagenta, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax1.transAxes)
ax1.text(0.335, 0.25, 'Mars', color=qmagenta, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax1.transAxes)

# CAIs

ax1.annotate("", xy=(2.0, 3.62), xycoords='data', xytext=(1.7, 3.05), textcoords='data', arrowprops=dict(arrowstyle="-|>,head_length=0.3, head_width=0.15", ls="--", lw=lwa, connectionstyle="arc3", fc="white", ec=qturq), transform=ax1.transAxes, annotation_clip=False)

ax1.text(0.93, 0.73, 'CAIs', color=qturq, rotation=0, ha="left", va="center", fontsize=fs_small+2, transform=ax1.transAxes)

ax1.axhline(0, color=qgray_light, zorder=zorder-5, alpha=0.5)
ax1.axvline(0, color=qgray_light, zorder=zorder-5, alpha=0.5)

## S PROCESS VARIATIONS

# CAIs
Mo100       = [ 1.31 ]
Mo100_err   = [ 0.22 ]
Zr96        = [ 1.90 ]
Zr96_err    = [ 0.09 ]

ax2.errorbar(Mo100, Zr96, xerr=Mo100_err, yerr=Zr96_err, fmt='s', color=qturq, ms=markersize, mec="white", zorder=zorder, label="CAIs")

ax2.text(0.85, 0.85, 'CAIs', color=qturq, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax2.transAxes)

# CC chondrites
Mo100 	   = [ 0.44, 1.03, 1.00, 0.66, 0.59 ]
Mo100_err   = [ 0.34, 0.11, 0.30, 0.33, 0.23 ]
Zr96 	      = [ 0.24, 1.10, 1.11, 0.64, 0.90 ]
Zr96_err    = [ 0.22, 0.38, 0.17, 0.19, 0.25 ]

ax2.errorbar(Mo100, Zr96, xerr=Mo100_err, yerr=Zr96_err, fmt='v', color=qblue, ms=markersize, mec="white", zorder=zorder, label="CC chondrites")

# NC chondrites
Mo100 	   = [ 0.21, 0.12 ]
Mo100_err   = [ 0.08, 0.03 ]
Zr96 	      = [ 0.46, -0.02 ]
Zr96_err    = [ 0.14, 0.24 ]

ax2.errorbar(Mo100, Zr96, xerr=Mo100_err, yerr=Zr96_err, fmt='v', color=qred, ms=markersize, mec="white", zorder=zorder, label="NC chondrites")

ax2.text(0.60, 0.60, 'CV', color=qblue, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax2.transAxes)
ax2.text(0.69, 0.54, 'CR', color=qblue, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax2.transAxes)
ax2.text(0.37, 0.52, 'CB', color=qblue, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax2.transAxes)
ax2.text(0.48, 0.36, 'CO', color=qblue, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax2.transAxes)
ax2.text(0.35, 0.21, 'CI', color=qblue, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax2.transAxes)

ax2.text(0.15, 0.355, 'OC', color=qred, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax2.transAxes)
ax2.text(0.165, 0.105, 'EC', color=qred, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax2.transAxes)

# Earth
Mo100       = [ -0.03 ]
Mo100_err   = [ 0.04 ]
Zr96        = [ 0.00 ]
Zr96_err    = [ 0.04 ]

ax2.errorbar(Mo100, Zr96, xerr=Mo100_err, yerr=Zr96_err, fmt='o', color=qmagenta, ms=markersize, mec="white", zorder=zorder, label="Earth")

ax2.text(0.015, 0.105, 'Earth', color=qmagenta, rotation=0, ha="left", va="center", fontsize=fs_small, transform=ax2.transAxes)

# S deficit
ax2.annotate("", xy=(0.6, 0.8), xycoords='axes fraction', xytext=(0.48, 0.68), textcoords='axes fraction', arrowprops=dict(arrowstyle="-|>,head_length=0.3, head_width=0.15", ls="-", lw=lwa, connectionstyle="arc3", fc=qgray, ec=qgray), transform=ax2.transAxes, annotation_clip=False)
ax2.text(0.54, 0.76, '$s$ deficit', color=qgray, rotation=0, ha="right", va="center", fontsize=fs_small+2, transform=ax2.transAxes)

# FURTHER ANNOTATIONS

ax2.axhline(0, color=qgray_light, zorder=zorder-5, alpha=0.5)
ax2.axvline(0, color=qgray_light, zorder=zorder-5, alpha=0.5)

ax1.text(0.02, 0.98, 'A', color="k", rotation=0, ha="left", va="top", zorder=100, fontsize=fsize+3, transform=ax1.transAxes)
ax2.text(0.02, 0.98, 'B', color="k", rotation=0, ha="left", va="top", zorder=100, fontsize=fsize+3, transform=ax2.transAxes)

ax1.text(0.09, 0.952, 'Variations in neutron-rich supernovae-derived isotopes', color="k", rotation=0, ha="left", va="center", fontsize=fsize-7, transform=ax1.transAxes)
ax2.text(0.09, 0.952, 'Variations in $s$-process isotopes', color="k", rotation=0, ha="left", va="center", fontsize=fsize-7, transform=ax2.transAxes)

# Axes settings
ax1.tick_params(axis="both", labelsize=fsize-4)
ax2.tick_params(axis="both", labelsize=fsize-4)
ax1.set_ylim(top=4.85)
ax2.set_ylim(top=2.2)
ax1.set_xlabel(r"$\epsilon \, ^{54}$Cr", fontsize=fsize+2)
ax1.set_ylabel(r"$\epsilon \, ^{50}$Ti", fontsize=fsize+2)
ax2.set_xlabel(r"$\epsilon \, ^{100}$Mo", fontsize=fsize+2)
ax2.set_ylabel(r"$\epsilon \, ^{96}$Zr", fontsize=fsize+2)
ax1.yaxis.set_label_coords(-0.06, 0.5)
ax2.yaxis.set_label_coords(-0.10, 0.5)

# remove the errorbars in legends
handles1, labels1 = ax1.get_legend_handles_labels()
handles1 = [h[0] for h in handles1]
handles2, labels2 = ax2.get_legend_handles_labels()
handles2 = [h[0] for h in handles2]

lg1 = ax1.legend(handles1, labels1, loc=4, fontsize=fsize-8, ncol=1)
lg2 = ax2.legend(handles2, labels2, loc=4, fontsize=fsize-8, ncol=1)

sns.despine(ax=ax1, top=True, right=True, left=False, bottom=False)
sns.despine(ax=ax2, top=True, right=True, left=False, bottom=False)

plt.savefig(image_dir+"fig_s12"+".pdf", bbox_inches='tight')
plt.savefig(image_dir+"fig_s12"+".jpg", bbox_inches='tight', dpi=300)

