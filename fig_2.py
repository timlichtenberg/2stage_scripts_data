from jd_natconst import *
from jd_readdata import *
from jd_plot import *

### ! REQUIRES: Ormel & Liu (2018) python script "OL18.py"
### https://staff.fnwi.uva.nl/c.w.ormel/software.html -> "epsilon.tar.gz"
import OL18

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

q = readdata()

# Plot settings
lw      = 2
fscale  = 2.0
fsize   = 20

plt.rcParams.update({'font.size': 18})
plt.rcParams['axes.linewidth'] = 1.5 

## Calculate growth timescales
# Ida & Lin 2004, Eq. 6, all units in CGS
# Sigma_d : g/cm^2
# a       : au
# M_c     : M_earth
# M_star  : M_sun
# Sigma_g : g/cm^2
# m       : g
def calc_runaway_growth_timescale ( Sigma_d, a, M_c, M_star, Sigma_g, m ):

    tau_acc = 1.2e+5 * (Sigma_d / 10)**(-1) * a**(1/2) * M_c**(1/3) * M_star**(-1/6) * ( (Sigma_g/2.4e+3)**(-1/5) * a**(1/20) * (m/1e+18)**(1/15) )**(2.) # yr

    return tau_acc # yr

# time : yr
def calc_runaway_growth_mass( Sigma_d, a, time, M_star, Sigma_g, m ):

    M_c = (time/3.5e+5)**(3) * (Sigma_d/10)**(3) * (Sigma_g/2.4e+3)**(6/5) * (m/1e+18)**(-2/5) * (a)**(-9/5) * M_star**(1/2) # M_earth

    return M_c # M_earth

# rad in km
def m_plts ( frac_water, rad ):

    frac_silicates = 1. - frac_water
    rho            = frac_water*1. + frac_silicates*3.5
    m_plts         = rho*(4./3.)*np.pi*((rad*1e5)**3) # g

    return m_plts

# https://seaborn.pydata.org/tutorial/aesthetics.html
# https://seaborn.pydata.org/examples/scatterplot_matrix.html
sns.set_style("whitegrid")
# sns.set_context("poster")
sns.set(style="ticks", font_scale=fscale)

ls         = "-"
ls_pebble  = "-"
ls_runaway = "--"

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11,13))

axes = plt.gca()
ax1 = plt.subplot(211)
ax2 = plt.subplot(212) # , sharex=ax1

## Pebble flux unit
norm=1e6*year/Mea


for a in [ 2, 15 ]: # au

    # Number of points to calculate rolling mean
    if a == 2:
        N = 3
    if a == 15:
        N = 15

    if a == 1:
        color=qgreen
    if a == 2:
        color = qred
        color_dark = qred_dark
    if a == 3:
        color=qyellow
    if a == 5:
        color=qblue_light
    if a >= 10:
        color = qblue
        color_dark = qblue_dark
    label=str(a)+" au"

    idx_a               = find_nearest_idx(q.r, a*au)
    pebble_flux_old     = 2.*np.pi*q.r[idx_a]*q.sigmad[:,idx_a]*np.abs(q.vdust[:,idx_a])*norm
    
    rolling_time        = np.convolve(q.time/year, np.ones((N,))/N, mode='valid')

    pebble_flux         = np.abs(q.mflux[:,idx_a])*norm # directly derived from sim
    rolling_pebble_flux = np.convolve(pebble_flux, np.ones((N,))/N, mode='valid')

    pebble_flux_inward  = -1*q.mflux[:,idx_a]*norm # directly derived from sim
    rolling_pebble_flux_inward = np.convolve(pebble_flux_inward, np.ones((N,))/N, mode='valid')

    pebble_flux_outward = q.mflux[:,idx_a]*norm # directly derived from sim
    rolling_pebble_flux_outward = np.convolve(pebble_flux_outward, np.ones((N,))/N, mode='valid')

    ax1.plot(rolling_time, rolling_pebble_flux, ls=ls_pebble, c=color, lw=lw)
    ax1.plot(rolling_time, rolling_pebble_flux_outward, ls=":", c=color_dark, lw=lw)

    if a == 2:
        rolling_pebble_flux_2   = rolling_pebble_flux
    if a == 15:
        rolling_pebble_flux_15  = rolling_pebble_flux

    ## PLOT PEBBLE ACCRETION

    time = q.time / year # define time in years for convenience

    pebflux = np.abs(q.mflux)

    # Calculate quantities needed for the pebble accretion efficiency
    cs = np.sqrt(q.tmid*kk/(2.34*mp))
    omegaK = np.zeros((np.size(q.time),np.size(q.r)))
    eta = np.zeros((np.size(q.time),np.size(q.r)))
    for it in range(np.size(q.time)):
        omegaK[it,:]= np.sqrt(GG*q.mstar[it]/q.r[:]**3)
    Hgas = cs/omegaK/q.r # CAUTION: this is Chris' definition Hgas = H/r
    Pg = q.sigma*omegaK/(np.sqrt(2.*pi)) * cs
    for ir in range(np.size(q.r)-1):
        eta[:,ir]= -0.5*q.r[ir]/(Pg[:,ir]+1.e-20)*(Pg[:,ir+1]-Pg[:,ir])/(q.r[ir+1]-q.r[ir])*Hgas[:,ir]**2.

    rdis = a*au
    pebble_growth_timescale = []
    runaway_growth_timescale = []

    embryo_radius = 300                          # km
    embryo_mass   = m_plts(0.3, embryo_radius)   # g

    plts_radius   = 50                           # km
    plts_mass     = m_plts(0.3, plts_radius)     # g

    print("Embryo mass / radius:", embryo_mass, embryo_radius)
    print("Plts mass / radius:", plts_mass, plts_radius)

    for tyear in time:

        ir = q.r.searchsorted(rdis)
        it = time.searchsorted(tyear)

        # Properties of the accreting planet
        Rpor = embryo_radius*1e+5 / q.r[ir] # planet radius over orbital radius; km->cm
        pmos = embryo_mass / q.mstar[it]    # planet mass over stellar mass

        # pebble accretion efficiency
        eps = OL18.epsilon_general(tau=q.stokesnr[it,ir], qp=pmos, eta=eta[it,ir], hgas=Hgas[it,ir], alphaz=1e-4, ip=0., nvec=(1,1,1), Rp=Rpor, mode='')
        # print(eps)

        # growth rate of the planet in g/s
        Mdot = eps * pebflux[it,ir]

        # PEBBLE growth timescale: how long will it take to double the mass
        tgrow = embryo_mass / Mdot
        tgrow = tgrow / year # years
        tgrow = tgrow / 1e+6 # Myr

        # Pebble growth == 0 if no planetesimals!
        if q.sigmaplts[it,ir] >= 1e-4:
            pebble_growth_timescale.append(tgrow)
        else:
            pebble_growth_timescale.append(1e+99)

        # PLANETESIMAL growth timescale
        pebble_enhancement_factor   = 1.
        tau_acc = calc_runaway_growth_timescale ( q.sigmaplts[it,ir]*pebble_enhancement_factor, q.r[ir]/au, embryo_mass*pebble_enhancement_factor/Mea, q.mstar[it]/MS, q.sigma[it, ir], plts_mass*pebble_enhancement_factor )
        tau_acc = tau_acc / 1e+6 # Myr

        runaway_growth_timescale.append(tau_acc) 

    # filter out NaNs
    pebble_growth_timescale = [0 if math.isnan(x) else x for x in pebble_growth_timescale]
    
    # # yr -> Myr
    # # time = [ t/1e+6 for t in time ]
    # pebble_growth_timescale = [ t/1e+6 for t in pebble_growth_timescale ]

    # smoothed data
    rolling_time = np.convolve(time, np.ones((N,))/N, mode='valid')
    rolling_pebble_growth_timescale = np.convolve(pebble_growth_timescale, np.ones((N,))/N, mode='valid')
    rolling_runaway_growth_timescale = np.convolve(runaway_growth_timescale, np.ones((N,))/N, mode='valid')

    # plot
    ax2.plot(rolling_time, rolling_pebble_growth_timescale, ls=ls_pebble, c=color, lw=lw)
    ax2.plot(rolling_time, rolling_runaway_growth_timescale, ls=ls_runaway, c=color, lw=lw)


###### Annotations
N = 1000 # number of vspan blocks

ax2.text(0.25e+6, 4.8, r'$\approx$ disk lifetime', color=qgray_dark, rotation=0, ha="left", va="center", fontsize=fsize-4)
ax2.axhspan(4, 6, color=qgray_light, alpha=.3, lw=0)

ax1.text(0.71, 0.650, 'Reservoir\n                   separation', color="k", rotation=0, ha="center", va="top", fontsize=fsize, transform=ax1.transAxes)

# # Shading ax1
# start_point = 5.91
# end_point   = 6.78
# xspan       = np.logspace(start_point, end_point, N)
# for i in range(0, N-1):
#     alpha_min   = 0.05
#     alpha_max   = 0.99
#     alpha_scale = float(i)/float(N)
#     alpha       = alpha_min + (alpha_max-alpha_min)*alpha_scale
#     ax1.axvspan(xspan[i], xspan[i+1], color=qgray_light, alpha=alpha, lw=-0.0)

# ax1.fill_between(rolling_time, rolling_pebble_flux_2, rolling_pebble_flux_15, color=qgray_light)

ax2.text(0.4e+6, 0.4, 'Collision-dominated growth', color=qred, rotation=0, ha="center", va="bottom", fontsize=fsize-4)
ax2.text(1.21e+6, 0.4, 'Pebble-aided growth', color=qred_dark, rotation=0, ha="center", va="bottom", fontsize=fsize-4)
ax2.text(3.10e+6, 0.4, 'Pebble-starved', color=qred, rotation=0, ha="center", va="bottom", fontsize=fsize-4)
from scipy import signal
alpha_distribution = signal.gaussian(N, std=int(0.15*N))
xspan              = np.logspace(5.71, 6.5, N)
for i in range(0, N-1):
    alpha_max   = 0.99
    alpha       = alpha_distribution[i]*alpha_max
    ax2.axvspan(xspan[i], xspan[i+1], color=qred_light, alpha=alpha, lw=-0.0, ymax=0.3235)

ax1.text(0.985, 0.97, 'A', color="k", rotation=0, ha="right", va="top", fontsize=fsize+4, transform=ax1.transAxes)
ax2.text(0.985, 0.97, 'B', color="k", rotation=0, ha="right", va="top", fontsize=fsize+4, transform=ax2.transAxes)

ax1.plot([0], [0], ls="-", c=qred, lw=lw, label="2 au, total flux")
ax1.plot([0], [0], ls=":", c=qred_dark, lw=lw, label="2 au, outward flux")
ax1.plot([0], [0], ls="-", c=qblue, lw=lw, label="15 au, total flux")
ax1.plot([0], [0], ls=":", c=qblue_dark, lw=lw, label="15 au, outward flux")

ax2.plot([0], [0], ls="-", c=qred, lw=lw, label="2 au, pebbles")
ax2.plot([0], [0], ls="--", c=qred, lw=lw, label="2 au, planetesimals")
ax2.plot([0], [0], ls="-", c=qblue, lw=lw, label="15 au, pebbles")
ax2.plot([0], [0], ls="--", c=qblue, lw=lw, label="15 au, planetesimals")


## ARROWS
lwa=1.5
ax1.annotate('', xy=(0.6, 0.48), xycoords='axes fraction', xytext=(0.6, 0.56), textcoords='axes fraction', arrowprops={'arrowstyle': 'simple,head_length=0.3,head_width=0.3,tail_width=0.07', 'facecolor': 'k', 'edgecolor': 'k', 'lw': lwa}, va='center', transform=ax1.transAxes)
ax1.annotate('', xy=(0.6, 0.595), xycoords='axes fraction', xytext=(0.6, 0.52), textcoords='axes fraction', arrowprops={'arrowstyle': 'simple,head_length=0.3,head_width=0.3,tail_width=0.07', 'facecolor': 'k', 'edgecolor': 'k', 'lw': lwa}, va='center', transform=ax1.transAxes)
ax1.annotate('', xy=(0.85, 0.08), xycoords='axes fraction', xytext=(0.85, 0.40), textcoords='axes fraction', arrowprops={'arrowstyle': 'simple,head_length=0.3,head_width=0.3,tail_width=0.07', 'facecolor': 'k', 'edgecolor': 'k', 'lw': lwa}, va='center', transform=ax1.transAxes)
ax1.annotate('', xy=(0.85, 0.44), xycoords='axes fraction', xytext=(0.85, 0.20), textcoords='axes fraction', arrowprops={'arrowstyle': 'simple,head_length=0.3,head_width=0.3,tail_width=0.07', 'facecolor': 'k', 'edgecolor': 'k', 'lw': lwa}, va='center', transform=ax1.transAxes)
# ax1.text(0.80, 0.33, r'$\frac{\mathrm{d} M_{\mathrm{Res} \, \mathrm{II}}}{\mathrm{d}t}$', color="black", rotation=0, ha="center", va="top", fontsize=fsize+2, transform=ax1.transAxes)

ax1.text(0.015, 0.97, 'Disk build-up', color=qgray_dark, size=fsize, rotation=0, va='center', ha='left', fontsize=fsize-5, transform=ax1.transAxes)
ax1.annotate("", xy=(0.34, 0.97), xycoords='axes fraction', xytext=(0.17, 0.97), textcoords='axes fraction', arrowprops=dict(arrowstyle="-|>,head_length=0.2,head_width=0.1", lw=lwa, connectionstyle="arc3", fc=qgray, ec=qgray), transform=ax1.transAxes)


###### PLOT SETTINGS

legend1 = ax1.legend(fontsize=fsize-5, loc="lower left", ncol=2, title="Flux contribution")
legend2 = ax2.legend(fontsize=fsize-5, loc="upper left", ncol=2, title="Orbit & growth mode")
plt.setp(legend1.get_title(), fontsize=fsize-4)
plt.setp(legend2.get_title(), fontsize=fsize-4)

x_left  = 2.2e+5
x_right = 4.2e+6
xticks  = [3e+5, 5e+5, 7e+5, 1e+6, 2e+6, 3e+6, 4e+6, 5e+6]
xticklabels = ["0.3", "0.5", "0.7", "1", "2", "3", "4", "5"]

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylabel(r'Pebble flux ($M_{\mathrm{Earth}}$ Myr$^{-1}$)', fontsize=fsize)
ax1.set_xlim([x_left, x_right])
ax1.set_ylim(bottom=1e+0, top=3e3)
ax1.set_xticks(xticks)
ax1.set_yticks([1e+0, 1e1, 1e2, 1e3])
ax1.tick_params(axis="both", labelsize=fsize)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylabel(r'Growth timescale (Myr)', fontsize=fsize)
ax2.set_xlim([x_left, x_right])
ax2.set_xticks(xticks)
ax2.set_xticklabels(xticklabels)
ax2.set_ylim(bottom=3e-1, top=9e+2)
ax2.tick_params(axis="both", labelsize=fsize)
ax2.set_xlabel(r'Time after CAI formation, $\Delta t_{\mathrm{CAI}}\ (\rm{Myr})$', fontsize=fsize)

plt.setp(ax1.get_xticklabels(), visible=False)
plt.tight_layout(pad=2.5, w_pad=0.5, h_pad=0.5)

plt.savefig(image_dir+"fig_2"+".pdf", bbox_inches='tight', dpi=300)
plt.savefig(image_dir+"fig_2"+".jpg", bbox_inches='tight', dpi=300)
