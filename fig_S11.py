# Script from: https://github.com/timlichtenberg/2stage_scripts_data
# Part of the combined repository: https://osf.io/e2kfv/
from jd_natconst import *
from jd_readdata import *
from jd_plot import *

### ! REQUIRES: Ormel & Liu (2018) python script "OL18.py" in the main directory
### Download "epsilon.tar.gz" from https://staff.fnwi.uva.nl/c.w.ormel/software.html
### Unzip into main directory, then run this script
import OL18

q = readdata()

# Plot settings
lw      = 2
fscale  = 2.0
fsize   = 18

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

sns.set_style("whitegrid")
sns.set(style="ticks", font_scale=fscale)

ls         = "-"
ls_pebble  = "-"
ls_runaway = "--"

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13,11))

axes = plt.gca()
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

## Pebble flux unit
norm=1e6*year/Mea

embryo_radius_km_range = [ 100, 300, 427, 1800 ]

for a in [ 2, 15 ]: # au

    for embryo_radius_km in embryo_radius_km_range:

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
            color = qblue_light
            color_dark = qblue_dark

        if embryo_radius_km == embryo_radius_km_range[0]:
            ls = "--"
        if embryo_radius_km == embryo_radius_km_range[1]:
            ls = "-"
        if embryo_radius_km == embryo_radius_km_range[2]:
            ls = "-."
        if embryo_radius_km == embryo_radius_km_range[3]:
            ls = ":"


        label=str(a)+" au"

        idx_a               = find_nearest_idx(q.r, a*au)
        pebble_flux_old         = 2.*np.pi*q.r[idx_a]*q.sigmad[:,idx_a]*np.abs(q.vdust[:,idx_a])*norm
        
        rolling_time        = np.convolve(q.time/year, np.ones((N,))/N, mode='valid')

        pebble_flux        = np.abs(q.mflux[:,idx_a])*norm # directly derived from sim
        rolling_pebble_flux = np.convolve(pebble_flux, np.ones((N,))/N, mode='valid')

        pebble_flux_inward        = -1*q.mflux[:,idx_a]*norm # directly derived from sim
        rolling_pebble_flux_inward = np.convolve(pebble_flux_inward, np.ones((N,))/N, mode='valid')

        pebble_flux_outward        = q.mflux[:,idx_a]*norm # directly derived from sim
        rolling_pebble_flux_outward = np.convolve(pebble_flux_outward, np.ones((N,))/N, mode='valid')

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

        # When and where to calculate the pebble accretion efficiency 
        rdis = a*au
        pebble_growth_timescale = []
        runaway_growth_timescale = []
        embryo_radius = embryo_radius_km             # km
        embryo_mass   = m_plts(0.3, embryo_radius)   # g
        plts_radius   = 50                           # km
        plts_mass     = m_plts(0.3, plts_radius)     # g
        M_Ceres       = 8.96e+23                     # g

        print("Embryo mass / radius:", embryo_mass, embryo_radius)
        print("Plts mass / radius:", plts_mass, plts_radius)

        for tyear in time:

            ir = q.r.searchsorted(rdis)
            it = time.searchsorted(tyear)

            # properties of the accreting planet (Moon in here)
            Rpor = embryo_radius*1e+5 / q.r[ir] # planet radius over orbital radius; km->cm
            pmos = embryo_mass / q.mstar[it]    # planet mass over stellar mass

            # pebble accretion efficiency
            eps = OL18.epsilon_general(tau=q.stokesnr[it,ir], qp=pmos, eta=eta[it,ir], hgas=Hgas[it,ir], alphaz=1e-4, ip=0., nvec=(1,1,1), Rp=Rpor, mode='')

            # growth rate of the planet in g/s
            Mdot = eps * pebflux[it,ir]

            # PEBBLE growth timescale: how long will it take to double the mass
            tgrow = embryo_mass / Mdot
            tgrow = tgrow / year # years
            tgrow = tgrow / 1e+6 # Myr

            ## Pebble growth == 0 if no planetesimals!
            if q.sigmaplts[it,ir] >= 1e-4:
                pebble_growth_timescale.append(tgrow)
            else:
                pebble_growth_timescale.append(1e+99)

            #### PLANETESIMAL growth timescale
            pebble_enhancement_factor   = 1.

            tau_acc = calc_runaway_growth_timescale ( q.sigmaplts[it,ir]*pebble_enhancement_factor, q.r[ir]/au, embryo_mass*pebble_enhancement_factor/Mea, q.mstar[it]/MS, q.sigma[it, ir], plts_mass*pebble_enhancement_factor )
            tau_acc = tau_acc / 1e+6 # Myr

            runaway_growth_timescale.append(tau_acc) 

        # filter out NaNs
        pebble_growth_timescale = [0 if math.isnan(x) else x for x in pebble_growth_timescale]
        
        rolling_time = np.convolve(time, np.ones((N,))/N, mode='valid')
        rolling_pebble_growth_timescale = np.convolve(pebble_growth_timescale, np.ones((N,))/N, mode='valid')
        rolling_runaway_growth_timescale = np.convolve(runaway_growth_timescale, np.ones((N,))/N, mode='valid')

        ax1.plot(rolling_time, rolling_runaway_growth_timescale, ls=ls, c=color, lw=lw, label=str(embryo_radius_km))

        M_label = str(round(embryo_mass/M_Ceres,2))
        if embryo_mass/M_Ceres >= 1:
            M_label = str(int(round(embryo_mass/M_Ceres,0)))

        ax2.plot(rolling_time, rolling_pebble_growth_timescale, ls=ls, c=color, lw=lw, label=M_label)

###### Annotations

N = 1000

ax1.text(0.6e+6, 4.8, r'$\approx$ disk lifetime', color=qgray_dark, rotation=0, ha="right", va="center", fontsize=fsize-5)
ax1.axhspan(4, 6, color=qgray_light, alpha=.3, lw=0)
ax2.axhspan(4, 6, color=qgray_light, alpha=.3, lw=0)


ax1.text(0.6e+6, 20, r'$r$ = 2 au', color=qred, rotation=0, ha="center", va="bottom", fontsize=fsize-2)
ax1.text(0.77e+6, 300, r'$r$ = 15 au', color=qblue_light, rotation=0, ha="center", va="bottom", fontsize=fsize-2)
ax2.text(0.6e+6, 20, r'$r$ = 2 au', color=qred, rotation=0, ha="center", va="bottom", fontsize=fsize-2)
ax2.text(0.77e+6, 300, r'$r$ = 15 au', color=qblue_light, rotation=0, ha="center", va="bottom", fontsize=fsize-2)

ax1.text(0.4e+6, 8e-2, 'Collision-dominated growth', color=qred_dark, rotation=0, ha="center", va="bottom", fontsize=fsize-4, bbox=dict(fc='white', ec="white", alpha=0.5, pad=0.1, boxstyle='round'))
ax2.text(1.05e+6, 8e-2, 'Pebble-aided growth', color=qred_dark, rotation=0, ha="center", va="bottom", fontsize=fsize-4, bbox=dict(fc='white', ec="white", alpha=0.5, pad=0.1, boxstyle='round'))

ax1.text(0.985, 0.95, 'A', color="k", rotation=0, ha="right", va="center", fontsize=fsize+4, transform=ax1.transAxes, bbox=dict(fc='white', ec="white", alpha=0.5, pad=0.1, boxstyle='round'))
ax2.text(0.015, 0.95, 'B', color="k", rotation=0, ha="left", va="center", fontsize=fsize+4, transform=ax2.transAxes, bbox=dict(fc='white', ec="white", alpha=0.5, pad=0.1, boxstyle='round'))
ax1.text(0.95, 0.95, 'Planetesimal accretion', color="k", rotation=0, ha="right", va="center", fontsize=fsize-2, transform=ax1.transAxes, bbox=dict(fc='white', ec="white", alpha=0.5, pad=0.1, boxstyle='round'), zorder=10)
ax2.text(0.05, 0.95, 'Pebble accretion', color="k", rotation=0, ha="left", va="center", fontsize=fsize-2, transform=ax2.transAxes, bbox=dict(fc='white', ec="white", alpha=0.5, pad=0.1, boxstyle='round'), zorder=10)

sns.despine(ax=ax1, top=True, right=True, left=False, bottom=False)
sns.despine(ax=ax2, top=True, right=True, left=False, bottom=False)

lwa=1.5

ax1.annotate('', xy=(0.525, 0.8), xycoords='axes fraction', xytext=(0.525, 0.65), textcoords='axes fraction', arrowprops={'arrowstyle': '-|>,head_length=0.2,head_width=0.1', 'facecolor': qgray_dark, 'edgecolor': qgray_dark, 'lw': lwa}, va='center', transform=ax1.transAxes)
ax1.text(0.53, 0.71, r'Larger $R_{\mathrm{embryo}}$ / $M_{\mathrm{embryo}}$', color=qgray_dark, rotation=0, ha="left", va="center", fontsize=fsize-5, transform=ax1.transAxes)

ax1.annotate('', xy=(0.525, 0.1), xycoords='axes fraction', xytext=(0.525, 0.3), textcoords='axes fraction', arrowprops={'arrowstyle': '-|>,head_length=0.2,head_width=0.1', 'facecolor': qgray_dark, 'edgecolor': qgray_dark, 'lw': lwa}, va='center', transform=ax1.transAxes)
ax1.text(0.53, 0.19, r'Smaller $R_{\mathrm{P}}$', color=qgray_dark, rotation=0, ha="left", va="center", fontsize=fsize-5, transform=ax1.transAxes)

ax1.text(0.99, 0.01, r'Accreting planetesimals: $R_{\mathrm{P}}$ = 50 km $\approx$'+str(round(plts_mass/M_Ceres,3))+r" $M_\mathrm{Ceres}$", color=qgray_dark, rotation=0, ha="right", va="bottom", fontsize=fsize-6, transform=ax1.transAxes)

ax2.annotate('', xy=(0.7, 0.15), xycoords='axes fraction', xytext=(0.7, 0.35), textcoords='axes fraction', arrowprops={'arrowstyle': '-|>,head_length=0.2,head_width=0.1', 'facecolor': qgray_dark, 'edgecolor': qgray_dark, 'lw': lwa}, va='center', transform=ax2.transAxes)
ax2.text(0.705, 0.25, r'Larger $R_{\mathrm{embryo}}$ / $M_{\mathrm{embryo}}$', color=qgray_dark, rotation=0, ha="left", va="center", fontsize=fsize-5, transform=ax2.transAxes)


###### PLOT SETTINGS
legend1 = ax1.legend(fontsize=fsize-5, loc=2, ncol=2, title=r"$R_\mathrm{embryo}$ (km)")
plt.setp(legend1.get_title(), fontsize=fsize-3)

legend2 = ax2.legend(fontsize=fsize-5, loc=3, ncol=2, title=r"$M_\mathrm{embryo}$ ($M_\mathrm{Ceres}$)")
plt.setp(legend2.get_title(), fontsize=fsize-3)

x_left  = 2.2e+5
x_right = 4.2e+6
xticks  = [3e+5, 5e+5, 7e+5, 1e+6, 2e+6, 3e+6, 4e+6, 5e+6]
xticklabels = ["0.3", "0.5", "0.7", "1", "2", "3", "4", "5"]

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylabel(r'Growth timescale (Myr)', fontsize=fsize)
ax1.set_xlim([x_left, x_right])
ax1.set_xticks(xticks)
ax1.tick_params(axis="both", labelsize=fsize)

# Growth timescale y axis
y_top       = 1e+3
y_bottom    = 3e-2
ax1.set_ylim(bottom=y_bottom, top=y_top)
ax2.set_ylim(bottom=y_bottom, top=y_top)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylabel(r'Growth timescale (Myr)', fontsize=fsize)
ax2.set_xlim([x_left, x_right])
ax2.set_xticks(xticks)
ax2.set_xticklabels(xticklabels)

ax2.tick_params(axis="both", labelsize=fsize)
ax2.set_xlabel(r'Time after CAIs, $\Delta t_{\mathrm{CAI}}\ (\rm{Myr})$', fontsize=fsize)

plt.setp(ax1.get_xticklabels(), visible=False)

plt.tight_layout(pad=2.5, w_pad=0.5, h_pad=0.5)

plt.savefig(image_dir+"fig_s11"+".pdf", bbox_inches='tight')
plt.savefig(image_dir+"fig_s11"+".jpg", bbox_inches='tight', dpi=300)

