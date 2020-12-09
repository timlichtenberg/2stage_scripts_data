# Script from: https://github.com/timlichtenberg/2stage_scripts_data
# Part of the combined repository: https://osf.io/e2kfv/
import jd_readdata as readdata
import numpy
import scipy.stats as stats
import scipy.optimize as opt
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = "serif"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix" # Use stix with mathtext
plt.rcParams['font.size'] = 12

G_cgs = 6.674e-8
M_sun_grams = 1.989e33
G = 4 * np.pi**2
eta = 1e4

# To adjust ratio between viscosities
alpha_factor = 1

def au_to_cm(a):
    return a * 1.496e13

def cm_to_au(a):
    return a/1.496e13

def gpercm2_to_msunperau2(m):
    return m * 1.126e-7

def g_to_msun(m):
    return m * 5.03e-34

def seconds_to_years(s):
    return s *  3.171e-8

def L_to_a(L,M,m):
    print(L)
    print(M)
    print(G)
    print(m)
    return L**2/(m**2*G*M);
            
def optimise(q, R, alpha, omega, H):
    h = H/R
    reynolds = R**2 * omega/(alpha * H* H * omega)
    rh = R * (q/3.0)**0.33333333
    return 3.0/4.0 * H/rh + 50.0/(q * reynolds) - 1.0

class migration(object):
    def __init__(self, D = None, a = 1, m =  1 * 3e-6, t_init = 1e5):
        if D == None:
            self.D = readdata.disk()
            self.D.time =  np.logspace(3, 7, num=200)
            self.D.mstar = np.array([1 for ab in self.D.time])
            self.D.r = np.array(np.logspace(-2, 2, num=1300))
            sig = (self.D.r**-1.0) * gpercm2_to_msunperau2(1000)
            T   = (self.D.r**-0.5)    * 650
            self.D.sigma = np.array([sig for ab in self.D.time])
            self.D.tmid = np.array([T for ab in self.D.time])
            alph = [1e-3 for r in self.D.r]
            self.D.alpha = np.array([alph for ab in self.D.time])
            print(self.D.r.shape)
            print(self.D.sigma.shape)
            print(self.D.tmid.shape)
            print(self.D.alpha.shape)
            print(type(self.D.tmid))
        else:
            self.D = D
            # Convert from cm g s to au m_sun yr
            self.D.time = seconds_to_years(self.D.time)
            self.D.sigma = gpercm2_to_msunperau2(self.D.sigma)
            self.D.r = cm_to_au(self.D.r)
            self.D.mstar = g_to_msun(self.D.mstar)
            print(self.D.r.shape)
            print(self.D.sigma.shape)
            print(self.D.tmid.shape)
            print(self.D.alpha.shape)
            print(type(self.D.tmid))
        self.D.alpha = self.D.alpha*alpha_factor
        self.planet_a  = a
        self.planet_L  = np.sqrt(G * self.interpolate_at_T(self.D.mstar, t_init) * a) * m
        print("init")
        self.planet_m  = m
        self.T = t_init
        self.dt = 0
        self.gap = False

    def interpolate_at_T(self, quantity, T):
        t = self.T_to_t(T)
        print(t)
        print(self.D.time[t + 1])
        d_quantity = (quantity[t + 1] - quantity[t])/(self.D.time[t + 1] - self.D.time[t])
        return quantity[t] + d_quantity * (T - self.D.time[t])

    # Linear interpolation between neighbouring time-dumps
    def interpolate(self, quantity):
        t = self.T_to_t(self.T)
        print(t)
        d_quantity = (quantity[t + 1] - quantity[t])/(self.D.time[t + 1] - self.D.time[t])
        interp = quantity[t] + d_quantity * (self.T - self.D.time[t])
        return interp
        
    def r_to_i(self,rad):
        for i,r in enumerate(self.D.r):
            if r >= rad:
                return i - 1

    def T_to_t(self,rad):
        for i,r in enumerate(self.D.time):
            if r >= rad:
                return i - 1

    def estimate_density_powerlaw(self, i, spread = 3):
        sig = self.interpolate(self.D.sigma)
        i_minus = i - spread
        i_plus  = i + spread
        print("Planet at cell " +  str(i) + " radius of cell is " + str(self.D.r[i]))
        log_r   = np.log10(self.D.r[i_minus:i_plus])
        log_sig = np.log10(sig[i_minus:i_plus])
        fit = stats.linregress(log_r, log_sig)
        print("Slope = " + str(fit[0]))
        return(fit[0])
        
    def estimate_temperature_powerlaw(self, i, spread = 3):
        T = self.interpolate(self.D.tmid)
        i_minus = i - spread
        i_plus  = i + spread
        log_r   = np.log10(self.D.r[i_minus:i_plus])
        log_T = np.log10(T[i_minus:i_plus])
        fit = stats.linregress(log_r, log_T)
        print("Slope = " + str(fit[0]))
        return(fit[0])

    def compute_omega(self,i):
        return np.sqrt(G *  self.interpolate(self.D.mstar)/self.D.r[i]**3)
    
    def compute_cs(self,i):
        boltzmann_mu = (1.38065e-16/3.914e-24) # Boltzmann constant divided by proton mass, CGS units
        cs_cgs = np.sqrt(boltzmann_mu * self.interpolate(self.D.tmid)[i])
        cs = cs_cgs * 2.108e-6
        return cs

    def gap_opening(self,q, h, alpha):
        return 3.0**(4.0/3.0)/4.0 * h/q**(0.33333) + 50.0/q * alpha * h * h

    def unsaturated(self, q, h, alpha):
        q15 = q**(1.5)
        crit1 = 0.16 * q15/h**3.5
        crit2 = crit1/h
        print("Alpha viscosity:", alpha)
        if alpha > crit1 and alpha < crit2:
            print("Good!")
            return "standard"
        else:
            print("Regime crit1 = " + str(crit1) + " crit2 = "  + str(crit2) + " alpha = " + str(alpha))
            if alpha < crit1:
                return "saturated"
            elif alpha > crit2:
                return "linear"
            
    def lambda_0(self, i):
        omega = self.compute_omega(i)
        cs =    self.compute_cs(i)
        h = cs/omega/self.planet_a
        lambda_0 = self.interpolate(self.D.sigma)[i] * self.planet_a**4 * omega**2
        q = self.planet_m/self.interpolate(self.D.mstar)
        print("cs  " + str(cs) + " q " + str(q) + " omega " + str(omega) + " h " + str(h) + " sigma " + str(self.interpolate(self.D.sigma)[i]))
        alpha =  self.interpolate(self.D.alpha)[i]
        P = self.gap_opening(q, h, alpha)
        regime = self.unsaturated(q,h,alpha)
        print("P = " + str(P) + " alpha = " + str(alpha))
        if P < 1.0:
            print("Gap opening")
            R = self.planet_a
            H = h * self.planet_a
            rh = R * (q)**0.33333333
            reynolds = R**2 * omega/(alpha * H* H * omega)
            print( H/rh + 50/(q * reynolds))
            self.gap = True
        return q*q/(h*h) * lambda_0,regime
        
    def torque_fac(self,i, regime="standard"):
        density_slope =   -self.estimate_density_powerlaw(i)
        temp_slope    =   -self.estimate_temperature_powerlaw(i)
        factor = -0.85 - 1.0 * density_slope - 0.9 * temp_slope # Hands, Alexander 2018 version
        factor_lindblad = -2.5 - 1.7 * temp_slope + 0.1 * density_slope
        factor_corot    = 0
        if regime == "standard": # Locally isothermal EOS
            factor_corot    = 0.8 * temp_slope + 1.1 * (3.0/2.0 - density_slope)
        elif regime == "linear":
            factor_corot    = 0.8 * temp_slope + 0.7 * (3.0/2.0 - density_slope)
        else: # Corotation torque saturated, only Lindblad remains
            factor_corot    = 0
        print("paper version: " + str(factor))
        factor = factor_lindblad + factor_corot
        #factor = -1
        print("new version: " + str(factor))
        return factor
    
    def take_step(self):
        dt = self.dt
        print("======")
        i = self.r_to_i(self.planet_a)
        t = self.T_to_t(self.T)
        print("Sim t is " + str(t) + " or " + str(self.T) + "yr")
        L0,regime =  self.lambda_0(i)
        print("Regime check done")
        print("Regime: " + regime)
        tf = self.torque_fac(i, regime)
        torque =  tf * L0
        print("tf     = " + str(tf))
        print("Lambda = " + str(L0))
        print("Torque = " + str(torque))
        print("L      = " + str(self.planet_L))
        self.planet_L = self.planet_L + torque * dt
        self.planet_a = L_to_a(self.planet_L,  self.interpolate(self.D.mstar), self.planet_m)
        print("New a is " + str(self.planet_a))
        self.T = self.T + dt
        self.dt =  np.abs(self.planet_L/torque/eta)
        if self.dt > 5e5:
            print("Limiting dt")
            self.dt = 5e5
        print("New timestep is " + str(self.dt))

    def gap_map(self):
        rcut = 0
        for i,R in enumerate(self.D.r):
            if R > 100:
                rcut = i
                break
        start = 50
        f = open("data/gapmap.dat", "w")
        f.write("T, " + str(list(self.D.r[:rcut]))[1:-2] + "\n")
        for _t, T in enumerate(self.D.time[start:]):
            self.T = T
            t = _t + start
            Ms = []
            Ms_2 = []
            for i,R in enumerate(self.D.r[:rcut]):
                omega = self.compute_omega(i)
                cs =    self.compute_cs(i)
                alpha =  self.interpolate(self.D.alpha)[i]
                H = cs/omega
                h = H/R
                reynolds = R**2 * omega/(alpha * H* H * omega)
                X = np.sqrt(1.0 + 3.0 * reynolds * h*h*h/800.0)
                q = 100.0/reynolds * ((X + 1.0)**0.33333 - (X - 1.0)**0.33333)**(-3)
                rh = R * (q/3)**0.33333333
                p = opt.fsolve(optimise, 1e-8, args=(R, alpha, omega, H))
                M = q*self.D.mstar[t]
                Ms.append(M)
                Ms_2.append(p[0] *self.D.mstar[t])
            f.write(str(T) + ", " + str(Ms)[1:-2] + "\n")
        f.close()
                
        
    def run(self, show_plot = True):
        ts = []
        a = []
        a_init = self.planet_a
        print("a_init " + str(a_init))
        t_init = self.T
        i = self.r_to_i(self.planet_a)
        t = self.T_to_t(self.T)
        print("Sim t is " + str(t) + " or " + str(self.T) + "yr")
        
        L0,regime =  self.lambda_0(i)
        print("Regime: " + regime)
        tf = self.torque_fac(i, regime)
        torque =  tf * L0
        self.dt = np.abs(self.planet_L/torque/eta) #initial time-step estimate
        while self.planet_a > 0.3  and self.T_to_t(self.T) < len(self.D.time) - 2 and self.gap == False:#len(self.D.time):
            print("STEP START")
            print(self.D.time[-1])
            t = self.T_to_t(self.T)
            print(t)
            if t == None:
                print("Migration outlived the disc...")
                break
                #input()
            print(len(self.D.time))
            self.take_step()
            a.append(self.planet_a)
            ts.append(self.T)
        if self.gap:
            a.append(-1) #indicates a gap opened
            ts.append(self.T)
        plt.plot(ts, a)
        plt.xlabel("T/yr")
        plt.ylabel("a/au")
        plt.title("Planet mass " + str(self.planet_m/3e-6) + "$M_\mathrm{earth}$ (ended at " + str(a[-1]) + "au)")
        file = "data/migration/"+ str(t_init) + "yr" + str(self.planet_m/3e-6) +"Mearth" + str(a_init) + "au"
        if alpha_factor != 1:
            alpha = self.D.alpha[-1][-1]
            print("Alpha factor changed! Alpha:", alpha)
            file = file + str(alpha) + "alpha"
        plt.savefig(file + ".png")
        if show_plot:
            plt.show()
        plt.close()
        #dump track to disk
        f = open(file + ".dat", "w")
        f.write(str(ts)[1:-2])
        f.write("\n")
        f.write(str(a)[1:-2])
        f.close()

        
def main():
    for t in [ 0.7e6 ]:
        for M in [ 0.15 ]:
            for a in [ 7.0 ]:
                mig = migration(readdata.readdata(), t_init = t, a = a, m = M * 3e-6)
                mig.run(show_plot = False)
    # Other combinations:
    # for t in [5e5, 7.5e5, 8e5, 1e6, 1.3e6, 2e6, 3e6, 4e6]:
    #     for a in [7, 10, 12.5, 17]:
    #         for M in [0.01, 0.1, 0.3, 1.0, 10.0, 20.0, 50.0, 100.0, 318.0]:
    #             mig = migration(readdata.readdata(), t_init=t, a = a, m =M *  3e-6)#3.5e5)
    #             mig.run(show_plot = False)
    # for t in [1.0e6]:
    #     for a in [5]:
    #         for M in [0.1]:
    #             mig = migration(readdata.readdata(), t_init=t, a = a, m =M *  3e-6)#3.5e5)
    #             mig.run(show_plot = False)
def do_gap_map():
    mig = migration(readdata.readdata(), t_init=1e6, a = 8, m=1*3e-6)#3.5e5)
    mig.gap_map()


for alpha_factor in [ 1 ]: # 1, 1e-1, 1e-2
    main()

do_gap_map()
