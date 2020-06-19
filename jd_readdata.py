import numpy as np
import math
import os
from jd_natconst import *

working_dir  = os.getcwd()
dat_d18a_dir = working_dir+"/data/Drazkowska_Dullemond_2018/"
# dat_d18a_dir = "/Users/tim/Dropbox/work/Projects/19_2stage/dat/drazkowska18a/"
image_dir    = working_dir+"/figures/"
fig_dir      = image_dir

class disk(object):
    """
    Data container of the results of the viscous disk model.
    """
    def __init__(self):
        self.r         = np.zeros(0, dtype=np.float64)      # radius
        self.time      = np.zeros(0, dtype=np.float64)      # time
        self.sigma     = np.zeros((0,0), dtype=np.float64)  # surface density gas
        self.sigmad    = np.zeros((0,0), dtype=np.float64)  # surface density pebbles
        self.sice      = np.zeros((0,0), dtype=np.float64)  # surface density of ice --> ice fraction of planetesimals: sice/sigmad
        self.sigmavap  = np.zeros((0,0), dtype=np.float64)  # surface density water vapor
        self.sigmaplts = np.zeros((0,0), dtype=np.float64)  # surface density planetesimals
        self.dmplts    = np.zeros((0,0), dtype=np.float64)  # used to calculate dmdt
        self.dmdt      = np.zeros((0,0), dtype=np.float64)  # dmplts / dt
        self.dplts     = np.zeros((0,0), dtype=np.float64)  # used to calculate dSigmadt
        self.dSigmadt  = np.zeros((0,0), dtype=np.float64)  # dSigmaplts / dt
        self.etamid    = np.zeros((0,0), dtype=np.float64)  # midplane dust-to-gas ratio
        self.dustsize  = np.zeros((0,0), dtype=np.float64)  # representative size of dust
        self.stokesnr  = np.zeros((0,0), dtype=np.float64)  # stokes number
        self.tmid      = np.zeros((0,0), dtype=np.float64)  # midplane temperature of gas
        self.mdot      = np.zeros((0,0), dtype=np.float64)  # mass flow of gas through the disk cell
        self.mflux     = np.zeros((0,0), dtype=np.float64)  # dust mass flux (g/s) through cell interfaces
        self.mstar     = np.zeros(0, dtype=np.float64)      # mass of star
        self.nu        = np.zeros((0,0), dtype=np.float64)  # gas viscosity
        self.vr        = np.zeros((0,0), dtype=np.float64)  # gas velocity
        self.vdust     = np.zeros((0,0), dtype=np.float64)  # dust velocity
        self.alpha     = np.zeros((0,0), dtype=np.float64)  # disk viscosity
        self.mdisk     = np.zeros(0, dtype=np.float64)      # total mass of gas
        self.mdust     = np.zeros(0, dtype=np.float64)      # total mass of dust
        self.mplts     = np.zeros(0, dtype=np.float64)      # total mass of planetesimals
        self.mvap      = np.zeros(0, dtype=np.float64)      # mass of water vapor
        self.rsnow     = np.zeros(0, dtype=np.float64)      # position of snowline
        self.gapmap    = np.zeros((0,0), dtype=np.float64)  # TH data: minimum mass to open gap in disk at location and time

    def read(self):
        with open(dat_d18a_dir+'time.info','r') as f:
            ntime = int(f.readline()) + 1
        # ntime = 201
        with open(dat_d18a_dir+'time.dat','r') as f:
            self.time = np.zeros(ntime)
            f.readline()
            for itime in range(ntime):
                self.time[itime] = float(f.readline())
        with open(dat_d18a_dir+'grid.info','r') as f:
            nr = int(f.readline())
            f.readline()
            self.r = np.zeros(nr)
            for ir in range(nr):
                self.r[ir]=float(f.readline())
        self.sigma = np.zeros((ntime,nr))
        with open(dat_d18a_dir+'sigma.dat','r') as f:
            f.readline()
            for itime in range(ntime):
                f.readline()
                for ir in range(nr):
                    self.sigma[itime,ir] = float(f.readline())
        self.sigmad = np.zeros((ntime,nr))
        with open(dat_d18a_dir+'sigmad.dat','r') as f:
            f.readline()
            for itime in range(ntime):
                f.readline()
                for ir in range(nr):
                    self.sigmad[itime,ir] = float(f.readline())
        self.mflux = np.zeros((ntime,nr))
        with open(dat_d18a_dir+'mflux.dat','r') as f:
            f.readline()
            for itime in range(ntime):
                f.readline()
                for ir in range(nr):
                    self.mflux[itime,ir] = float(f.readline())
        self.sigmaplts = np.zeros((ntime,nr))
        with open(dat_d18a_dir+'sigmaplts.dat','r') as f:
            f.readline()
            for itime in range(ntime):
                f.readline()
                for ir in range(nr):
                    self.sigmaplts[itime,ir] = float(f.readline())
        self.sigmavap = np.zeros((ntime,nr))
        with open(dat_d18a_dir+'sigmavap.dat','r') as f:
            f.readline()
            for itime in range(ntime):
                f.readline()
                for ir in range(nr):
                    self.sigmavap[itime,ir] = float(f.readline())
        self.sice = np.zeros((ntime,nr))
        with open(dat_d18a_dir+'sigmaice.dat','r') as f:
            f.readline()
            for itime in range(ntime):
                f.readline()
                for ir in range(nr):
                    self.sice[itime,ir] = float(f.readline())
        self.etamid = np.zeros((ntime,nr))
        with open(dat_d18a_dir+'etamid.dat','r') as f:
            f.readline()
            for itime in range(ntime):
                f.readline()
                for ir in range(nr):
                    self.etamid[itime,ir] = float(f.readline())
        self.dustsize = np.zeros((ntime,nr))
        with open(dat_d18a_dir+'dustsize.dat','r') as f:
            f.readline()
            for itime in range(ntime):
                f.readline()
                for ir in range(nr):
                    self.dustsize[itime,ir] = float(f.readline())
        self.stokesnr = np.zeros((ntime,nr))
        with open(dat_d18a_dir+'stokesnr.dat','r') as f:
            f.readline()
            for itime in range(ntime):
                f.readline()
                for ir in range(nr):
                    self.stokesnr[itime,ir] = float(f.readline())
        self.mdot = np.zeros((ntime,nr))
        with open(dat_d18a_dir+'mdot.dat','r') as f:
            f.readline()
            for itime in range(ntime):
                f.readline()
                for ir in range(nr):
                    self.mdot[itime,ir] = float(f.readline())
        self.tmid = np.zeros((ntime,nr))
        with open(dat_d18a_dir+'temperature.dat','r') as f:
            f.readline()
            for itime in range(ntime):
                f.readline()
                for ir in range(nr):
                    self.tmid[itime,ir] = float(f.readline())
        self.nu = np.zeros((ntime,nr))
        with open(dat_d18a_dir+'visc.dat','r') as f:
            f.readline()
            for itime in range(ntime):
                f.readline()
                for ir in range(nr):
                    self.nu[itime,ir] = float(f.readline())
        self.vr = np.zeros((ntime,nr))
        with open(dat_d18a_dir+'velo.dat','r') as f:
            f.readline()
            for itime in range(ntime):
                f.readline()
                for ir in range(nr):
                    self.vr[itime,ir] = float(f.readline())
        self.vdust = np.zeros((ntime,nr))
        with open(dat_d18a_dir+'vdust.dat','r') as f:
            f.readline()
            for itime in range(ntime):
                f.readline()
                for ir in range(nr):
                    self.vdust[itime,ir] = float(f.readline())
        self.alpha = np.zeros((ntime,nr))
        with open(dat_d18a_dir+'alpha.dat','r') as f:
            f.readline()
            for itime in range(ntime):
                f.readline()
                for ir in range(nr):
                    self.alpha[itime,ir] = float(f.readline())
        self.mstar = np.zeros((ntime))
        with open(dat_d18a_dir+'mstar.dat','r') as f:
            f.readline()
            for itime in range(ntime):
                self.mstar[itime] = float(f.readline())
        self.mdisk = np.zeros(ntime)
        for itime in range(ntime):
            tprsigma = self.sigma[itime,:]*2*math.pi*self.r
            tprsav   = 0.5*(tprsigma[1:]+tprsigma[:-1])
            dr       = self.r[1:]-self.r[:-1]
            dum      = tprsav*dr
            self.mdisk[itime] = dum.sum()
        self.mdust = np.zeros(ntime)
        for itime in range(ntime):
            tprsigmad = (self.sigmad[itime,:])*2*math.pi*self.r
            tprsav   = 0.5*(tprsigmad[1:]+tprsigmad[:-1])
            dr       = self.r[1:]-self.r[:-1]
            dum      = tprsav*dr
            self.mdust[itime] = dum.sum()
        self.mplts = np.zeros(ntime)
        for itime in range(ntime):
            tprsigmap = (self.sigmaplts[itime,:])*2*math.pi*self.r
            tprsav   = 0.5*(tprsigmap[1:]+tprsigmap[:-1])
            dr       = self.r[1:]-self.r[:-1]
            dum      = tprsav*dr
            self.mplts[itime] = dum.sum()
        self.mvap = np.zeros(ntime)
        for itime in range(ntime):
            tprsigmav = (self.sigmavap[itime,:])*2*math.pi*self.r
            tprsav   = 0.5*(tprsigmav[1:]+tprsigmav[:-1])
            dr       = self.r[1:]-self.r[:-1]
            dum      = tprsav*dr
            self.mvap[itime] = dum.sum()
        self.dplts  	= np.zeros((ntime,nr))
        self.dSigmadt   = np.zeros((ntime,nr))
        bla = np.zeros((nr))
        for itime in range(1,ntime):
            for ir in range(nr):
                self.dplts[itime,ir] = (self.sigmaplts[itime,ir] - self.sigmaplts[itime-1,ir])
                bla[ir] = self.dplts[itime,ir]/(self.time[itime]-self.time[itime-1])
            for ir in range(1,nr):
                self.dSigmadt[itime,ir] =  bla[ir]
        self.dmplts = np.zeros((ntime,nr))
        self.dmdt   = np.zeros((ntime,nr))
        bla = np.zeros((nr))
        self.rsnow = np.zeros(ntime)
        for itime in range(1,ntime):
            for ir in range(nr):
                self.dmplts[itime,ir] = (self.sigmaplts[itime,ir] - self.sigmaplts[itime-1,ir])*math.pi*((self.r[ir] * self.r[ir]) - (self.r[ir-1] * self.r[ir-1])) # are -1 notation right, or should it be +1 first and no -1 then?
                bla[ir] = self.dmplts[itime,ir]/(self.time[itime]-self.time[itime-1])
            for ir in range(1,nr):
                self.dmdt[itime,ir] =  bla[ir]
            ir = 10
            while (self.sice[itime,ir] < 0.01*self.sigmad[itime,ir]):
                ir = ir + 1
            self.rsnow[itime] = self.r[ir]


        ##################################################################################
        ##################################################################################
        ##################################################################################
        ######### Additional fields: TL #########
        #### Reservoir I: tform < 0.5 Myr
        #### Reservoir II: 0.5 Myr < tform < 5.0 Myr
        #### Reservoir III: 0.5 Myr < tform < 2.0 Myr, r > dju
        #### Reservoir IV: 0.5 Myr < tform
        #### Reservoir V: 0.5 Myr < tform < 2.0 Myr
        #### Reservoir VI: 2.0 Myr < tform < 5.0 Myr

        self.sigmaplts_resI 	= np.zeros((ntime,nr))
        self.sigmaplts_resII 	= np.zeros((ntime,nr))
        self.sigmaplts_resIII 	= np.zeros((ntime,nr))
        self.sigmaplts_resIV    = np.zeros((ntime,nr))
        self.sigmaplts_resV     = np.zeros((ntime,nr))
        self.sigmaplts_resVI    = np.zeros((ntime,nr))
        self.mplts_resI 		= np.zeros(ntime)
        self.mplts_resII 		= np.zeros(ntime)
        self.mplts_resIII 		= np.zeros(ntime)
        self.mplts_resIV        = np.zeros(ntime)
        self.mplts_resV         = np.zeros(ntime)
        self.mplts_resVI        = np.zeros(ntime)
        self.dmplts_resI 		= np.zeros((ntime,nr))
        self.dmplts_resII 		= np.zeros((ntime,nr))
        self.dmplts_resIII 		= np.zeros((ntime,nr))
        self.dmplts_resIV       = np.zeros((ntime,nr))
        self.dmplts_resV        = np.zeros((ntime,nr))
        self.dmplts_resVI       = np.zeros((ntime,nr))
        self.dmpltstot_resI 	= np.zeros((ntime,nr))
        self.dmpltstot_resII 	= np.zeros((ntime,nr))
        self.dmpltstot_resIII 	= np.zeros((ntime,nr))
        self.dmpltstot_resIV    = np.zeros((ntime,nr))
        self.dmpltstot_resV     = np.zeros((ntime,nr))
        self.dmpltstot_resVI    = np.zeros((ntime,nr))
        self.dmdt_resI   		= np.zeros((ntime,nr))
        self.dmdt_resII   		= np.zeros((ntime,nr))
        self.dmdt_resIII   		= np.zeros((ntime,nr))
        self.dmdt_resIV         = np.zeros((ntime,nr))
        self.dmdt_resV          = np.zeros((ntime,nr))
        self.dmdt_resVI         = np.zeros((ntime,nr))
        self.dplts_resI      	= np.zeros((ntime,nr))
        self.dplts_resII 	    = np.zeros((ntime,nr))
        self.dplts_resIII 	    = np.zeros((ntime,nr))
        self.dplts_resIV        = np.zeros((ntime,nr))
        self.dplts_resV         = np.zeros((ntime,nr))
        self.dplts_resVI        = np.zeros((ntime,nr))
        self.dSigmadt_resI   	= np.zeros((ntime,nr))
        self.dSigmadt_resII   	= np.zeros((ntime,nr))
        self.dSigmadt_resIII   	= np.zeros((ntime,nr))
        self.dSigmadt_resIV     = np.zeros((ntime,nr))
        self.dSigmadt_resV      = np.zeros((ntime,nr))
        self.dSigmadt_resVI     = np.zeros((ntime,nr))
        bla_resI 				= np.zeros((nr))
        bla_resII 				= np.zeros((nr))
        bla_resIII 				= np.zeros((nr))
        bla_resIV               = np.zeros((nr))
        bla_resV                = np.zeros((nr))
        bla_resVI               = np.zeros((nr))
        blam_resI 				= np.zeros((nr))
        blam_resII 				= np.zeros((nr))
        blam_resIII 			= np.zeros((nr))
        blam_resIV              = np.zeros((nr))
        blam_resV               = np.zeros((nr))
        blam_resVI              = np.zeros((nr))
        dr       				= self.r[1:]-self.r[:-1]

        with open(dat_d18a_dir+'sigmaplts.dat','r') as f:

            f.readline()

            for itime in range(ntime):

                f.readline()

                for ir in range(nr):

                    entry = float(f.readline())

                    #### Reservoir I: tform < 0.6 Myr
                    if ( self.time[itime]/(year*1e+6) < 0.6 ):
                        self.sigmaplts_resI[itime,ir] = entry
                    else:
                        self.sigmaplts_resI[itime,ir] = self.sigmaplts_resI[itime-1,ir]

                    #### Reservoir II: 0.6 Myr < tform < 5.0 Myr
                    if ( self.time[itime]/(year*1e+6) > 0.6 ) and ( self.time[itime]/(year*1e+6) <= 5.0 ):
                        self.sigmaplts_resII[itime,ir] = entry - self.sigmaplts[71,ir]
                    else:
                        self.sigmaplts_resII[itime,ir] = 0.

                    #### Reservoir III: 0.5 Myr < tform < 5.0 Myr, r > dju
                    if ( self.time[itime]/(year*1e+6) > 0.5 ) and ( self.time[itime]/(year*1e+6) <= 10.0 ) and ( self.r[ir] > dju ):
                        self.sigmaplts_resIII[itime,ir] = entry - self.sigmaplts[71,ir]
                    else:
                        self.sigmaplts_resIII[itime,ir] = 0.

                    #### Reservoir IV: 0.5 Myr < tform
                    if ( self.time[itime]/(year*1e+6) > 0.5 ) and ( self.time[itime]/(year*1e+6) <= 10.0 ):
                        self.sigmaplts_resII[itime,ir] = entry - self.sigmaplts[71,ir]
                    else:
                        self.sigmaplts_resII[itime,ir] = 0.

                    #### Reservoir V: 0.5 Myr < tform < 2.0 Myr
                    if ( self.time[itime]/(year*1e+6) > 0.5 ) and ( self.time[itime]/(year*1e+6) <= 2.0 ):
                        self.sigmaplts_resV[itime,ir] = entry
                    else:
                        self.sigmaplts_resV[itime,ir] = self.sigmaplts_resV[itime-1,ir]

                    #### Reservoir VI: 2.0 Myr < tform < 5.0 Myr
                    if ( self.time[itime]/(year*1e+6) > 2.0 ) and ( self.time[itime]/(year*1e+6) <= 5.0 ):
                        self.sigmaplts_resVI[itime,ir] = entry
                    else:
                        self.sigmaplts_resVI[itime,ir] = self.sigmaplts_resVI[itime-1,ir]

        for itime in range(ntime):

            tprsigmap 		= (self.sigmaplts_resI[itime,:])*2*math.pi*self.r
            tprsav   		= 0.5*(tprsigmap[1:]+tprsigmap[:-1])
            dum      		= tprsav*dr
            self.mplts_resI[itime] = dum.sum()

            tprsigmap 		= (self.sigmaplts_resII[itime,:])*2*math.pi*self.r
            tprsav   		= 0.5*(tprsigmap[1:]+tprsigmap[:-1])
            dum      		= tprsav*dr
            self.mplts_resII[itime] = dum.sum()

            tprsigmap 		= (self.sigmaplts_resIII[itime,:])*2*math.pi*self.r
            tprsav   		= 0.5*(tprsigmap[1:]+tprsigmap[:-1])
            dum      		= tprsav*dr
            self.mplts_resIII[itime] = dum.sum()

            tprsigmap       = (self.sigmaplts_resIV[itime,:])*2*math.pi*self.r
            tprsav          = 0.5*(tprsigmap[1:]+tprsigmap[:-1])
            dum             = tprsav*dr
            self.mplts_resIV[itime] = dum.sum()

            tprsigmap       = (self.sigmaplts_resV[itime,:])*2*math.pi*self.r
            tprsav          = 0.5*(tprsigmap[1:]+tprsigmap[:-1])
            dum             = tprsav*dr
            self.mplts_resV[itime] = dum.sum()

            tprsigmap       = (self.sigmaplts_resVI[itime,:])*2*math.pi*self.r
            tprsav          = 0.5*(tprsigmap[1:]+tprsigmap[:-1])
            dum             = tprsav*dr
            self.mplts_resVI[itime] = dum.sum()

            self.dmpltstot_resI[itime]   = self.mplts_resI[itime]-self.mplts_resI[itime-1]
            self.dmpltstot_resII[itime]  = self.mplts_resII[itime]-self.mplts_resII[itime-1]
            self.dmpltstot_resIII[itime] = self.mplts_resIII[itime]-self.mplts_resIII[itime-1]
            self.dmpltstot_resIV[itime]  = self.mplts_resIV[itime]-self.mplts_resIV[itime-1]
            self.dmpltstot_resV[itime]   = self.mplts_resV[itime]-self.mplts_resV[itime-1]
            self.dmpltstot_resVI[itime]  = self.mplts_resVI[itime]-self.mplts_resVI[itime-1]



        # self.dplts  	= np.zeros((ntime,nr))
        # self.dSigmadt   = np.zeros((ntime,nr))
        # bla = np.zeros((nr))

        # for itime in range(1,ntime):

        #     for ir in range(nr):
        #         self.dplts_resI[itime,ir] = (self.sigmaplts_resI[itime,ir] - self.sigmaplts_resI[itime-1,ir])
        #         bla_resI[ir] = self.dplts_resI[itime,ir]/(self.time[itime]-self.time[itime-1])

        #     for ir in range(1,nr):
        #         self.dSigmadt_resI[itime,ir] =  bla_resI[ir]

       	# self.dmdt_resI   = np.zeros((ntime,nr))
        # bla_resI = np.zeros((nr))
        # for itime in range(1,ntime):
        #     for ir in range(nr):
        #         self.dmplts_resI[itime,ir] = (self.sigmaplts_resI[itime,ir] - self.sigmaplts_resI[itime-1,ir])*math.pi*((self.r[ir] * self.r[ir]) - (self.r[ir-1] * self.r[ir-1])) # are -1 notation right, or should it be +1 first and no -1 then?
        #         bla_resI[ir] = self.dmplts_resI[itime,ir]/(self.time[itime]-self.time[itime-1])
        #     for ir in range(1,nr):
        #         self.dmdt_resI[itime,ir] =  bla[ir]

        for itime in range(1,ntime):

            for ir in range(nr):

                # are -1 notation right, or should it be +1 first and no -1 then?
                self.dmplts_resI[itime,ir] = (self.sigmaplts_resI[itime,ir] - self.sigmaplts_resI[itime-1,ir])*math.pi*((self.r[ir] * self.r[ir]) - (self.r[ir-1] * self.r[ir-1]))
                self.dmplts_resII[itime,ir] = (self.sigmaplts_resII[itime,ir] - self.sigmaplts_resII[itime-1,ir])*math.pi*((self.r[ir] * self.r[ir]) - (self.r[ir-1] * self.r[ir-1]))
                self.dmplts_resIII[itime,ir] = (self.sigmaplts_resIII[itime,ir] - self.sigmaplts_resIII[itime-1,ir])*math.pi*((self.r[ir] * self.r[ir]) - (self.r[ir-1] * self.r[ir-1]))
                self.dmplts_resIV[itime,ir] = (self.sigmaplts_resIV[itime,ir] - self.sigmaplts_resIV[itime-1,ir])*math.pi*((self.r[ir] * self.r[ir]) - (self.r[ir-1] * self.r[ir-1]))
                self.dmplts_resV[itime,ir] = (self.sigmaplts_resV[itime,ir] - self.sigmaplts_resV[itime-1,ir])*math.pi*((self.r[ir] * self.r[ir]) - (self.r[ir-1] * self.r[ir-1]))
                self.dmplts_resVI[itime,ir] = (self.sigmaplts_resVI[itime,ir] - self.sigmaplts_resVI[itime-1,ir])*math.pi*((self.r[ir] * self.r[ir]) - (self.r[ir-1] * self.r[ir-1]))

                self.dplts_resI[itime,ir] = (self.sigmaplts_resI[itime,ir] - self.sigmaplts_resI[itime-1,ir])
                self.dplts_resII[itime,ir] = (self.sigmaplts_resII[itime,ir] - self.sigmaplts_resII[itime-1,ir])
                self.dplts_resIII[itime,ir] = (self.sigmaplts_resIII[itime,ir] - self.sigmaplts_resIII[itime-1,ir])
                self.dplts_resIV[itime,ir] = (self.sigmaplts_resIV[itime,ir] - self.sigmaplts_resIV[itime-1,ir])
                self.dplts_resV[itime,ir] = (self.sigmaplts_resV[itime,ir] - self.sigmaplts_resV[itime-1,ir])
                self.dplts_resVI[itime,ir] = (self.sigmaplts_resVI[itime,ir] - self.sigmaplts_resVI[itime-1,ir])

                blam_resI[ir]   = self.dmplts_resI[itime,ir]/(self.time[itime]-self.time[itime-1])
                blam_resII[ir]  = self.dmplts_resII[itime,ir]/(self.time[itime]-self.time[itime-1])
                blam_resIII[ir] = self.dmplts_resIII[itime,ir]/(self.time[itime]-self.time[itime-1])
                blam_resIV[ir]  = self.dmplts_resIV[itime,ir]/(self.time[itime]-self.time[itime-1])
                blam_resV[ir]   = self.dmplts_resV[itime,ir]/(self.time[itime]-self.time[itime-1])
                blam_resVI[ir]  = self.dmplts_resVI[itime,ir]/(self.time[itime]-self.time[itime-1])

                bla_resI[ir]    = self.dplts_resI[itime,ir]/(self.time[itime]-self.time[itime-1])
                bla_resII[ir]   = self.dplts_resII[itime,ir]/(self.time[itime]-self.time[itime-1])
                bla_resIII[ir]  = self.dplts_resIII[itime,ir]/(self.time[itime]-self.time[itime-1])
                bla_resIV[ir]   = self.dplts_resIV[itime,ir]/(self.time[itime]-self.time[itime-1])
                bla_resV[ir]    = self.dplts_resV[itime,ir]/(self.time[itime]-self.time[itime-1])
                bla_resVI[ir]   = self.dplts_resVI[itime,ir]/(self.time[itime]-self.time[itime-1])


            for ir in range(1,nr):

                self.dmdt_resI[itime,ir] 		=  blam_resI[ir]
                self.dmdt_resII[itime,ir] 		=  blam_resII[ir]
                self.dmdt_resIII[itime,ir] 		=  blam_resIII[ir]
                self.dmdt_resIV[itime,ir]       =  blam_resIV[ir]
                self.dmdt_resV[itime,ir]        =  blam_resV[ir]
                self.dmdt_resVI[itime,ir]       =  blam_resVI[ir]

                self.dSigmadt_resI[itime,ir] 	=  bla_resI[ir]
                self.dSigmadt_resII[itime,ir] 	=  bla_resII[ir]
                self.dSigmadt_resIII[itime,ir] 	=  bla_resIII[ir]
                self.dSigmadt_resIV[itime,ir]   =  bla_resIV[ir]
                self.dSigmadt_resV[itime,ir]    =  bla_resV[ir]
                self.dSigmadt_resVI[itime,ir]   =  bla_resVI[ir]

        ### Reservoir II: tform > 0.5 Myr, r > dju

        # with open(dat_d18a_dir+'sigmaplts.dat','r') as f:
        #     f.readline()
        #     for itime in range(ntime):
        #         f.readline()
        #         for ir in range(nr): # and ( self.r[ir] < dju ):
        #             entry = float(f.readline())
        #             if ( self.time[itime]/(year*1e+6) > 0.5 ) and ( self.r[ir] > dju ):
        #                 self.sigmaplts_resII[itime,ir] = entry - self.sigmaplts[71,ir]
        #             else:
        #                 self.sigmaplts_resII[itime,ir] = 0.

        # bla = np.zeros((nr))
        # for itime in range(1,ntime):
        #     for ir in range(nr):
        #         self.dmplts_resII[itime,ir] = (self.sigmaplts_resII[itime,ir] - self.sigmaplts_resII[itime-1,ir])*math.pi*((self.r[ir] * self.r[ir]) - (self.r[ir-1] * self.r[ir-1])) # are -1 notation right, or should it be +1 first and no -1 then?
        #         bla[ir] = self.dmplts_resII[itime,ir]/(self.time[itime]-self.time[itime-1])
        #     for ir in range(1,nr):
        #         self.dmdt_resII[itime,ir] =  bla[ir]

        # Reservoir III: tform > 0.5 Myr, r < dju (, tform < 4.0 Myr)

        # with open(dat_d18a_dir+'sigmaplts.dat','r') as f:
        #     f.readline()
        #     for itime in range(ntime):
        #         f.readline()
        #         for ir in range(nr):
        #             entry = float(f.readline())
        #             if ( self.time[itime]/(year*1e+6) > 0.5 ) and ( self.r[ir] < dju ):
        #                 self.sigmaplts_resIII[itime,ir] = entry - self.sigmaplts[71,ir]
        #             else:
        #                 self.sigmaplts_resIII[itime,ir] = 0.

        # bla = np.zeros((nr))
        # for itime in range(1,ntime):
        #     for ir in range(nr):
        #         self.dmplts_resIII[itime,ir] = (self.sigmaplts_resIII[itime,ir] - self.sigmaplts_resIII[itime-1,ir])*math.pi*((self.r[ir] * self.r[ir]) - (self.r[ir-1] * self.r[ir-1])) # are -1 notation right, or should it be +1 first and no -1 then?
        #         bla[ir] = self.dmplts_resIII[itime,ir]/(self.time[itime]-self.time[itime-1])
        #     for ir in range(1,nr):
        #         self.dmdt_resIII[itime,ir] =  bla[ir]


def readdata():
    """
    Reading routine for the disk model data. Just do a=readdata() and you will get
    the object containing all the data.
    """
    dum = disk()
    dum.read()
    return dum
