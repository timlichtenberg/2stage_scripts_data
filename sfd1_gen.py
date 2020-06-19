from sfd_functions import *
from jd_natconst import *
from jd_readdata import *
from jd_plot import *
from matplotlib.ticker import LogLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import subprocess

q = readdata()

# Check if data was already read in
if os.path.isfile(dat_dir+"plts_all_data.csv"):
    df = pd.read_csv(dat_dir+"plts_all_data.csv")

else:

    # Select files
    FILES       = [ "r???t???al05250fe01150tmp150.dat" ]

    os.chdir(dat_dir)
    FILE_LIST   = []
    for i in range(0,len(FILES)):
        glob_runs=glob.glob(FILES[i])
        for item in glob_runs:
            FILE_LIST.append(item)

    print (FILE_LIST)

    df = pd.DataFrame(index=["run","rad","tform","time","sol_frac","liq_frac","hydrous_frac","primitive_frac","n2co2_frac","cocl_frac","h2o_frac","phyllo1_frac","phyllo2_frac","phyllo3_frac","phyllo4_frac","perco_frac","melt1_frac","melt2_frac","maxtk","t_max_body","meantk","t_mean_body","count_toohot"])

    for FILE in FILE_LIST:

        # Get the settings of the current file
        RUN       = str(FILE[:-4])
        r_init    = float(RUN[1:4])          # km
        t_form    = float(RUN[5:8])*1e-2     # Myr
        al_init   = float(RUN[10:15])*1e-8   # 26Al/27Al # to normalize: /5.25e-5
        file_name = dat_dir+FILE

        print (file_name)

        df2 = pd.read_csv(file_name, sep=" ", names=["time","sol_frac","liq_frac","hydrous_frac","primitive_frac","n2co2_frac","cocl_frac","h2o_frac","phyllo1_frac","phyllo2_frac","phyllo3_frac","phyllo4_frac","perco_frac","melt1_frac","melt2_frac","maxtk","t_max_body","meantk","t_mean_body","count_toohot"])
        df2.insert(0, "run", RUN)
        df2.insert(1, "rad", r_init)
        df2.insert(2, "tform", t_form)

        df = df.append(df2, sort=False, ignore_index=True)
        df = df.dropna()

    # Save to csv
    df.to_csv(dat_dir+"plts_all_data.csv")

# Binning setting
bin_list_powerlaw   = np.linspace(rad_min, rad_max, n_bins)
bin_list_gaussian   = n_bins


# Calculate radii for each bin
bin_centers = [ round(k,2) for k in bin_list_powerlaw[:-1]+np.diff(bin_list_powerlaw)/2.]
bin_centers_powerlaw = [ round(k,2) for k in bin_list_powerlaw[:-1]+np.diff(bin_list_powerlaw)/2.]

# DD18 time series
time_series = [ round(k,6) for k in q.time/(1e+6*year)] # [Myr]
res_list = [ "resI", "resII" ] 

if os.path.isfile(csv_dir+"sfd_dMdt_powerlaw.csv"):
    df_plts_powerlaw       = pd.read_csv(csv_dir+"sfd_dMdt_powerlaw.csv")
    df_plts_gaussian       = pd.read_csv(csv_dir+"sfd_dMdt_gaussian.csv")
    print (datetime.datetime.now(), "Import planetesimal SFD families")

else:

    print (datetime.datetime.now(), "Generate/write planetesimal SFD families")

    for res in res_list:

        if os.path.isfile(csv_dir+"sfd_dMdt_"+res+"_powerlaw.csv") and os.path.isfile(csv_dir+"sfd_dMdt_"+res+"_gaussian.csv"):

            df_plts_powerlaw       = pd.read_csv(csv_dir+"sfd_dMdt_"+res+"_powerlaw.csv")
            df_plts_gaussian       = pd.read_csv(csv_dir+"sfd_dMdt_"+res+"_gaussian.csv")

        else:

            labels = ["tform", "res", "imf"]+list(bin_centers)+["overflow"]
            df_plts_powerlaw  = pd.DataFrame(columns=labels)#index=labels)
            df_plts_gaussian  = pd.DataFrame(columns=labels)#index=labels)

            # Variable to incorporate overflow mass next timestep!! 0 at beginning
            overflow_powerlaw = 0.
            overflow_gaussian = 0.

            # Count total mass for output
            mass_tot_generated_plts     = 0.
            mass_tot_generated_plts_pot = 0.

            # For approximating dMdt's that take too long
            if res == "resII":
                if os.path.isfile(csv_dir+"sfd_mass_resI_source_powerlaw.csv") and os.path.isfile(csv_dir+"sfd_mass_resI_source_gaussian.csv"):

                    powerlaw_mass_source = np.genfromtxt(csv_dir+"sfd_mass_resI_source_powerlaw.csv", delimiter=",")
                    gaussian_mass_source = np.genfromtxt(csv_dir+"sfd_mass_resI_source_gaussian.csv", delimiter=",")

                    powerlaw_labels = np.genfromtxt(csv_dir+"sfd_mass_resI_source_powerlaw_labels.csv", delimiter=",")
                    gaussian_labels = np.genfromtxt(csv_dir+"sfd_mass_resI_source_gaussian_labels.csv", delimiter=",")

                else:
                    df_plts_powerlaw_resI   = pd.read_csv(csv_dir+"sfd_dMdt_resI_powerlaw.csv")
                    df_plts_gaussian_resI   = pd.read_csv(csv_dir+"sfd_dMdt_resI_gaussian.csv")

                    df_plts_powerlaw_resI_2 = df_plts_powerlaw_resI.drop(["tform", "res", "imf", "overflow"], axis=1)
                    powerlaw_labels         = [ float(k) for k in df_plts_powerlaw_resI_2.columns.tolist() ]
                    df_plts_powerlaw_resI_2 = df_plts_powerlaw_resI_2.sum(axis=0, skipna=True)
                    powerlaw_mass_source    = df_plts_powerlaw_resI_2.tolist()

                    df_plts_gaussian_resI_2 = df_plts_gaussian_resI.drop(["tform", "res", "imf", "overflow"], axis=1)
                    gaussian_labels         = [ float(k) for k in df_plts_gaussian_resI_2.columns.tolist() ]
                    df_plts_gaussian_resI_2 = df_plts_gaussian_resI_2.sum(axis=0, skipna=True)
                    gaussian_mass_source    = df_plts_gaussian_resI_2.tolist()

                    
                    np.savetxt(csv_dir+"sfd_mass_resI_source_powerlaw.csv", powerlaw_mass_source, delimiter=",")
                    np.savetxt(csv_dir+"sfd_mass_resI_source_gaussian.csv", gaussian_mass_source, delimiter=",")
                    np.savetxt(csv_dir+"sfd_mass_resI_source_powerlaw_labels.csv", powerlaw_labels, delimiter=",")
                    np.savetxt(csv_dir+"sfd_mass_resI_source_gaussian_labels.csv", gaussian_labels, delimiter=",")

                powerlaw_mass_source_sum = np.sum(powerlaw_mass_source)
                gaussian_mass_source_sum = np.sum(gaussian_mass_source)
                powerlaw_mass_min = mass_func(powerlaw_labels)
                gaussian_mass_min = mass_func(gaussian_labels)

            for idx, time in enumerate(time_series):

                if res == "resI":
                    mplts_resI      = q.mplts_resI/1000.                    # kg
                    dmplts_resI     = np.sum(q.dmplts_resI,axis=1)/1000.    # kg
                    dMdt            = dmplts_resI[idx]
                    dMdt_pot        = mplts_resI[idx]
                if res == "resII":
                    dmplts_resII    = np.sum(q.dmplts_resII,axis=1)/1000.   # kg
                    mplts_resII     = q.mplts_resII/1000.                   # kg
                    dMdt            = dmplts_resII[idx]
                    dMdt_pot        = mplts_resII[idx]

                rad_family_powerlaw     = [ 0.0 ]
                mass_family_powerlaw    = [ 0.0 ]
                rad_family_gaussian     = [ 0.0 ]
                mass_family_gaussian    = [ 0.0 ]

                # Disk assumed to vanish at 5 Myr
                if time >= 5.0:
                    dMdt = 0.0

                print("-")
                print (idx, datetime.datetime.now(), "->", res, ":", round(time,6), "Myr | dM/dt:", round(dMdt/M_earth,4),)

                # Prevent error in case of 0:
                if dMdt > mass_min:

                    # Check if all planetesimal families were already generated before
                    if res == "resI" and os.path.isfile(csv_dir+"mass_families/"+res+"_"+str(time)+"_overflow.csv") and os.path.isfile(csv_dir+"mass_families/"+res+"_"+str(time)+"_rad_family_powerlaw.csv") and os.path.isfile(csv_dir+"mass_families/"+res+"_"+str(time)+"_mass_family_powerlaw.csv") and os.path.isfile(csv_dir+"mass_families/"+res+"_"+str(time)+"_rad_family_gaussian.csv") and os.path.isfile(csv_dir+"mass_families/"+res+"_"+str(time)+"_mass_family_gaussian.csv"):

                        print ("| LOAD |",)
                        overflow = np.genfromtxt(csv_dir+"mass_families/"+res+"_"+str(time)+"_overflow.csv", delimiter=",")
                        overflow_powerlaw = overflow[0]
                        overflow_gaussian = overflow[1]

                        if overflow_powerlaw < dMdt:
                            rad_family_powerlaw     = np.genfromtxt(csv_dir+"mass_families/"+res+"_"+str(time)+"_rad_family_powerlaw.csv", delimiter=",")
                            mass_family_powerlaw    = np.genfromtxt(csv_dir+"mass_families/"+res+"_"+str(time)+"_mass_family_powerlaw.csv", delimiter=",")
                        else:
                            rad_family_powerlaw     = [ 0.0 ]
                            mass_family_powerlaw    = [ 0.0 ]

                        if overflow_gaussian < dMdt:
                            rad_family_gaussian     = np.genfromtxt(csv_dir+"mass_families/"+res+"_"+str(time)+"_rad_family_gaussian.csv", delimiter=",")
                            mass_family_gaussian    = np.genfromtxt(csv_dir+"mass_families/"+res+"_"+str(time)+"_mass_family_gaussian.csv", delimiter=",")
                        else:
                            rad_family_gaussian     = [ 0.0 ]
                            mass_family_gaussian    = [ 0.0 ]

                    # If not, generate them
                    else:

                        # But only for resI
                        if res == "resI":

                            print ("| CALC |",)
                            rad_family_powerlaw, mass_family_powerlaw, overflow_powerlaw, rad_family_gaussian, mass_family_gaussian, overflow_gaussian = generate_planetesimal_family(random_reals_mass, imf_slope, mass_min, mass_max, mass_mean, dMdt, overflow_powerlaw, overflow_gaussian)
                            np.savetxt(csv_dir+"mass_families/"+res+"_"+str(time)+"_overflow.csv", [overflow_powerlaw, overflow_gaussian], delimiter=",")

                            np.savetxt(csv_dir+"mass_families/"+res+"_"+str(time)+"_rad_family_powerlaw.csv", rad_family_powerlaw, delimiter=",")
                            np.savetxt(csv_dir+"mass_families/"+res+"_"+str(time)+"_mass_family_powerlaw.csv", mass_family_powerlaw, delimiter=",")
                            np.savetxt(csv_dir+"mass_families/"+res+"_"+str(time)+"_rad_family_gaussian.csv", rad_family_gaussian, delimiter=",")
                            np.savetxt(csv_dir+"mass_families/"+res+"_"+str(time)+"_mass_family_gaussian.csv", mass_family_gaussian, delimiter=",")
                        else:

                            print ("| SCALE |",)
                            # Scale source approx. distribution to current dM
                            dM_ratio_powerlaw   = dMdt / np.sum(powerlaw_mass_source_sum)
                            powerlaw_mass       = np.zeros(np.size(powerlaw_mass_source))
                            for idx, m in enumerate(powerlaw_mass_source):
                                m_plts = float(m) * dM_ratio_powerlaw
                                if m_plts < powerlaw_mass_min[idx]:
                                    powerlaw_mass[idx] = 0.
                                else:
                                    powerlaw_mass[idx] = m_plts
                            overflow_powerlaw   = dMdt - np.sum(powerlaw_mass)

                            dM_ratio_gaussian   = dMdt / np.sum(gaussian_mass_source_sum)
                            gaussian_mass       = np.zeros(np.size(gaussian_mass_source))
                            for idx, m in enumerate(gaussian_mass_source):
                                m_plts = float(m) * dM_ratio_gaussian
                                if m_plts < gaussian_mass_min[idx]:
                                    gaussian_mass[idx] = 0.
                                else:
                                    gaussian_mass[idx] = m_plts
                            overflow_gaussian   = dMdt - np.sum(gaussian_mass)

                else:
                    powerlaw_mass = np.zeros(np.size(bin_list_powerlaw)-1)
                    gaussian_mass = np.zeros(np.size(bin_list_powerlaw)-1)
                    overflow_gaussian       = dMdt
                    overflow_powerlaw       = dMdt
                    
                mass_tot_generated_plts       += dMdt
                mass_tot_generated_plts_pot   = dMdt_pot # as defined by SI condition

                if res == "resI":
                    if (np.size(rad_family_powerlaw) > 0) and (np.size(mass_family_powerlaw) > 0):
                        powerlaw_mass, powerlaw_edges, powerlaw_binnumber = stats.binned_statistic(rad_family_powerlaw, mass_family_powerlaw, statistic='sum', bins=bin_list_powerlaw)
                    if (np.size(rad_family_gaussian) > 0) and (np.size(mass_family_gaussian) > 0):
                        gaussian_mass, gaussian_edges, gaussian_binnumber = stats.binned_statistic(rad_family_gaussian, mass_family_gaussian, statistic='sum', bins=bin_list_powerlaw) # bin_list_gaussian

                if dMdt > mass_min:
                    print ("M_tot:", round(mass_tot_generated_plts/M_earth,4), round(mass_tot_generated_plts_pot/M_earth,4), "| dM families:", round(np.sum(powerlaw_mass)/M_earth,4), round(np.sum(gaussian_mass)/M_earth,4), "| overflow:", round(overflow_powerlaw/M_ceres,4), round(overflow_gaussian/M_ceres,4))
                else:
                    print ("M_tot:", round(mass_tot_generated_plts/M_earth,4), round(mass_tot_generated_plts_pot/M_earth,4))

                entries_powerlaw = [list([time, res, "powerlaw"]+list(powerlaw_mass)+[overflow_powerlaw])]
                entries_gaussian = [list([time, res, "gaussian"]+list(gaussian_mass)+[overflow_gaussian])]
                labels = ["tform", "res", "imf"]+list(bin_centers)+["overflow"]

                df_plts_powerlaw2 = pd.DataFrame.from_records(entries_powerlaw, columns=labels)
                df_plts_gaussian2 = pd.DataFrame.from_records(entries_gaussian, columns=labels)

                df_plts_powerlaw = df_plts_powerlaw.append(df_plts_powerlaw2, sort=False, ignore_index=True)
                df_plts_gaussian = df_plts_gaussian.append(df_plts_gaussian2, sort=False, ignore_index=True)

            df_plts_powerlaw = df_plts_powerlaw.dropna()
            df_plts_gaussian = df_plts_gaussian.dropna()
            df_plts_powerlaw.to_csv(csv_dir+"sfd_dMdt_"+res+"_powerlaw.csv", index=None)
            df_plts_gaussian.to_csv(csv_dir+"sfd_dMdt_"+res+"_gaussian.csv", index=None)

    # Generate combined dataframes
    df_plts_powerlaw_resI   = pd.read_csv(csv_dir+"sfd_dMdt_resI_powerlaw.csv")
    df_plts_gaussian_resI   = pd.read_csv(csv_dir+"sfd_dMdt_resI_gaussian.csv")
    df_plts_powerlaw_resII  = pd.read_csv(csv_dir+"sfd_dMdt_resII_powerlaw.csv")
    df_plts_gaussian_resII  = pd.read_csv(csv_dir+"sfd_dMdt_resII_gaussian.csv")

    df_plts_powerlaw = pd.concat([df_plts_powerlaw_resI, df_plts_powerlaw_resII], axis=0, sort=False, ignore_index=True)
    df_plts_gaussian = pd.concat([df_plts_gaussian_resI, df_plts_gaussian_resII], axis=0, sort=False, ignore_index=True)
    df_plts_all     = pd.concat([df_plts_powerlaw, df_plts_gaussian], axis=0, sort=False, ignore_index=True)

    df_plts_powerlaw = df_plts_powerlaw.round({'time': 6, 'tform': 6})
    df_plts_gaussian = df_plts_gaussian.round({'time': 6, 'tform': 6})
    df_plts_all = df_plts_all.round({'time': 6, 'tform': 6})

    # Save to disk
    df_plts_powerlaw.to_csv(csv_dir+"sfd_dMdt_powerlaw.csv", index=None)
    df_plts_gaussian.to_csv(csv_dir+"sfd_dMdt_gaussian.csv", index=None)
    df_plts_all.to_csv(dat_dir+"sfd_dMdt_all.csv", index=None)

    print (df_plts_all)