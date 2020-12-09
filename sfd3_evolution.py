# Script from: https://github.com/timlichtenberg/2stage_scripts_data
# Part of the combined repository: https://osf.io/e2kfv/
from sfd_functions import *

## Dataframe labels for evolutionary results
evolution_labels = [ "res", "imf", "time", "sol_frac", "liq_frac", "hydrous_frac", "primitive_frac", "n2co2_frac", "cocl_frac", "h2o_frac", "perco_frac", "melt1_frac", "melt2_frac", "Tmax", "Tmean", "m_plts_tot"]

if os.path.isfile(dat_dir+"sfd_evolution.csv"):

    df_evolution = pd.read_csv(dat_dir+"sfd_evolution.csv")

    # Remove unnecessary index columns
    if "Unnamed: 0" in df_evolution.columns:
        print (datetime.datetime.now(), "Remove unnecessary 'Unnamed: 0' index columns")
        df_evolution = df_evolution.drop(["Unnamed: 0"], axis=1)
        df_evolution.to_csv(dat_dir+"sfd_evolution.csv", index=None)
    if "Unnamed: 0.1" in df_evolution.columns:
        print (datetime.datetime.now(), "Remove unnecessary 'Unnamed: 0.1' index columns")
        df_evolution = df_evolution.drop(["Unnamed: 0.1"], axis=1)
        df_evolution.to_csv(dat_dir+"sfd_evolution.csv", index=None)

    print (datetime.datetime.now(), "Import SFD evolution results: sfd_evolution.csv")

else:

    print (datetime.datetime.now(), "Generate SFD evolution results")

    df_evolution = pd.DataFrame(columns=evolution_labels)

# Read in data from former steps
print (datetime.datetime.now(), "Read in data from former steps")
df  = pd.read_csv(dat_dir+"plts_all_data.csv")
df_sfd = pd.read_csv(dat_dir+"sfd_dMdt_all.csv")
df_interpolated = pd.read_csv(dat_dir+"sfd_interpolated.csv")

df_sfd = df_sfd.round({'time': 6, 'tform': 6})
df_interpolated = df_interpolated.round({'time': 6, 'tform': 6})

reservoir_list = [ "resI", "resII" ] # , "resIII", "resIV", "resV", "resVI"

# Define which reservoir we look at
for res in reservoir_list:

    for imf in [ "powerlaw" ]: # , "gaussian"

        # Check if res and imf combination already exists in written data frame
        if df_evolution.loc[(df_evolution['res'] == res) & (df_evolution['imf'] == imf)].empty:

            # Load specific sub data frame
            df_plts = df_sfd.loc[(df_sfd['imf'] == imf) & (df_sfd['res'] == res)]
            running_time = sorted(set(df_plts[df_plts['imf']==imf].tform.tolist()))

            print (datetime.datetime.now(), res, imf, "Generate SFD evolution results:")

            # Quantity lists to build dataframe from
            res_list            = []
            imf_list            = []
            time_list           = []#sorted(running_time)#[0::1]
            sol_frac_list       = []
            liq_frac_list       = []
            hydrous_frac_list   = []
            primitive_frac_list = []
            n2co2_frac_list     = []
            cocl_frac_list      = []
            h2o_frac_list       = []
            perco_frac_list     = []
            melt1_frac_list     = []
            melt2_frac_list     = []
            Tmax_grid_list      = []
            Tmean_markers_list  = []
            mass_tot_time_list  = []

            # Loop through all times
            for tidx, time in enumerate(running_time):

                print (datetime.datetime.now(), "..", res, imf, round(time,6),)

                tform_max = np.min([5.0, time])

                tform_list_relevant = sorted(tform for tform in running_time if tform <= tform_max)

                # Total mass over all tforms for current timestep, start
                mass_tot_time = 0.

                # Quantities for current timestep
                sol_frac_time       = np.zeros(np.size(tform_list_relevant))
                liq_frac_time       = np.zeros(np.size(tform_list_relevant))
                hydrous_frac_time   = np.zeros(np.size(tform_list_relevant))
                primitive_frac_time = np.zeros(np.size(tform_list_relevant))
                n2co2_frac_time     = np.zeros(np.size(tform_list_relevant))
                cocl_frac_time      = np.zeros(np.size(tform_list_relevant))
                h2o_frac_time       = np.zeros(np.size(tform_list_relevant))
                perco_frac_time     = np.zeros(np.size(tform_list_relevant))
                melt1_frac_time     = np.zeros(np.size(tform_list_relevant))
                melt2_frac_time     = np.zeros(np.size(tform_list_relevant))
                Tmax_grid_time      = np.zeros(np.size(tform_list_relevant))
                Tmean_markers_time  = np.zeros(np.size(tform_list_relevant))

                # Loop through all formation times before that timestep
                for tfidx, tform in enumerate(tform_list_relevant):

                    mass_bins_tform = df_plts.loc[(df_plts['imf'] == imf) & (df_plts['tform'] == tform)]
                    mass_bins_tform = mass_bins_tform.drop(["tform", "res", "imf", "overflow"], axis=1) # , "Unnamed: 0"

                    # Total mass over all rad bins for current tform
                    mass_tot_tform = np.sum(mass_bins_tform.values)

                    # List of radius bin columns
                    rad_list    = [ float(k) for k in mass_bins_tform.columns.tolist() ]

                    # Only do sth if there are any planetesimals
                    if mass_tot_tform > 0.:
                       
                        # Set to zero initially, to add weighted quantities later on
                        sol_frac_tform       = np.zeros(np.size(rad_list))
                        liq_frac_tform       = np.zeros(np.size(rad_list))
                        hydrous_frac_tform   = np.zeros(np.size(rad_list))
                        primitive_frac_tform = np.zeros(np.size(rad_list))
                        n2co2_frac_tform     = np.zeros(np.size(rad_list))
                        cocl_frac_tform      = np.zeros(np.size(rad_list))
                        h2o_frac_tform       = np.zeros(np.size(rad_list))
                        perco_frac_tform     = np.zeros(np.size(rad_list))
                        melt1_frac_tform     = np.zeros(np.size(rad_list))
                        melt2_frac_tform     = np.zeros(np.size(rad_list))
                        Tmax_grid_tform      = np.zeros(np.size(rad_list))
                        Tmean_markers_tform  = np.zeros(np.size(rad_list))

                        df_interpolated_current = df_interpolated.loc[(df_interpolated['res'] == res) & (df_interpolated['imf'] == imf) & (df_interpolated['time'] == time) & (df_interpolated['tform'] == tform)]
                       
                        if mass_bins_tform.empty or df_interpolated_current.empty:#2.86606712805
                            print (tidx, time, tform)
                            print ("Formation times, plts:", sorted(set(df_plts.loc[(df_plts['imf'] == imf)].tform.tolist())))
                            print ("Times, plts:", sorted(set(df_interpolated.loc[(df_interpolated['imf'] == imf)].time.tolist())))
                            print ("Formation times, interpolated:", sorted(set(df_interpolated.loc[(df_interpolated['imf'] == imf)].tform.tolist())))
                            print ("Times, interpolated:", sorted(set(df_interpolated.loc[(df_interpolated['imf'] == imf) & (df_interpolated['time'] == float(time))].tform.tolist())))
                            # print(df_interpolated)

                        # For each rad bin in current tform
                        for ridx, rad_bin in enumerate(rad_list):

                            # print (tidx, time, tfidx, tform, ridx, rad_bin)

                            # Mass/weight of current radius bin
                            mass_bin_tform_radbin = mass_bins_tform[str(rad_bin)].tolist()[0]

                            # Add current mass of radbin-tform combi to total mass over all tforms
                            mass_tot_time   += mass_bin_tform_radbin        

                            # Only add stuff if there's something to add
                            if mass_bin_tform_radbin > 0.:

                                sol_frac_tform_radbin       = df_interpolated_current.loc[(df_interpolated_current['rad_bin'] == rad_bin)].sol_frac_interpolated.tolist()[0]
                                liq_frac_tform_radbin       = df_interpolated_current.loc[(df_interpolated_current['rad_bin'] == rad_bin)].liq_frac_interpolated.tolist()[0]
                                hydrous_frac_tform_radbin   = df_interpolated_current.loc[(df_interpolated_current['rad_bin'] == rad_bin)].hydrous_frac_interpolated.tolist()[0]
                                primitive_frac_tform_radbin = df_interpolated_current.loc[(df_interpolated_current['rad_bin'] == rad_bin)].primitive_frac_interpolated.tolist()[0]
                                n2co2_frac_tform_radbin     = df_interpolated_current.loc[(df_interpolated_current['rad_bin'] == rad_bin)].n2co2_frac_interpolated.tolist()[0]
                                cocl_frac_tform_radbin      = df_interpolated_current.loc[(df_interpolated_current['rad_bin'] == rad_bin)].cocl_frac_interpolated.tolist()[0]
                                h2o_frac_tform_radbin       = df_interpolated_current.loc[(df_interpolated_current['rad_bin'] == rad_bin)].h2o_frac_interpolated.tolist()[0]
                                perco_frac_tform_radbin     = df_interpolated_current.loc[(df_interpolated_current['rad_bin'] == rad_bin)].perco_frac_interpolated.tolist()[0]
                                melt1_frac_tform_radbin     = df_interpolated_current.loc[(df_interpolated_current['rad_bin'] == rad_bin)].melt1_frac_interpolated.tolist()[0]
                                melt2_frac_tform_radbin     = df_interpolated_current.loc[(df_interpolated_current['rad_bin'] == rad_bin)].melt2_frac_interpolated.tolist()[0]
                                Tmax_grid_tform_radbin      = df_interpolated_current.loc[(df_interpolated_current['rad_bin'] == rad_bin)].Tmax_grid_interpolated.tolist()[0]
                                Tmean_markers_tform_radbin  = df_interpolated_current.loc[(df_interpolated_current['rad_bin'] == rad_bin)].Tmean_markers_interpolated.tolist()[0]

                                # Add weighted frac to current tform line
                                sol_frac_tform[ridx]       = mass_bin_tform_radbin * sol_frac_tform_radbin
                                liq_frac_tform[ridx]       = mass_bin_tform_radbin * liq_frac_tform_radbin
                                hydrous_frac_tform[ridx]   = mass_bin_tform_radbin * hydrous_frac_tform_radbin
                                primitive_frac_tform[ridx] = mass_bin_tform_radbin * primitive_frac_tform_radbin
                                n2co2_frac_tform[ridx]     = mass_bin_tform_radbin * n2co2_frac_tform_radbin
                                cocl_frac_tform[ridx]      = mass_bin_tform_radbin * cocl_frac_tform_radbin
                                h2o_frac_tform[ridx]       = mass_bin_tform_radbin * h2o_frac_tform_radbin
                                perco_frac_tform[ridx]     = mass_bin_tform_radbin * perco_frac_tform_radbin
                                melt1_frac_tform[ridx]     = mass_bin_tform_radbin * melt1_frac_tform_radbin
                                melt2_frac_tform[ridx]     = mass_bin_tform_radbin * melt2_frac_tform_radbin
                                Tmax_grid_tform[ridx]      = mass_bin_tform_radbin * Tmax_grid_tform_radbin
                                Tmean_markers_tform[ridx]  = mass_bin_tform_radbin * Tmean_markers_tform_radbin
                        
                    # Add current tform line to total frac
                    if mass_tot_tform > 0.:
                        sol_frac_time[tfidx]       = np.sum(sol_frac_tform)
                        liq_frac_time[tfidx]       = np.sum(liq_frac_tform)
                        hydrous_frac_time[tfidx]   = np.sum(hydrous_frac_tform)
                        primitive_frac_time[tfidx] = np.sum(primitive_frac_tform)
                        n2co2_frac_time[tfidx]     = np.sum(n2co2_frac_tform)
                        cocl_frac_time[tfidx]      = np.sum(cocl_frac_tform)
                        h2o_frac_time[tfidx]       = np.sum(h2o_frac_tform)
                        perco_frac_time[tfidx]     = np.sum(perco_frac_tform)
                        melt1_frac_time[tfidx]     = np.sum(melt1_frac_tform)
                        melt2_frac_time[tfidx]     = np.sum(melt2_frac_tform)
                        Tmax_grid_time[tfidx]      = np.sum(Tmax_grid_tform)
                        Tmean_markers_time[tfidx]  = np.sum(Tmean_markers_tform)

                # Normalize by the total mass budget at the current timestep
                if mass_tot_time > 0.:
                    
                    sol_frac_time_weighted       = np.sum(sol_frac_time)/mass_tot_time
                    liq_frac_time_weighted       = np.sum(liq_frac_time)/mass_tot_time
                    hydrous_frac_time_weighted   = np.sum(hydrous_frac_time)/mass_tot_time
                    primitive_frac_time_weighted = np.sum(primitive_frac_time)/mass_tot_time
                    n2co2_frac_time_weighted     = np.sum(n2co2_frac_time)/mass_tot_time
                    cocl_frac_time_weighted      = np.sum(cocl_frac_time)/mass_tot_time
                    h2o_frac_time_weighted       = np.sum(h2o_frac_time)/mass_tot_time
                    perco_frac_time_weighted     = np.sum(perco_frac_time)/mass_tot_time
                    melt1_frac_time_weighted     = np.sum(melt1_frac_time)/mass_tot_time
                    melt2_frac_time_weighted     = np.sum(melt2_frac_time)/mass_tot_time
                    Tmax_grid_time_weighted      = np.sum(Tmax_grid_time)/mass_tot_time
                    Tmean_markers_time_weighted  = np.sum(Tmean_markers_time)/mass_tot_time

                    # Add weighted quantities to list/array
                    res_list.append(res)
                    imf_list.append(imf)
                    time_list.append(time)
                    sol_frac_list.append(round(sol_frac_time_weighted,6))
                    liq_frac_list.append(round(liq_frac_time_weighted,6))
                    hydrous_frac_list.append(round(hydrous_frac_time_weighted,6))
                    primitive_frac_list.append(round(primitive_frac_time_weighted,6))
                    n2co2_frac_list.append(round(n2co2_frac_time_weighted,6))
                    cocl_frac_list.append(round(cocl_frac_time_weighted,6))
                    h2o_frac_list.append(round(h2o_frac_time_weighted,6))
                    perco_frac_list.append(round(perco_frac_time_weighted,6))
                    melt1_frac_list.append(round(melt1_frac_time_weighted,6))
                    melt2_frac_list.append(round(melt2_frac_time_weighted,6))
                    Tmax_grid_list.append(round(Tmax_grid_time_weighted,6))
                    Tmean_markers_list.append(round(Tmean_markers_time_weighted,6))
                    mass_tot_time_list.append(round(mass_tot_time/M_earth,6))

                    print (round(mass_tot_time/M_earth,6), round(h2o_frac_time_weighted,6), round(Tmean_markers_time_weighted,6))

                else:
                    print (round(mass_tot_time/M_earth,6), "no planetesimals")

            df_evolution_tmp   = pd.DataFrame( list(zip(res_list,imf_list,time_list,sol_frac_list, liq_frac_list, hydrous_frac_list, primitive_frac_list, n2co2_frac_list, cocl_frac_list, h2o_frac_list, perco_frac_list, melt1_frac_list, melt2_frac_list, Tmax_grid_list, Tmean_markers_list, mass_tot_time_list)), columns=evolution_labels)
            df_evolution    = df_evolution.append(df_evolution_tmp, sort=False, ignore_index=True, verify_integrity=True)
            df_evolution.to_csv(dat_dir+"sfd_evolution.csv", index=None)

        # If res and imf combination already exists in written data frame
        else:
            print (datetime.datetime.now(), res, imf, "combination already exists in written data frame")
