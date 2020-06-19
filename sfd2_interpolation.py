from sfd_functions import *

df = pd.read_csv(dat_dir+"plts_all_data.csv")

reservoir_list = [ "resI", "resII" ] # , "resIII", "resIV", "resV", "resVI"

if (os.path.isfile(dat_dir+"sfd_grid_3d_all_data_points.csv")):# or (recalc == 0):

    print (datetime.datetime.now(), "Import data grid points and values")

    with open(dat_dir+"sfd_grid_3d_all_data_points.csv", "r") as csvfile:
        grid_points = []
        for row in csv.reader(csvfile):
            grid_points.append(row)

    with open(dat_dir+"sfd_grid_3d_all_data_values.csv", "r") as csvfile:
        grid_values = []
        for row in csv.reader(csvfile):
            grid_values.append(row)

        sol_frac_data        = np.asarray( [ float(k) for k in list(grid_values[0]) ] )
        liq_frac_data        = np.asarray( [ float(k) for k in list(grid_values[1]) ] )
        hydrous_frac_data    = np.asarray( [ float(k) for k in list(grid_values[2]) ] )
        primitive_frac_data  = np.asarray( [ float(k) for k in list(grid_values[3]) ] )
        n2co2_frac_data      = np.asarray( [ float(k) for k in list(grid_values[4]) ] )
        cocl_frac_data       = np.asarray( [ float(k) for k in list(grid_values[5]) ] )
        h2o_frac_data        = np.asarray( [ float(k) for k in list(grid_values[6]) ] )
        perco_frac_data      = np.asarray( [ float(k) for k in list(grid_values[7]) ] )
        melt1_frac_data      = np.asarray( [ float(k) for k in list(grid_values[8]) ] )
        melt2_frac_data      = np.asarray( [ float(k) for k in list(grid_values[9]) ] )
        Tmax_grid_data       = np.asarray( [ float(k) for k in list(grid_values[10]) ] )
        Tmean_markers_data   = np.asarray( [ float(k) for k in list(grid_values[11]) ] )

else:

    print (datetime.datetime.now(), "Generate/write data grid points and values")
    
    run_list    = natsorted(set(df.run.tolist()))
    rad_list    = natsorted(set(df.rad.tolist()))
    tform_list  = natsorted(set(set(df.tform.tolist())))

    grid_points          = list([])
    sol_frac_data        = list([])
    liq_frac_data        = list([])
    hydrous_frac_data    = list([])
    primitive_frac_data  = list([])
    n2co2_frac_data      = list([])
    cocl_frac_data       = list([])
    h2o_frac_data        = list([])
    perco_frac_data      = list([])
    melt1_frac_data      = list([])
    melt2_frac_data      = list([])
    Tmax_grid_data       = list([])
    Tmean_markers_data   = list([])

    for rad in rad_list:

        for tform in tform_list:

            print ("Read data for:", rad, tform)

            time_list = [ round(float(k),6) for k in df.loc[(df['tform'] == tform) & (df['rad'] == rad)].time.tolist() ]

            for time in time_list:
                
                grid_points.append([rad, round(float(tform),6), round(float(time),6)])

            # https://www.geeksforgeeks.org/append-extend-python/
            sol_frac_data.extend(df.loc[(df['tform'] == tform) & (df['rad'] == rad)].sol_frac.tolist())
            liq_frac_data.extend(df.loc[(df['tform'] == tform) & (df['rad'] == rad)].liq_frac.tolist())
            hydrous_frac_data.extend(df.loc[(df['tform'] == tform) & (df['rad'] == rad)].hydrous_frac.tolist())
            primitive_frac_data.extend(df.loc[(df['tform'] == tform) & (df['rad'] == rad)].primitive_frac.tolist())
            n2co2_frac_data.extend(df.loc[(df['tform'] == tform) & (df['rad'] == rad)].n2co2_frac.tolist())
            cocl_frac_data.extend(df.loc[(df['tform'] == tform) & (df['rad'] == rad)].cocl_frac.tolist())
            h2o_frac_data.extend(df.loc[(df['tform'] == tform) & (df['rad'] == rad)].h2o_frac.tolist())
            perco_frac_data.extend(df.loc[(df['tform'] == tform) & (df['rad'] == rad)].perco_frac.tolist())
            melt1_frac_data.extend(df.loc[(df['tform'] == tform) & (df['rad'] == rad)].melt1_frac.tolist())
            melt2_frac_data.extend(df.loc[(df['tform'] == tform) & (df['rad'] == rad)].melt2_frac.tolist())
            Tmax_grid_data.extend(df.loc[(df['tform'] == tform) & (df['rad'] == rad)].maxtk.tolist())
            Tmean_markers_data.extend(df.loc[(df['tform'] == tform) & (df['rad'] == rad)].t_mean_body.tolist())

    grid_points              = np.asarray(grid_points)
    sol_frac_data            = np.asarray(sol_frac_data)
    liq_frac_data            = np.asarray(liq_frac_data)
    hydrous_frac_data        = np.asarray(hydrous_frac_data)
    primitive_frac_data      = np.asarray(primitive_frac_data)
    n2co2_frac_data          = np.asarray(n2co2_frac_data)
    cocl_frac_data           = np.asarray(cocl_frac_data)
    h2o_frac_data            = np.asarray(h2o_frac_data)
    perco_frac_data          = np.asarray(perco_frac_data)
    melt1_frac_data          = np.asarray(melt1_frac_data)
    melt2_frac_data          = np.asarray(melt2_frac_data)
    Tmax_grid_data           = np.asarray(Tmax_grid_data)
    Tmean_markers_data       = np.asarray(Tmean_markers_data)

    grid_values = [
                                sol_frac_data,
                                liq_frac_data,
                                hydrous_frac_data,
                                primitive_frac_data,
                                n2co2_frac_data,
                                cocl_frac_data,
                                h2o_frac_data,
                                perco_frac_data,
                                melt1_frac_data,
                                melt2_frac_data,
                                Tmax_grid_data,
                                Tmean_markers_data
                            ]

    with open(dat_dir+"sfd_grid_3d_all_data_points.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(grid_points)

    with open(dat_dir+"sfd_grid_3d_all_data_values.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(grid_values)

print (datetime.datetime.now(), "Start SFD interpolation")

## Interpolation values to dataframe
interpolated_labels = ["res","rad_bin","tform","time","imf","sol_frac_interpolated", "liq_frac_interpolated", "hydrous_frac_interpolated", "primitive_frac_interpolated", "n2co2_frac_interpolated", "cocl_frac_interpolated", "h2o_frac_interpolated", "perco_frac_interpolated", "melt1_frac_interpolated", "melt2_frac_interpolated", "Tmax_grid_interpolated", "Tmean_markers_interpolated"]
df_interpolated = pd.DataFrame(columns=interpolated_labels)

# Define which reservoir we look at
for res in reservoir_list:

    ### Generate/read grid points for desired values at which the data needs to be interpolated
    for imf in [ "powerlaw" ]: # , "gaussian"

        df_plts = pd.read_csv(dat_dir+"sfd_dMdt_all.csv")
        if imf == "powerlaw":
            df_plts = df_plts.loc[(df_plts['imf'] == imf) & (df_plts['res'] == res)]
        if imf == "gaussian":
            df_plts = df_plts.loc[(df_plts['imf'] == imf) & (df_plts['res'] == res)]

        # Generate or load interpolation points on 3D grid
        if (os.path.isfile(csv_dir+"grid_3d_interpolated_points_"+res+"_"+imf+".csv")):

            print  (datetime.datetime.now(), "->", res, imf, ": Import interpolation grid points")

            with open(csv_dir+"grid_3d_interpolated_points_"+res+"_"+imf+".csv", "r") as csvfile:
                grid_interpolated_points = []
                for row in csv.reader(csvfile):
                    grid_interpolated_points.append(row)

            with open(csv_dir+"grid_3d_interpolated_points_tform_real_"+res+"_"+imf+".csv", "r") as csvfile:
                grid_interpolated_points_tform_real = []
                for row in csv.reader(csvfile):
                    grid_interpolated_points_tform_real.append(row)

        else:

            print (datetime.datetime.now(), "->", res, imf, ": Write interpolation grid points")

            grid_interpolated_points                = list([])
            grid_interpolated_points_tform_real     = list([])

            running_time = df_plts[df_plts['imf']==imf].tform.tolist()

            for idx, time in enumerate(running_time):

                current_line = df_plts.loc[(df_plts['imf'] == imf) & (df_plts['tform'] == time)]
                current_line = current_line.drop(["overflow", "tform", "imf", "res"], axis=1)

                rad_list = current_line.columns.tolist()

                for rad_bin in rad_list:

                    r_plts = round(float(rad_bin),2)

                    total_bin_mass = current_line[rad_bin].tolist()[0]

                    # Exclude tforms outside of planetesimal phase space
                    for tform in running_time[:idx+1]:
                        tform = round(tform,6)

                        if tform > 3.0:
                            tform_print = 3.0
                        else:
                            tform_print = tform

                        grid_interpolated_points.append([r_plts, tform_print, round(time,6)]) 
                        grid_interpolated_points_tform_real.append([r_plts, tform, round(time,6)]) 

            grid_interpolated_points = np.asarray(grid_interpolated_points)
            grid_interpolated_points_tform_real = np.asarray(grid_interpolated_points_tform_real)

            with open(csv_dir+"grid_3d_interpolated_points_"+res+"_"+imf+".csv", "w") as f:
                writer = csv.writer(f)
                writer.writerows(grid_interpolated_points)

            with open(csv_dir+"grid_3d_interpolated_points_tform_real_"+res+"_"+imf+".csv", "w") as f:
                writer = csv.writer(f)
                writer.writerows(grid_interpolated_points_tform_real)

        if (os.path.isfile(csv_dir+"grid_3d_interpolated_values_"+res+"_"+imf+".csv")):

            print  (datetime.datetime.now(), "->", res, imf, ": Import interpolated values")

            with open(csv_dir+"grid_3d_interpolated_values_"+res+"_"+imf+".csv", "r") as csvfile:
                grid_interpolated_values = []
                for row in csv.reader(csvfile):
                    grid_interpolated_values.append(list([float(k) for k in row]))
                     
                sol_frac_interpolated     = np.asarray([float(k) for k in list(grid_interpolated_values[0])])
                liq_frac_interpolated     = np.asarray([float(k) for k in list(grid_interpolated_values[1])])
                hydrous_frac_interpolated = np.asarray([float(k) for k in list(grid_interpolated_values[2])])
                primitive_frac_interpolated = np.asarray([float(k) for k in list(grid_interpolated_values[3])])
                n2co2_frac_interpolated   = np.asarray([float(k) for k in list(grid_interpolated_values[4])])
                cocl_frac_interpolated    = np.asarray([float(k) for k in list(grid_interpolated_values[5])])
                h2o_frac_interpolated     = np.asarray([float(k) for k in list(grid_interpolated_values[6])])
                perco_frac_interpolated   = np.asarray([float(k) for k in list(grid_interpolated_values[7])])
                melt1_frac_interpolated   = np.asarray([float(k) for k in list(grid_interpolated_values[8])])
                melt2_frac_interpolated   = np.asarray([float(k) for k in list(grid_interpolated_values[9])])
                Tmax_grid_interpolated    = np.asarray([float(k) for k in list(grid_interpolated_values[10])])
                Tmean_markers_interpolated  = np.asarray([float(k) for k in list(grid_interpolated_values[11])])
        else:

            print  (datetime.datetime.now(), "->", res, imf, ": Interpolate data")

            interpolate_sol_frac        = scipy.interpolate.LinearNDInterpolator(grid_points, sol_frac_data)
            sol_frac_interpolated       = interpolate_sol_frac(grid_interpolated_points)
            interpolate_liq_frac        = scipy.interpolate.LinearNDInterpolator(grid_points, liq_frac_data)
            liq_frac_interpolated       = interpolate_liq_frac(grid_interpolated_points)
            interpolate_hydrous_frac    = scipy.interpolate.LinearNDInterpolator(grid_points, hydrous_frac_data)
            hydrous_frac_interpolated   = interpolate_hydrous_frac(grid_interpolated_points)
            interpolate_primitive_frac  = scipy.interpolate.LinearNDInterpolator(grid_points, primitive_frac_data)
            primitive_frac_interpolated = interpolate_primitive_frac(grid_interpolated_points)
            interpolate_n2co2_frac      = scipy.interpolate.LinearNDInterpolator(grid_points, n2co2_frac_data)
            n2co2_frac_interpolated     = interpolate_n2co2_frac(grid_interpolated_points)
            interpolate_cocl_frac       = scipy.interpolate.LinearNDInterpolator(grid_points, cocl_frac_data)
            cocl_frac_interpolated      = interpolate_cocl_frac(grid_interpolated_points)
            interpolate_h2o_frac        = scipy.interpolate.LinearNDInterpolator(grid_points, h2o_frac_data)
            h2o_frac_interpolated       = interpolate_h2o_frac(grid_interpolated_points)
            interpolate_perco_frac      = scipy.interpolate.LinearNDInterpolator(grid_points, perco_frac_data)
            perco_frac_interpolated     = interpolate_perco_frac(grid_interpolated_points)
            interpolate_melt1_frac      = scipy.interpolate.LinearNDInterpolator(grid_points, melt1_frac_data)
            melt1_frac_interpolated     = interpolate_melt1_frac(grid_interpolated_points)
            interpolate_melt2_frac      = scipy.interpolate.LinearNDInterpolator(grid_points, melt2_frac_data)
            melt2_frac_interpolated     = interpolate_melt2_frac(grid_interpolated_points)
            interpolate_Tmax_grid       = scipy.interpolate.LinearNDInterpolator(grid_points, Tmax_grid_data)
            Tmax_grid_interpolated      = interpolate_Tmax_grid(grid_interpolated_points)
            interpolate_Tmean_markers   = scipy.interpolate.LinearNDInterpolator(grid_points, Tmean_markers_data)
            Tmean_markers_interpolated  = interpolate_Tmean_markers(grid_interpolated_points)

            print (datetime.datetime.now(), "Write grid_3d_interpolated_values_"+res+"_"+imf+".csv")

            grid_interpolated_values = [
                                        np.asarray(sol_frac_interpolated),
                                        np.asarray(liq_frac_interpolated),
                                        np.asarray(hydrous_frac_interpolated),
                                        np.asarray(primitive_frac_interpolated),
                                        np.asarray(n2co2_frac_interpolated),
                                        np.asarray(cocl_frac_interpolated),
                                        np.asarray(h2o_frac_interpolated),
                                        np.asarray(perco_frac_interpolated),
                                        np.asarray(melt1_frac_interpolated),
                                        np.asarray(melt2_frac_interpolated),
                                        np.asarray(Tmax_grid_interpolated),
                                        np.asarray(Tmean_markers_interpolated)
                                        ]

            with open(csv_dir+"grid_3d_interpolated_values_"+res+"_"+imf+".csv", "w") as f:
                writer = csv.writer(f)
                writer.writerows(grid_interpolated_values)

        # Check for NaNs, and replace with sensable values at edges of hypersurface
        if np.size(np.argwhere(np.isnan(grid_interpolated_values))) > 0:
            
            # For each field, like 'h2o_frac_interpolated'
            for meta_idx, field in enumerate(grid_interpolated_values):
                
                # For each index in each field
                for idx, value in enumerate(field):
                    
                    if math.isnan(float(value)):
                        
                        # Cases where the interpolation error is in the origin of the hypersurface, i.e., tform = 0.0, time = 0.0
                        rad_point   = grid_interpolated_points[idx][0]
                        tform_point = grid_interpolated_points[idx][1]
                        time_point  = grid_interpolated_points[idx][2]
                        
                        if tform_point == '0.0' and time_point == '0.0':
                            # Distinguish fields, set to values appropriate for beginning
                            sol_frac_interpolated[idx]       = 1.0
                            liq_frac_interpolated[idx]       = 0.0
                            hydrous_frac_interpolated[idx]   = 0.0
                            primitive_frac_interpolated[idx] = 1.0
                            n2co2_frac_interpolated[idx]     = 1.0
                            cocl_frac_interpolated[idx]      = 1.0
                            h2o_frac_interpolated[idx]       = 1.0
                            perco_frac_interpolated[idx]     = 0.0
                            melt1_frac_interpolated[idx]     = 0.0
                            melt2_frac_interpolated[idx]     = 0.0
                            Tmax_grid_interpolated[idx]      = 150.0
                            Tmean_markers_interpolated[idx]  = 150.0
                        else:
                            # If values beyond starting values appear, check for each
                            print(meta_idx, idx, ":", rad_point, tform_point, time_point, ":", value, "--> RESET!")

            grid_interpolated_values = [
                                        np.asarray(sol_frac_interpolated),
                                        np.asarray(liq_frac_interpolated),
                                        np.asarray(hydrous_frac_interpolated),
                                        np.asarray(primitive_frac_interpolated),
                                        np.asarray(n2co2_frac_interpolated),
                                        np.asarray(cocl_frac_interpolated),
                                        np.asarray(h2o_frac_interpolated),
                                        np.asarray(perco_frac_interpolated),
                                        np.asarray(melt1_frac_interpolated),
                                        np.asarray(melt2_frac_interpolated),
                                        np.asarray(Tmax_grid_interpolated),
                                        np.asarray(Tmean_markers_interpolated)
                                        ]

            print (datetime.datetime.now(), "->", res, imf, ": Re-write grid_3d_interpolated_values.csv due to NaNs")

            with open(csv_dir+"grid_3d_interpolated_values_"+res+"_"+imf+".csv", "w") as f:
                writer = csv.writer(f)
                writer.writerows(grid_interpolated_values)
        else:
            print (datetime.datetime.now(), "->", res, imf, ": No NaN values in interpolated values")

        # Create pandas dataframe from interpolated data points + values
        rplts_list  = []
        tform_list  = []
        time_list   = []
        imf_list    = []
        res_list    = []

        for idx, line in enumerate(grid_interpolated_points_tform_real):

            r_plts  = round(float(line[0]),2)
            tform   = round(float(line[1]),6)
            time    = round(float(line[2]),6)
            rplts_list.append(r_plts)
            tform_list.append(tform)
            time_list.append(time)
            imf_list.append(imf)
            res_list.append(res)

        # Add to complete interpolated dataframe
        data_tuples = list(zip(res_list,rplts_list,tform_list,time_list,imf_list,sol_frac_interpolated, liq_frac_interpolated, hydrous_frac_interpolated, primitive_frac_interpolated, n2co2_frac_interpolated, cocl_frac_interpolated, h2o_frac_interpolated, perco_frac_interpolated, melt1_frac_interpolated, melt2_frac_interpolated, Tmax_grid_interpolated, Tmean_markers_interpolated))
        df_interpolated2 = pd.DataFrame(data_tuples, columns=interpolated_labels)
        df_interpolated = df_interpolated.append(df_interpolated2, sort=False, ignore_index=True, verify_integrity=True)

    # Write to file
    df_interpolated = df_interpolated.dropna()
    df_interpolated = df_interpolated.round({'time': 6, 'tform': 6})
    df_interpolated.to_csv(dat_dir+"sfd_interpolated.csv", index=None)

    print (datetime.datetime.now(), "Write SFD interpolation: sfd_interpolated.csv")
