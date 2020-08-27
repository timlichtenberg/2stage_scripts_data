from jd_plot import *

fscale  = 3.5
fsize   = 22
sns.set(font_scale=fscale)

sns.axes_style("white")
sns.set_style("ticks")

# Working directories
working_dir = os.getcwd()
dat_dir     = working_dir+"/data/planetesimal_evolution/"
fig_dir     = working_dir+"/figures/"

df = pd.read_csv(dat_dir+"plts_all_data.csv")

# Check if data was already read in
if os.path.isfile(dat_dir+"plts_all_data.csv"):
    df = pd.read_csv(dat_dir+"plts_all_data.csv")
else:

    # Select files
    FILES       = [ "r???t???al05250fe01150tmp150.dat" ] #r??0p??ht???

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
        RUN      = str(FILE[:-4])
        r_init   = float(RUN[1:4])          # km
        t_form   = float(RUN[5:8])*1e-2     # Myr
        al_init  = float(RUN[10:15])*1e-8   # 26Al/27Al # to normalize: /5.25e-5
        # t_form   = abs(-1.0344*np.log(al_init/5.25e-5))  # Myr

        # print FILE, r_init, t_form, al_init

        # Open a file
        file_name=dat_dir+FILE

        print (file_name)

        df2 = pd.read_csv(file_name, sep=" ", names=["time","sol_frac","liq_frac","hydrous_frac","primitive_frac","n2co2_frac","cocl_frac","h2o_frac","phyllo1_frac","phyllo2_frac","phyllo3_frac","phyllo4_frac","perco_frac","melt1_frac","melt2_frac","maxtk","t_max_body","meantk","t_mean_body","count_toohot"])
        # https://www.interviewqs.com/ddi_code_snippets/add_new_col_df_default_value
        df2.insert(0, "run", RUN)
        df2.insert(1, "rad", r_init)
        df2.insert(2, "tform", t_form)

        # df2["rad"] = r_init
        # df2["tform"] = t_form

        df = df.append(df2, sort=False)
        df = df.dropna()

    # Save to csv
    df.to_csv(dat_dir+"plts_all_data.csv")

RUN_LIST = natsorted(set(df.run.tolist()))


recalc = 0
if (os.path.isfile(dat_dir+"grid_parameter_space.csv")) and (recalc == 0):
    with open(dat_dir+"grid_parameter_space.csv", 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        grid_parameter_space = []
        for row in csvreader:
            grid_parameter_space.append(row)

        rad             = [ float(k) for k in grid_parameter_space[0]]
        tform           = [ float(k) for k in grid_parameter_space[1]]
        sol_frac        = [ float(k) for k in grid_parameter_space[2]]
        liq_frac        = [ float(k) for k in grid_parameter_space[3]]
        hydrous_frac    = [ float(k) for k in grid_parameter_space[4]]
        primitive_frac  = [ float(k) for k in grid_parameter_space[5]]
        n2co2_frac      = [ float(k) for k in grid_parameter_space[6]]
        cocl_frac       = [ float(k) for k in grid_parameter_space[7]]
        h2o_frac        = [ float(k) for k in grid_parameter_space[8]]
        # phyllo1_frac,
        # phyllo2_frac,
        # phyllo3_frac,
        # phyllo4_frac,
        perco_frac      = [ float(k) for k in grid_parameter_space[9]]
        melt1_frac      = [ float(k) for k in grid_parameter_space[10]]
        melt2_frac      = [ float(k) for k in grid_parameter_space[11]]
        Tmax_grid       = [ float(k) for k in grid_parameter_space[12]]
        Tmean_markers   = [ float(k) for k in grid_parameter_space[13]]

else:

    rad             = list([])
    tform           = list([])
    al              = list([])
    sol_frac        = list([])
    liq_frac        = list([])
    hydrous_frac    = list([])
    primitive_frac  = list([])
    n2co2_frac      = list([])
    cocl_frac       = list([])
    h2o_frac        = list([])
    # phyllo1_frac    = list([])
    # phyllo2_frac    = list([])
    # phyllo3_frac    = list([])
    # phyllo4_frac    = list([])
    perco_frac      = list([])
    melt1_frac      = list([])
    melt2_frac      = list([])
    Tmax_grid       = list([])
    Tmean_markers   = list([])

    # for FILE in FILE_LIST:
    for RUN in RUN_LIST:

        # Get the settings of the current file
        # RUN      = str(FILE[:-4])
        r_init   = float(RUN[1:4])         # km
        t_init   = float(RUN[5:8])/100    # Myr
        al_init  = float(RUN[10:15])*1e-8/5.25e-5  # 26Al/27Al # to normalize: /5.25e-5
        fe_init  = float(RUN[17:22])*1e-11 # 60Fe/56Fe
        tmp_init = float(RUN[25:28])       # K
        # t_current = str(round(Time/1e6,2))

        print ("### Analyse:", RUN, r_init, t_init, al_init, fe_init, tmp_init)

        sol_frac_value = df[df['run']==RUN].iloc[(df[df['run']==RUN]['sol_frac']-0.0).abs().argsort()[:1]].sol_frac.tolist()[0]

        liq_frac_value = df[df['run']==RUN].iloc[(df[df['run']==RUN]['liq_frac']-1.0).abs().argsort()[:1]].liq_frac.tolist()[0]

        hydrous_frac_value = df[df['run']==RUN].iloc[(df[df['run']==RUN]['time']-1.0).abs().argsort()[:1]].hydrous_frac.tolist()[0]

        primitive_frac_value = df[df['run']==RUN].iloc[(df[df['run']==RUN]['time']-10.0).abs().argsort()[:1]].primitive_frac.tolist()[0]

        n2co2_frac_value = df[df['run']==RUN].iloc[(df[df['run']==RUN]['time']-10.0).abs().argsort()[:1]].n2co2_frac.tolist()[0]

        cocl_frac_value = df[df['run']==RUN].iloc[(df[df['run']==RUN]['time']-10.0).abs().argsort()[:1]].cocl_frac.tolist()[0]

        h2o_frac_value = df[df['run']==RUN].iloc[(df[df['run']==RUN]['time']-10.0).abs().argsort()[:1]].h2o_frac.tolist()[0]

        perco_frac_value = df[df['run']==RUN].iloc[(df[df['run']==RUN]['perco_frac']-1.0).abs().argsort()[:1]].perco_frac.tolist()[0]

        melt1_frac_value = df[df['run']==RUN].iloc[(df[df['run']==RUN]['melt1_frac']-1.0).abs().argsort()[:1]].melt1_frac.tolist()[0]

        melt2_frac_value = df[df['run']==RUN].iloc[(df[df['run']==RUN]['melt2_frac']-1.0).abs().argsort()[:1]].melt2_frac.tolist()[0]

        Tmax_grid_value = df[df['run']==RUN].iloc[(df[df['run']==RUN]['maxtk']-5000.0).abs().argsort()[:1]].maxtk.tolist()[0]

        Tmean_markers_value = df[df['run']==RUN].iloc[(df[df['run']==RUN]['t_mean_body']-5000.0).abs().argsort()[:1]].t_mean_body.tolist()[0]

        # plot_quantity = perco_frac_max

        # Parameter space
        rad.append(r_init)
        tform.append(t_init)

        # Define values
        sol_frac.append(sol_frac_value)
        liq_frac.append(liq_frac_value)
        hydrous_frac.append(hydrous_frac_value)
        primitive_frac.append(primitive_frac_value)
        n2co2_frac.append(n2co2_frac_value)
        cocl_frac.append(cocl_frac_value)
        h2o_frac.append(h2o_frac_value)
        # phyllo1_frac.append()
        # phyllo2_frac.append()
        # phyllo3_frac.append()
        # phyllo4_frac.append()
        perco_frac.append(perco_frac_value)
        melt1_frac.append(melt1_frac_value)
        melt2_frac.append(melt2_frac_value)
        Tmax_grid.append(Tmax_grid_value)
        Tmean_markers.append(Tmean_markers_value)

    # https://stackoverflow.com/questions/14037540/writing-a-python-list-of-lists-to-a-csv-file
    # https://docs.python.org/2/library/csv.html
    grid_parameter_space = [
                                rad,
                                tform,
                                sol_frac,
                                liq_frac,
                                hydrous_frac,
                                primitive_frac,
                                n2co2_frac,
                                cocl_frac,
                                h2o_frac,
                                # phyllo1_frac,
                                # phyllo2_frac,
                                # phyllo3_frac,
                                # phyllo4_frac,
                                perco_frac,
                                melt1_frac,
                                melt2_frac,
                                Tmax_grid,
                                Tmean_markers
                            ]

    print(grid_parameter_space)
    with open(dat_dir+"grid_parameter_space.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(grid_parameter_space)

rad                 = np.asarray(rad)
tform               = np.asarray(tform)
sol_frac            = np.asarray(sol_frac)
liq_frac            = np.asarray(liq_frac)
hydrous_frac        = np.asarray(hydrous_frac)
primitive_frac      = np.asarray(primitive_frac)
n2co2_frac          = np.asarray(n2co2_frac)
cocl_frac           = np.asarray(cocl_frac)
h2o_frac            = np.asarray(h2o_frac)
perco_frac          = np.asarray(perco_frac)
melt1_frac          = np.asarray(melt1_frac)
melt2_frac          = np.asarray(melt2_frac)
Tmax_grid           = np.asarray(Tmax_grid)
Tmean_markers       = np.asarray(Tmean_markers)

# define grid.
xi = np.linspace(min(rad), max(rad), 100)
yi = np.linspace(min(tform), max(tform), 100)

grid_x, grid_y = np.mgrid[min(rad):max(rad):100j, min(tform):max(tform):100j]

points = np.stack([rad, tform], axis=1)


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
zi_sol_frac = scipy.interpolate.griddata(points, sol_frac, (grid_x, grid_y), method='linear', rescale=False)
zi_liq_frac = scipy.interpolate.griddata(points, liq_frac, (grid_x, grid_y), method='linear', rescale=False)
zi_hydrous_frac = scipy.interpolate.griddata(points, hydrous_frac, (grid_x, grid_y), method='linear', rescale=False)
zi_primitive_frac = scipy.interpolate.griddata(points, primitive_frac, (grid_x, grid_y), method='linear', rescale=False)
zi_n2co2_frac = scipy.interpolate.griddata(points, n2co2_frac, (grid_x, grid_y), method='linear', rescale=False)
zi_cocl_frac = scipy.interpolate.griddata(points, cocl_frac, (grid_x, grid_y), method='linear', rescale=False)
zi_h2o_frac = scipy.interpolate.griddata(points, h2o_frac, (grid_x, grid_y), method='linear', rescale=False)
zi_perco_frac = scipy.interpolate.griddata(points, perco_frac, (grid_x, grid_y), method='linear', rescale=False)
zi_melt1_frac = scipy.interpolate.griddata(points, melt1_frac, (grid_x, grid_y), method='linear', rescale=False)
zi_melt2_frac = scipy.interpolate.griddata(points, melt2_frac, (grid_x, grid_y), method='linear', rescale=False)
zi_Tmax_grid = scipy.interpolate.griddata(points, Tmax_grid, (grid_x, grid_y), method='linear', rescale=False)
zi_Tmean_markers = scipy.interpolate.griddata(points, Tmean_markers, (grid_x, grid_y), method='linear', rescale=False)

zi_list = [ "zi_Tmax_grid", "zi_perco_frac" ]

# PLOT SETTINGS
label_font      = 22
tick_font       = 20
annotate_font   = 16
lw              = 10
tw              = 1.5

# loop over settings
# for zi in zi_list:

# print (zi)

fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(111)
plt.tight_layout(pad=0.2, w_pad=1, h_pad=0.2)

divider = make_axes_locatable(ax1)
cax = divider.append_axes("top", size="4%", pad="2%")
fig.add_axes(cax, label='a')

cbar_ticks = [ 0, 0.2, 0.4, 0.6, 0.8, 1.0 ]

zi = zi_Tmean_markers

# Colorbar settings
no_color_fields = 255
cbar_label = r"Highest mean temperature, $<T>_\mathrm{max}$ (K)"
cmap = sns.cubehelix_palette(light=1, as_cmap=True) # "magma", "YlGnBu", "viridis_r"
cbar_ticks        = [zi.min(), zi.min()+(zi.max()-zi.min())*0.2, zi.min()+(zi.max()-zi.min())*0.4, zi.min()+(zi.max()-zi.min())*0.6, zi.min()+(zi.max()-zi.min())*0.8, zi.max()]
cbar_ticks        = [ int(round(k)) for k in cbar_ticks ]
cbar_ticks_labels = cbar_ticks


### Color field loops for smoothing contourf
nloops = 3

for i in range(0, nloops):
    CS = ax1.contourf(grid_x, grid_y, zi, no_color_fields, cmap=cmap)

cbar = fig.colorbar(CS, cax=cax, orientation="horizontal", ticks=cbar_ticks)
cbar.outline.set_edgecolor('black')
cbar.outline.set_linewidth(1)

## Differenly colored data points for reservoirs

# # Scatter dots for *all* individual simulations 
# ax1.scatter(rad, tform, marker='o', facecolors='white', edgecolors='black', s=10, zorder=1, lw=1.0, alpha=0.5)

# Red dots for Reservoir I
tform_resI  = np.ma.masked_where((tform > 0.4), tform)
rad_resI    = np.ma.masked_where((tform > 0.4), rad)
tform_resI  = np.ma.masked_where((tform < 0.11), tform_resI)
rad_resI    = np.ma.masked_where((tform < 0.11), rad_resI)
ax1.scatter(rad_resI, tform_resI, marker='o', facecolors=reds[3], edgecolors='black', s=25, zorder=1, lw=0.5, alpha=0.8)

# Blue dots for Reservoir II
tform_resII = np.ma.masked_where((tform <= 0.7), tform)
rad_resII   = np.ma.masked_where((tform <= 0.7), rad)
ax1.scatter(rad_resII, tform_resII, marker='o', facecolors=blues[3], edgecolors='black', s=25, zorder=1, lw=0.5, alpha=0.8)


ax1.text(-0.12, +1.21, 'A', color="k", rotation=0, ha="left", va="top", fontsize=fsize+10, transform=ax1.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))

### Additional fields as contours

# Unaltered bodies
CS = ax1.contour(grid_x, grid_y, zi_primitive_frac, levels=[0.5], colors=qgreen, linestyles='--', linewidths=2.5)

CS = ax1.contour(grid_x, grid_y, zi_h2o_frac, levels=[0.1], colors=qgreen, linestyles='-', linewidths=2.5)

# Percolation / 1st stage core formation
CS = ax1.contour(grid_x, grid_y, zi_perco_frac, levels=[0.5], colors=qyellow, linestyles='--', linewidths=2.5)

# >0.4 melt fraction / 2nd stage core
CS = ax1.contour(grid_x, grid_y, zi_melt2_frac, levels=[0.9], colors=qyellow, linewidths=2.5)

## Annotation text
H1 = ax1.text(3.6, 0.11, 'H1', color=qgreen, rotation=0, fontsize=fsize+4, ha="left")
H2 = ax1.text(34, 0.11, 'H2', color=qgreen, rotation=0, fontsize=fsize+4, ha="left")
C1 = ax1.text(14.5, 0.11, 'C1', color=qyellow, rotation=0, fontsize=fsize+4, ha="left")
C2 = ax1.text(105, 0.11, 'C2', color=qyellow, rotation=0, fontsize=fsize+4, ha="left")

## Longish annotations
# H1_txt = ax1.text(4.2, 0.58, ' > 50 vol%\n'+r'$T_\mathrm{max} \geq T_\mathrm{hydr}$', color=qgreen, rotation=0, fontsize=fsize-3, ha="left", va="top")
# C1_txt = ax1.text(18, 0.45, ' > 50 vol%\n'+r'$T_\mathrm{max} \geq T_\mathrm{perc}$', color=qyellow, rotation=0, fontsize=fsize-3, ha="left", va="top")
# H2_txt = ax1.text(38, 0.28, ' > 90 vol%\n'+r'$T_\mathrm{max} \geq T_\mathrm{decomp}$', color=qgreen, rotation=0, fontsize=fsize-3, ha="left", va="top")
# C2_txt = ax1.text(110, 0.19, ' > 90 vol%\n'+r'$\phi_\mathrm{max} \geq \phi_\mathrm{rain}$', color=qyellow, rotation=0, fontsize=fsize-3, ha="left", va="top")

resI_txt = ax1.text(1.05, 0.6, 'Reservoir II planetesimals', color=blues[3], rotation=0, fontsize=fsize+1, ha="left", va="center", bbox=dict(fc='white', ec="white", alpha=0.0, pad=0.1, boxstyle='round'))
resII_txt = ax1.text(1.05, 0.24, 'Reservoir I planetesimals', color=reds[3], rotation=0, fontsize=fsize+1, ha="left", va="center", bbox=dict(fc='white', ec="white", alpha=0.0, pad=0.1, boxstyle='round'))
for text in [ resI_txt, resII_txt, H1, H2, C1, C2 ]:
    text.set_path_effects([path_effects.Stroke(linewidth=0.8, foreground='black'),
                       path_effects.Normal()])

# # Annotation for planetesimals that are not formed in the disk simulation
# nf_txt = ax1.text(1.05, 0.45, 'not formed', color="grey", rotation=0, fontsize=fsize-5, ha="left", va="center", alpha=0.5)

# for text in [ H1, H2, C1, C2, H1_txt, H2_txt, C1_txt, C2_txt ]:
#     text.set_path_effects([path_effects.Stroke(linewidth=0.8, foreground='white'),
#                        path_effects.Normal()])
# for text in [ resI_txt, resII_txt, nf_txt, H1, H1_txt ]:
#     text.set_path_effects([path_effects.Stroke(linewidth=0.8, foreground='black'),
#                        path_effects.Normal()])

# More colorbar settings
cbar.set_label(cbar_label, fontsize=fsize+7, labelpad=+20)
cbar.ax.tick_params(labelsize=fsize+4) 
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.set_xticklabels(cbar_ticks_labels)

# Axes settings
ax1.set_xlim(1, 300)
ax1.set_ylim(0.1, 3.0)
ax1.set_xscale("log") 
ax1.set_yscale("log") 
ax1.set_xticks([1, 2, 3, 5, 7, 10, 20, 30, 50, 100, 300])
ax1.set_xticklabels(["1", "2", "3", "5", "7", "10", "20", "30", "50", "100", "300"])
ax1.set_yticks([0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 3])
ax1.set_yticklabels(["0.1", "0.2", "0.3", "0.5", "0.7", "1", "2", "3"])
ax1.xaxis.tick_bottom()

ax1.tick_params(axis='x', which='both', right='off', top='off', labelsize=fsize+7, width=tw, color="white", pad=7.)
ax1.tick_params(axis='y', which='both', right='off', top='off', labelsize=fsize+7, width=tw, color="white", pad=1.)

ax1.set_xlabel(r"Planetesimal radius, $R_{\mathrm{P}}$ (km)", fontsize=fsize+8)
ax1.set_ylabel(r"Formation time after CAIs, $t_{\mathrm{form}}$ (Myr)", fontsize=fsize+8)
ax1.xaxis.set_label_coords(0.5, -0.1)

plt.gca().axes.get_yaxis().set_visible(False)

plt.savefig(fig_dir+"fig_3a"+".pdf", bbox_inches='tight')
plt.savefig(fig_dir+"fig_3a"+".jpg", bbox_inches='tight', dpi=300)
