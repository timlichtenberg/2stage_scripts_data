from jd_plot import *

fscale  = 3.5
fsize   = 20
sns.set(font_scale=fscale)

sns.axes_style("white")
sns.set_style("ticks")

df = pd.read_csv(dat_dir+"plts_all_data.csv")

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
    perco_frac      = list([])
    melt1_frac      = list([])
    melt2_frac      = list([])
    Tmax_grid       = list([])
    Tmean_markers   = list([])

    # for FILE in FILE_LIST:
    for RUN in RUN_LIST:

        r_init   = float(RUN[1:4])         # km
        t_init   = float(RUN[5:8])/100     # Myr
        al_init  = float(RUN[10:15])*1e-8/5.25e-5  # 26Al/27Al # to normalize: /5.25e-5
        fe_init  = float(RUN[17:22])*1e-11 # 60Fe/56Fe
        tmp_init = float(RUN[25:28])       # K

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

        rad.append(r_init)
        tform.append(t_init)

        sol_frac.append(sol_frac_value)
        liq_frac.append(liq_frac_value)
        hydrous_frac.append(hydrous_frac_value)
        primitive_frac.append(primitive_frac_value)
        n2co2_frac.append(n2co2_frac_value)
        cocl_frac.append(cocl_frac_value)
        h2o_frac.append(h2o_frac_value)
        perco_frac.append(perco_frac_value)
        melt1_frac.append(melt1_frac_value)
        melt2_frac.append(melt2_frac_value)
        Tmax_grid.append(Tmax_grid_value)
        Tmean_markers.append(Tmean_markers_value)

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

xi = np.linspace(min(rad), max(rad), 100)
yi = np.linspace(min(tform), max(tform), 100)

grid_x, grid_y = np.mgrid[min(rad):max(rad):100j, min(tform):max(tform):100j]

points = np.stack([rad, tform], axis=1)

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

label_font      = 22
tick_font       = 20
annotate_font   = 16
lw              = 10
tw              = 1.5


fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(111)
plt.tight_layout(pad=0.2, w_pad=1, h_pad=0.2)

divider = make_axes_locatable(ax1)
cax = divider.append_axes("top", size="4%", pad="2%")
fig.add_axes(cax, label='a')

zi = zi_melt2_frac

cbar_ticks = [ np.min(zi_melt2_frac), 0.2, 0.4, 0.6, 0.8, np.max(zi_melt2_frac) ]

no_color_fields = 255
cbar_label = r"Metal rain-out core formation, $f_\mathrm{rain, final}$ (vol%)"
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
cmap = "magma_r"
cbar_ticks_labels = [ str(int(round(round(k,2)*100))) for k in cbar_ticks ]

nloops = 3
for i in range(0, nloops):
    CS = ax1.contourf(grid_x, grid_y, zi, no_color_fields, cmap=cmap)

cbar = fig.colorbar(CS, cax=cax, orientation="horizontal", ticks=cbar_ticks)
cbar.outline.set_edgecolor('black')
cbar.outline.set_linewidth(1)

ax1.scatter(rad, tform, marker='o', facecolors='white', edgecolors='black', s=10, zorder=5, lw=1.0, alpha=0.5)

tform_resI  = np.ma.masked_where((tform > 0.3), tform)
rad_resI    = np.ma.masked_where((tform > 0.3), rad)
tform_resI  = np.ma.masked_where((tform < 0.1), tform_resI)
rad_resI    = np.ma.masked_where((tform < 0.1), rad_resI)
ax1.scatter(rad_resI, tform_resI, marker='o', facecolors=reds[3], s=25, zorder=5, lw=1.0, alpha=0.99)

tform_resII = np.ma.masked_where((tform <= 0.7), tform)
rad_resII   = np.ma.masked_where((tform <= 0.7), rad)
ax1.scatter(rad_resII, tform_resII, marker='o', facecolors=blues[3], s=25, zorder=5, lw=1.0, alpha=0.99)

# Special points for plot B
rad_B   = np.asarray([ 13., 30., 300. ])
tform_B = np.asarray([ 0.3, 0.3, 0.3 ])
print(rad_B, tform_B)
ax1.scatter(rad_B, tform_B, marker='o', facecolors=qgreen, edgecolors="white", s=120, zorder=6, lw=1.0, alpha=0.99)

# Special points for plot B
rad_B   = np.asarray([ 100., 100., 100. ])
tform_B = np.asarray([ 0.3, 0.72, 1.0 ])
print(rad_B, tform_B)
ax1.scatter(rad_B, tform_B, marker='o', facecolors=qmagenta, edgecolors="white", s=120, zorder=6, lw=1.0, alpha=0.99)

R1_text = ax1.text(12.0, 0.286, 'R1', color=qgreen, rotation=0, fontsize=fsize+2, ha="right", va="top", zorder=10, bbox=dict(fc='white', ec="white", alpha=0.0, pad=0.1, boxstyle='round'))
R2_text = ax1.text(28, 0.286, 'R2', color=qgreen, rotation=0, fontsize=fsize+2, ha="right", va="top", zorder=10, bbox=dict(fc='white', ec="white", alpha=0.0, pad=0.1, boxstyle='round'))
R3_text = ax1.text(285, 0.286, 'R3', color=qgreen, rotation=0, fontsize=fsize+2, ha="right", va="top", zorder=10, bbox=dict(fc='white', ec="white", alpha=0.0, pad=0.1, boxstyle='round'))
T1_text = ax1.text(93, 0.286, 'T1', color=qmagenta, rotation=0, fontsize=fsize+2, ha="right", va="top", zorder=10, bbox=dict(fc='white', ec="white", alpha=0.0, pad=0.1, boxstyle='round'))
T2_text = ax1.text(93, 0.68, 'T2', color=qmagenta, rotation=0, fontsize=fsize+2, ha="right", va="top", zorder=10, bbox=dict(fc='white', ec="white", alpha=0.0, pad=0.1, boxstyle='round'))
T3_text = ax1.text(93, 0.97, 'T3', color=qmagenta, rotation=0, fontsize=fsize+2, ha="right", va="top", zorder=10, bbox=dict(fc='white', ec="white", alpha=0.0, pad=0.1, boxstyle='round'))

ResI_text = ax1.text(1.05, 0.57, 'Reservoir II\nplanetesimals', color=blues[3], rotation=0, fontsize=fsize+1, ha="left", va="center", bbox=dict(fc='white', ec="white", alpha=0.0, pad=0.1, boxstyle='round'), zorder=15)
nf_text = ax1.text(1.05, 0.36, 'not formed', color="grey", rotation=0, fontsize=fsize-3, ha="left", va="center", alpha=0.8, zorder=15)
ResII_text = ax1.text(1.05, 0.15, 'Reservoir I\nplanetesimals', color=reds[3], rotation=0, fontsize=fsize+1, ha="left", va="center", bbox=dict(fc='white', ec="white", alpha=0.0, pad=0.1, boxstyle='round'), zorder=15)

for text in [ R1_text, R2_text, R3_text, T1_text, T2_text, T3_text ]:
    text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='white'),
                       path_effects.Normal()])

# for text in [ R1_text ]:
#     text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
#                        path_effects.Normal()])

ax1.text(-0.12, +1.13, 'A', color="k", rotation=0, ha="left", va="top", fontsize=fsize+10, transform=ax1.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))

cbar.set_label(cbar_label, fontsize=fsize+7, labelpad=+20)
cbar.ax.tick_params(labelsize=fsize+2) 
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.set_xticklabels(cbar_ticks_labels)

ax1.set_xlim(1, 300)
ax1.set_ylim(0.1, 3.0)
ax1.set_xscale("log") 
ax1.set_yscale("log") 
ax1.set_xticks([1, 2, 3, 5, 7, 10, 20, 30, 50, 100, 300])
ax1.set_xticklabels(["1", "2", "3", "5", "7", "10", "20", "30", "50", "100", "300"], fontsize=fsize+2)
ax1.set_yticks([0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 3])
ax1.set_yticklabels(["0.1", "0.2", "0.3", "0.5", "0.7", "1", "2", "3"], fontsize=fsize+2)
ax1.xaxis.tick_bottom()

ax1.tick_params(axis='x', which='both', right='off', top='off', labelsize=fsize+7, width=tw, color="white", pad=7.)
ax1.tick_params(axis='y', which='both', right='off', top='off', labelsize=fsize+7, width=tw, color="white", pad=1.)

ax1.set_xlabel(r"Planetesimal radius, $R_{\mathrm{P}}$ (km)", fontsize=fsize+5)
ax1.set_ylabel(r"Formation time after CAI formation (Myr)", fontsize=fsize+5)
ax1.xaxis.set_label_coords(0.5, -0.1)

plt.gca().axes.get_yaxis().set_visible(False)

plt.savefig(fig_dir+"fig_s7a"+".pdf", bbox_inches='tight')
