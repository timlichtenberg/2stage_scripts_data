from jd_plot import *

# Plot settings
lw      = 2.5
fscale  = 2.0
fsize   = 20

sns.set_style("whitegrid")
sns.set(style="ticks", font_scale=fscale)

cmap = "YlGnBu"

df = pd.read_csv(dat_dir+"plts_all_data.csv")

RUN_LIST_ALL = natsorted(set(df.run.tolist()))


RUN_LIST       = [ 
                    "r003t030al05250fe01150tmp150", # R1
                    "r010t030al05250fe01150tmp150", # R2
                    "r300t030al05250fe01150tmp150", # R3
                    "r100t250al05250fe01150tmp150", # T3
                    "r100t072al05250fe01150tmp150", # T2
                    "r100t030al05250fe01150tmp150", # T1
                    
                    ]

fig = plt.figure(figsize=(10,9))
ax1 = fig.add_subplot(111)

c_counter1 = 0
c_counter2 = 0

rad_palette = sns.color_palette("Set1")

for RUN in RUN_LIST:

    time = df[df['run']==RUN].time.tolist()
    quantity = df[df['run']==RUN].primitive_frac.tolist()

    colorpalette_color = rad_palette[c_counter1]
    ls="-"

    print(RUN, colorpalette_color)

    if RUN == RUN_LIST[0]:
        ls    = ":"
        color = qgreen
        char  = "(R1) " 
    if RUN == RUN_LIST[1]:
        ls    = "--"
        color = qgreen
        char  = "(R2) " 
    if RUN == RUN_LIST[2]:
        ls    = "-"
        color = qgreen
        char  = "(R3) " 
    if RUN == RUN_LIST[3]:
        ls    = ":"
        color = qmagenta
        char  = "(T3) " 
    if RUN == RUN_LIST[4]:
        ls    = "--"
        color = qmagenta
        char  = "(T2) " 
    if RUN == RUN_LIST[5]:
        ls    = "-"
        color = qmagenta
        char  = "(T1) " 

    r_init   = int(float(RUN[1:4]))     # km
    t_form   = float(RUN[5:8])*1e-2     # Myr
    al_init  = float(RUN[10:15])*1e-8   # 26Al/27Al # to normalize: /5.25e-5

    tstart_str = str(float(RUN[1:4]))

    plt.plot(time, quantity, color=color, ls=ls, lw=2.5, label=char+str(r_init)+" km, "+str(t_form)+" Myr")

    c_counter1+=1

sns.set_style("whitegrid")
sns.despine(ax=ax1, top=True, right=True, left=False, bottom=False)
sns.set(style="ticks", font_scale=fscale)

xticks      = [ 0.3, 0.5, 1, 2, 3, 5 ]
xticklabels = [ "0.3", "0.5", "1", "2", "3", "5" ]
yticks      = [ 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1 ]
yticklabels = [ str(round(i*100)) for i in yticks ]

ax1.text(0.98, 0.98, 'B', color="k", rotation=0, ha="right", va="top", fontsize=fsize+10, transform=ax1.transAxes, bbox=dict(fc='white', ec="white", alpha=0.01, pad=0.1, boxstyle='round'))

plt.xscale("log")
plt.yscale("log")
plt.xlim(left=np.min(xticks), right=np.max(xticks))
plt.ylim(top=np.max(yticks), bottom=np.min(yticks))

ax1.set_xticks( xticks )
ax1.set_xticklabels( xticklabels, fontsize=fsize+2 )
ax1.set_yticks( yticks )
ax1.set_yticklabels( yticklabels, fontsize=fsize+2 )

legend = ax1.legend(loc=3, title=r"Planetesimal radius $R_\mathrm{P}$, formation time $t_\mathrm{form}$", fontsize=fsize-2, ncol=2)
plt.setp(legend.get_title(), fontsize=fsize-2)

plt.ylabel(r"Retained water ice, $f_{\mathrm{ice}}$ (vol%)", fontsize=fsize+5)
plt.xlabel(r"Time after CAI formation, $\Delta t_\mathrm{CAI}$ (Myr)", fontsize=fsize+5)

figure_name="fig_s5b"+".pdf"
plt.savefig(fig_dir+figure_name, bbox_inches='tight')
