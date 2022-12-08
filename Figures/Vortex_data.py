import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool
num_cores            = multiprocessing.cpu_count()
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
import time
from scipy.optimize import curve_fit
from itertools import combinations

num_vertex_mean = np.zeros(101)
list_num        = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
std_vertex      = np.zeros((len(list_num),len(num_vertex_mean)))
mean_distance   = np.zeros(101)
std_distance    = np.zeros((len(list_num),len(num_vertex_mean)))
def dist(p1, p2):
    (x1, y1), (x2, y2) = p1, p2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

for num in list_num:
    num_vertex = []
    L          = 16
    tind       = 0
    for T in np.linspace(0.1,1.5,101):
        XY = np.loadtxt('vortex_data/config'+str(num)+'/config_L'+str(L)+'_N101_T_'+str(T)+'.data')
        X,Y                = np.mgrid[0:L,0:L]
        vortex_density     = np.zeros((L-1,L-1))
        num_vortex         = 0
        num_vortex_        = 0
        num_antivortex     = 0
        x_pos_vortex       = []
        y_pos_vortex       = []
        x_pos_antivortex   = []
        y_pos_antivortex   = []
        x_pos_whole_vortex = []
        y_pos_whole_vortex = []
        for i in range(L-1):
            for j in range(L-1):
                vortex_density[i][j] = XY[i][j]+XY[i+1][j]+XY[i][j+1]+XY[i+1][j+1]#d1+d2+d3+d4
                coproj               = np.cos(XY[i][j])+np.cos(XY[i+1][j])+np.cos(XY[i][j+1])+np.cos(XY[i+1][j+1])
                sinproj              = np.sin(XY[i][j])+np.sin(XY[i+1][j])+np.sin(XY[i][j+1])+np.sin(XY[i+1][j+1])

                if np.abs(coproj) < 1 and np.abs(sinproj) < 1:
                    num_vortex +=1
                    if np.sign(vortex_density[i][j]) == +1:
                        x_pos_vortex.append(i+0.5)
                        y_pos_vortex.append(j+0.5)
                        num_vortex_ += 1
                    else:
                        x_pos_antivortex.append(i+0.5)
                        y_pos_antivortex.append(j+0.5)
                        num_antivortex +=1
                    x_pos_whole_vortex.append(i+0.5)
                    y_pos_whole_vortex.append(j+0.5)

        num_vertex.append(num_vortex/L**2)
        std_vertex[int(num-1)][tind] =  (num_vortex/L**2)
        num_vertex_mean[tind]       += (num_vortex/L**2)/len(list_num)
        points = list(zip(x_pos_whole_vortex,y_pos_whole_vortex))
        distances = [dist(p1, p2) for p1, p2 in combinations(points, 2)]
        if len(distances) != 0:
            avg_distance = sum(distances) / len(distances)
            mean_distance[tind]           += avg_distance/len(list_num)
            std_distance[int(num-1)][tind] = avg_distance
        tind += 1

x_theoretical = [0.6664025356576861   , 0.690174326465927    , 0.713946117274168    , 0.740095087163233    ,  0.768621236133122   ,  0.7995245641838352  ,  0.8328050713153724 , 0.8708399366085579   , 0.9088748019017432  ,  0.9516640253565768,  0.9992076069730587,  1.0515055467511885,  1.1109350237717908,  1.1774960380348654, 1.248811410459588, 1.3320126782884312,  1.4294770206022187, 1.5364500792393028, 1.6648177496038035,  1.8169572107765453,  2]
y_theoretical = [0.0003016591251885359, 0.0003016591251885359, 0.0006033182503770718, 0.0007541478129713536, 0.0010558069381598756, 0.0015082956259426794, 0.002262443438914033, 0.0034690799396681765, 0.004977375565610856, 0.007541478129713425, 0.011764705882352941, 0.01825037707390649, 0.025791855203619915, 0.03453996983408748, 0.04343891402714932, 0.05279034690799397, 0.06229260935143289, 0.07179487179487179, 0.08099547511312218, 0.08974358974358976, 0.09849170437405733]
true_std      = []
true_d_std    = []
for i in range(len(num_vertex_mean)):
    true_std.append(   np.std( [std_vertex[jj][i]    for jj in range(len(list_num))] ))
    true_d_std.append( np.std( [std_distance[jj][i]  for jj in range(len(list_num))] ))

up_bound   = []
lo_bound   = []
up_d_bound = []
lo_d_bound = []
for i in range(len(num_vertex_mean)):
    up_bound.append(num_vertex_mean[i]+true_std[i]/2.0)
    lo_bound.append(num_vertex_mean[i]-true_std[i]/2.0)
    up_d_bound.append(mean_distance[i]+true_d_std[i]/2.0)
    lo_d_bound.append(mean_distance[i]-true_d_std[i]/2.0)

##

fig,ax = plt.subplots()

ax.scatter(np.linspace(0.1,1.5,len(num_vertex_mean)),num_vertex_mean,s=10,marker='x',color='black',label='Vortex density')
ax.fill_between(np.linspace(0.1,1.5,len(num_vertex_mean)),up_bound,lo_bound,color='black',alpha=0.1,label='Fluctuations')
ax.set_xlabel(r'Temperature $T$ [$J/k_\mathrm{B}$]',fontsize=20)
ax.set_ylabel(r'Vortex density [a.u.]',fontsize=20)
ax.plot(x_theoretical,y_theoretical,ls='--',color='green',label='Theoretical data')
ax.vlines(0.893,0,max(num_vertex_mean),ls='-.',color='blue',label=r'Theoretical $T_\mathrm{KT}$')
ax.vlines(1.167,0,max(num_vertex_mean),ls='-.',color='red',label=r'Theoretical $T_\mathrm{C}$')
ax.set_xlim(0,1.5)
ax.set_ylim(0,max(num_vertex_mean))
ax.legend(loc='upper left',fontsize=12)

ax2 = ax.twinx()
ax2.scatter(np.linspace(0.1,1.5,len(num_vertex_mean)),mean_distance,s=10,marker='x',color='orange',label='Vortex distance')
ax2.fill_between(np.linspace(0.1,1.5,len(num_vertex_mean)),up_d_bound,lo_d_bound,color='red',alpha=0.1,label='Fluctuations')
ax2.set_xlim(0,1.5)
ax2.set_ylim(0,max(mean_distance))
ax2.set_xlabel(r'Temperature $T$ [$J/k_\mathrm{B}$]',fontsize=20)
ax2.set_ylabel(r'Mean distance [a.u.]',fontsize=20)

ax2.legend(loc='center left',fontsize=12)

plt.gcf()
plt.tight_layout()
plt.savefig('Vortex_data.png',dpi=300)
plt.clf()
