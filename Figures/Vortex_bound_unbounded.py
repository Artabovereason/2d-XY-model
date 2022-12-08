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

start_time = time.time()

###
num_vertex = []
L          = 16
num        = 11
T = np.linspace(0.1,1.5,101)
def job(b):

    XY                 = np.loadtxt('vortex_data/config'+str(num)+'/config_L'+str(L)+'_N101_T_'+str(T[b])+'.data')
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
            coproj  = np.cos(XY[i][j])+np.cos(XY[i+1][j])+np.cos(XY[i][j+1])+np.cos(XY[i+1][j+1])
            sinproj = np.sin(XY[i][j])+np.sin(XY[i+1][j])+np.sin(XY[i][j+1])+np.sin(XY[i+1][j+1])
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

    plt.scatter(x_pos_vortex    ,y_pos_vortex    ,s=1200,facecolors='none',edgecolors='blue')
    plt.scatter(x_pos_antivortex,y_pos_antivortex,s=1200,facecolors='none',edgecolors='red')

    X_new = np.cos(XY)
    Y_new = np.sin(XY)

    plt.quiver(X,Y,X_new,Y_new,XY,pivot='mid',cmap=plt.cm.binary,clim=[-10000000000,5])
    plt.axis('equal')
    plt.axis('off')
    #cbar = plt.colorbar(ticks=[-3.14, 0, 3.14])
    #cbar.ax.set_yticklabels([r"$+\pi$", r"0", r"$-\pi$"])
    #plt.matshow(XY,cmap='cool')
    plt.axis('off')
    plt.gcf()
    plt.title(r'Temperature $T=$%1.3f [$J/k_\mathrm{B}$]'%T[b])
    plt.savefig('vortex_data/system_T'+str(T[b])+'.png',dpi=300)
    plt.clf()


run = Parallel(n_jobs=num_cores)(delayed(job)(b) for b in trange(101))
