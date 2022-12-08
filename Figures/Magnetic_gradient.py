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
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

mag_susc    = []
value_critical_temp_cv = []
data_from_data         = []
eta = []
rfit = []
temp_ll = []
pcov_eta = []
pcov_rfit = []
mag_mag       = []
temp_temp     = []
magsus_magsus = []

L_list = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
for j in [11,12,13,14,15,16,17,18]:
    mag_susc    = []
    for i in L_list:
        array = np.loadtxt('Magnetic_gradient_data/L'+str(i)+'_N50.data')
        T     = array[:, 0]
        E     = array[:, 1]
        SpcH  = array[:, 2]
        M     = array[:,3]
        M_sus = array[:,4]
        if j == 11:
            temp_temp.append(T)
            mag_mag.append(M)
            magsus_magsus.append(M_sus)
        mag_susc.append(np.array(M_sus[j]))
    temp_ll.append(T[j])

cmapp = cm.get_cmap('Reds', len(L_list))

max_one = np.zeros(len(L_list))
for i in range(len(temp_temp)):
    cache_temp   = temp_temp[i]
    cache_magsus = magsus_magsus[i]
    plt.plot(cache_temp,cache_magsus,color=cmapp(L_list[i]/max(L_list)))
    max_one[i]   = (cache_temp[np.where(cache_magsus==max(cache_magsus))])

plt.grid()
plt.xlabel(r'Temperature $T$ [$J/k_\mathrm{B}$]',fontsize=20)
plt.ylabel(r'Magnetic susceptibility $\chi$ per spin site [$\mu/k_\mathrm{B}$]',fontsize=15)
plt.tight_layout()
plt.xlim(min(cache_temp),max(cache_temp))
plt.gcf()
plt.savefig('Magnetic_gradient.png',dpi=300)
plt.clf()
