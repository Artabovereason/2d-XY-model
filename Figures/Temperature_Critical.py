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

def func(x, a, b, c):
    array = []
    for i in x:
        array.append(a * np.exp(-b * i) + c)
    return array
fig, (ax1) = plt.subplots(1)
value_critical_temp    = []
value_critical_temp_cv = []
data_from_data         = []
L_list = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
for i in L_list:
    array = np.loadtxt('Temperature_Critical_data/L'+str(i)+'_N50.data')
    T     = array[:, 0]
    E     = array[:, 1]
    SpcH  = array[:, 2]
    M     = array[:,3]
    M_sus = array[:,4]
    value_critical_temp.append(np.array(T[np.where(M_sus==max(M_sus))]))
    value_critical_temp_cv.append(T[np.where(SpcH==max(SpcH))])

ax1.scatter(L_list,value_critical_temp,marker='x',color='red',label='from $\chi$')
ax1.scatter(L_list,value_critical_temp_cv,marker='x',color='black',label='from $C_v$')
ax1.set_xlabel(r'Size $L$ of the system', fontsize=20)
ax1.set_ylabel(r'Temperature max [$J/k_\mathrm{B}$]', fontsize=20)
ax1.hlines(1.167,min(L_list),max(L_list),ls='--',color='blue',label=r'Theoretical $T_c$')

ax1.grid()
xx               = np.array(L_list)
yy               = [i[0] for i in value_critical_temp]
param, param_cov = curve_fit(func, xx, yy,maxfev = 100000)
to_plot          = np.linspace(min(L_list),max(L_list),100)
ax1.plot(to_plot,func(to_plot,*param),label=('$T_c=$%1.5f$\pm$%1.5f'%(param[2],param_cov[2][2])),ls='--',color='pink')
yy               = [i[0] for i in value_critical_temp_cv]
param, param_cov = curve_fit(func, xx, yy,maxfev = 100000)
to_plot          = np.linspace(min(L_list),max(L_list),100)
ax1.plot(to_plot,func(to_plot,*param),label=('$T_c=$%1.5f$\pm$%1.5f'%(param[2],param_cov[2][2])),ls='--',color='grey')

ax1.set_xlim(min(L_list),max(L_list))
ax1.legend(fontsize=12)
fig.tight_layout()
plt.savefig('Temperature_Critical.png',dpi=300)
plt.clf()
