import numpy as np
from   joblib import Parallel, delayed
import multiprocessing
from   multiprocessing import Pool
num_cores = multiprocessing.cpu_count()
from   tqdm import trange
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
import time
from   scipy.optimize import curve_fit

#start_time = time.time()

def magnetic_susceptibility_scaling_law(x, eta,r,a):
    array = []
    b = 0
    C = 0
    for i in x:
        array.append( a*i**(-eta)*(np.log(i)+C)**(-2*r)+b )
    return array
'''
def magnetic_susceptibility_scaling_law_th(x,a,b):
    array = []
    eta   = 1/4.0
    r = -1/16.0
    for i in x:
        array.append( a*i**(-eta)*(np.log(i))**(-2*r)+b )
    return array
'''
mag_susc               = []
value_critical_temp_cv = []
data_from_data         = []
eta_critical           = []
rfit_critical          = []
temp_ll_critical       = []
pcov_eta_critical      = []
pcov_rfit_critical     = []
L_list_critical        = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40])
for j in (np.linspace(0,30,31)):
    mag_susc    = []
    for i in L_list_critical:
        array = np.loadtxt('Temperature_KT_transition_data/transition_region/L'+str(i)+'_N31_transition_region.data')
        T     = array[:, 0]
        E     = array[:, 1]
        SpcH  = array[:, 2]
        M     = array[:,3]
        M_sus = array[:,4]
        mag_susc.append(np.array(M_sus[int(j)]))
    xx                 = L_list_critical
    yy                 = mag_susc
    param , param_cov  = curve_fit(magnetic_susceptibility_scaling_law, xx, yy,maxfev = 1000000,p0=[2-0.25,-1/16.0,0])
    pcov_eta_critical  .append(param_cov[0][0])  # error of the fit on eta
    pcov_rfit_critical .append(param_cov[1][1])  # error of the fit on r
    eta_critical       .append(np.abs(param[0])) # value of eta fitted
    rfit_critical      .append(np.abs(param[1])) # value of r fitted
    temp_ll_critical   .append(T[int(j)])        # considered temperature

error_eta_crit        = [np.abs(i-(1/4.0)) for i in eta_critical]
error_r_crit          = [np.abs(i-1/16.0) for i in rfit_critical]
mean_error_crit       = [(error_eta_crit[i]+error_r_crit[i])/2.0 for i in range(len(error_eta_crit))]
temp_ll_critical_crit = temp_ll_critical

##################

mag_susc               = []
value_critical_temp_cv = []
data_from_data         = []
eta_critical           = []
rfit_critical          = []
temp_ll_critical       = []
pcov_eta_critical      = []
pcov_rfit_critical     = []

L_list_critical = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
for j in (np.linspace(0,19,20)):
    mag_susc    = []
    for i in L_list_critical :
        array = np.loadtxt('Temperature_KT_transition_data/global_region/L'+str(i)+'_N20_global_region.data')
        T     = array[:, 0]
        E     = array[:, 1]
        SpcH  = array[:, 2]
        M     = array[:,3]
        M_sus = array[:,4]
        mag_susc.append(np.array(M_sus[int(j)]))
    xx               = L_list_critical
    yy               = mag_susc
    param, param_cov = curve_fit(magnetic_susceptibility_scaling_law, xx, yy,maxfev = 1000000,p0=[2-0.25,-1/16.0,0])
    pcov_eta_critical .append(param_cov[0][0])
    pcov_rfit_critical.append(param_cov[1][1])
    eta_critical      .append(np.abs(param[0]))
    rfit_critical     .append(np.abs(param[1]))
    temp_ll_critical  .append(T[int(j)])

fig, (ax1) = plt.subplots(1)
#fig.set_figheight(8)
#fig.set_figwidth(12)
error_eta  = [np.abs(i-(1/4.0)) for i in eta_critical]
error_r    = [np.abs(i-1/16.0) for i in rfit_critical]
mean_error = [(error_eta[i]+error_r[i])/2.0 for i in range(len(error_eta))]

ax1.scatter(temp_ll_critical ,mean_error  ,color='red',marker='v',label='mean error')
ax1.scatter(temp_ll_critical ,error_eta  ,color='black',marker='+',label='$|\eta|$')
ax1.scatter(temp_ll_critical ,error_r ,color='green',marker='x',label='$|r|$')

ax1.scatter(temp_ll_critical_crit ,mean_error_crit  ,color='red',marker='v')
ax1.scatter(temp_ll_critical_crit ,error_eta_crit  ,color='black',marker='+')
ax1.scatter(temp_ll_critical_crit ,error_r_crit ,color='green',marker='x')


ax1.vlines(0.89295,0,max(max(error_eta),max(error_r)),ls='--',color='blue',label=r'Theoretical $T_{KT}=$%1.3f'%0.89295)
tuple_temp = np.where(np.array(mean_error_crit)==min(np.array(mean_error_crit)))
res = '.'.join(str(ele) for ele in tuple_temp)
res = int(res[1:-1])
ax1.vlines(temp_ll_critical_crit[res]  ,0,max(max(error_eta),max(error_r)),ls='--',color='red',label=r'Numerical $T_{KT}=$%1.3f'%temp_ll_critical_crit[res])

ax1.set_xlim(min(temp_ll_critical),max(temp_ll_critical))
ax1.set_ylim(min( min(error_eta_crit),min(error_r_crit))  ,max(max(error_eta),max(error_r)))

plt.yscale('log')
plt.grid()
plt.legend(loc='lower right', fontsize=12)
plt.xlabel('Temperature $T$ [$J/k_\mathrm{B}$]', fontsize=20)
plt.gcf()
plt.tight_layout()
plt.savefig('Temperature_KT_transition.png',dpi=300)
plt.clf()
