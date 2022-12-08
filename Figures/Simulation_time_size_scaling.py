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

total_size = [2 ,3 ,4 ,5  ,6  ,  7,  8, 9,11  ,12 ,13  ,14  ,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
total_time = [22,25,42,59,79,105,137,172,248,299,357,411,465,529,593,650,741,835,892,931,1035,1133,1227,1328,1427,1634,1651,1759]

def func(x, a, b, c):
    array = []
    for i in x:
        array.append(a * np.exp(b * i) + c)
    return array

xx = np.array(total_size)
yy = np.array(total_time)
param, param_cov = curve_fit(func, xx, yy,maxfev = 100000)
#plt.yscale('log')
to_plot = np.linspace(min(total_size),max(total_size),100)
plt.plot(to_plot,func(to_plot,*param),color='blue')
plt.scatter(total_size,total_time,color='black',marker='v')
plt.grid()
plt.text(12.5,75,'$\propto %1.2f \exp(%1.5f \\times L)$'%(param[0],param[1]),fontsize=20,color='blue')
plt.xlabel(r'Size $L$ of the system',fontsize=20)
plt.ylabel(r'Computation time [s]',fontsize=20)
plt.xlim(min(total_size),max(total_size))
plt.ylim(min(total_time),max(total_time))
plt.savefig('Simulation_time_size_scaling.png',dpi=300)
plt.clf()
