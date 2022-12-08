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

L_list = range(2,30)

E_mean     = np.zeros(101)
SpcH_mean  = np.zeros(101)
M_mean     = np.zeros(101)
M_sus_mean = np.zeros(101)

for i in L_list:
    array = np.loadtxt('vortex_data/config'+str(i)+'/L16_N101_thermodynamics.data')
    T     = array[:, 0]
    E     = array[:, 1]
    for j in range(len(E)):
        E_mean[j] += E[j]/len(L_list)

    SpcH  = array[:, 2]
    for j in range(len(SpcH)):
        SpcH_mean[j] += SpcH[j]/len(L_list)

    M     = array[:,3]
    for j in range(len(M)):
        M_mean[j] += M[j]/len(L_list)

    M_sus = array[:,4]
    for j in range(len(M_sus)):
        M_sus_mean[j] += M_sus[j]/len(L_list)

plt.xlabel(r'Temperature $T$ [$J/k_\mathrm{B}$]')
plt.ylabel(r'Mean energy $\langle E \rangle$ per site [$J$]')
plt.grid()
plt.scatter(T,E_mean,color='black',marker='x',label=r'$\langle E \rangle$')

# From Bramwell and Holdsworth:
temp_paper        = np.array([0.5,0.9,1.0,1.1,1.2,1.3,1.6])                      # From Ota 1992
mean_energy_paper = np.array([-1.730,-1.441,-1.329,-1.182,-1.041,-0.935,-0.718]) # From Ota 1992
mean_mag2_paper   = np.array([0.7107,0.4426,0.3251,0.1171,0.0453,0.0232,0.0088])
temp_BH=[0.10392156862745094,0.20392156862745092,0.3019607843137254,0.4019607843137255,0.5019607843137254,0.6039215686274508,0.6999999999999997,0.7980392156862743,0.8058823529411763,0.8156862745098037,0.8254901960784313,0.8372549019607842,0.8470588235294116,0.8568627450980391,0.8666666666666665,0.8764705882352939,0.8862745098039215,0.8941176470588235,0.9058823529411764,0.9137254901960783,0.9411764705882353,0.945098039215686,0.9549019607843137,0.9627450980392156,0.9666666666666666,0.980392156862452,0.9901960784313724,0.9980392156862745,1.0039215686274507,1.011764705882353,1.0196078431372546,1.0294117647058822,1.0411764705882351,1.0509803921568628,1.06078431372549,1.072549019607842,1.0823529411764705,1.0901960784313722,1.1019607843137256,1.2,1.2999999999999998,1.3999999999999997,1.4960784313725488]
magnetization_BH=[0.972058823529412,0.9397058823529414,0.9073529411764707,0.8764705882352943,0.8411764705882354,0.8058823529411767,0.7661764705882355,0.7205882352941178,0.7147058823529413,0.7088235294117649,0.7029411764705884,0.6955882352941178,0.6897058823529413,0.6838235294117648,0.6779411764705884,0.6720588235294119,0.6661764705882355,0.6588235294117648,0.6514705882352942,0.6441176470588237,0.6250000000000001,0.6088235294117648,0.6029411764705883,0.5970588235294119,0.5823529411764707,0.5691176470588237,0.5544117647058825,0.5352941176470589,0.5308823529411767,0.5250000000000001,0.5044117647058826,0.4838235294117649,0.4617647058823531,0.4308823529411766,0.4058823529411766,0.38088235294117667,0.361764705882353,0.33235294117647074,0.3058823529411766,0.18676470588235294,0.11911764705882366,0.09705882352941197,0.08088235294117663]

plt.scatter(temp_paper,mean_energy_paper,color='green',marker='v',label='Theoretical data')
plt.xlim(min(min(temp_paper),min(T)),max(max(temp_paper),max(T)))
plt.ylim(min(min(mean_energy_paper),min(E_mean)),max(max(mean_energy_paper),max(E_mean)))

plt.gcf()
plt.vlines(0.89295,-10,10,ls='-.',color='blue',label=r'Theoretical $T_{KT}$')
plt.vlines(1.167,-10,10,ls='-.',color='red',label=r'Theoretical $T_c$')
plt.legend(loc='upper left')
plt.savefig('mean_energy.png',dpi=300)
plt.clf()

#From FIG. 8 of DOI: 10.1103/PhysRevB.88.144104
temp_paper_berg =[0.800204081632653,0.8099999999999999,0.8197959183673469,0.83,0.840204081632653,0.85,0.8602040816326529,0.8699999999999999,0.880204081632653,0.8899999999999999,0.900204081632653,0.9099999999999999,0.920204081632653,0.9299999999999999,0.9402040816326529,0.95,0.9602040816326529,0.97,0.9797959183673468,0.9899999999999999,0.9997959183673468,1.0099999999999998,1.0197959183673468,1.0299999999999998,1.0397959183673469,1.0495918367346937,1.0597959183673469,1.0695918367346937,1.079795918367347,1.0895918367346937,1.099795918367347,1.1095918367346937,1.119795918367347,1.1295918367346938,1.1397959183673467,1.1495918367346938,1.1597959183673467,1.1695918367346938,1.1797959183673468,1.1895918367346936]
SpcH_paper_berg = [0.7880694143167029,0.8010845986984816,0.8140997830802603,0.827114967462039,0.8427331887201736,0.858351409978308,0.8752711496746205,0.8921908893709327,0.910412147505423,0.9286334056399133,0.9494577006507592,0.971583514099783,0.9937093275488069,1.0184381778741867,1.0431670281995662,1.0704989154013016,1.0991323210412147,1.1277657266811278,1.159002169197397,1.1915401301518438,1.2240780911062907,1.2579175704989154,1.294360086767896,1.3281995661605206,1.3594360086767896,1.3919739696312365,1.416702819956616,1.4414316702819956,1.4635574837310195,1.4778741865509761,1.4869848156182213,1.489587852494577,1.4843817787418656,1.4765726681127984,1.4635574837310195,1.4453362255965294,1.4232104121475055,1.3958785249457701,1.3672451193058568,1.3412147505422993]

SpcH_mean = [(max(SpcH_paper_berg)/max(SpcH_mean))*SpcH_mean[i] for i in range(len(SpcH_mean))]
plt.ylim(min(SpcH_mean),max(max(SpcH_paper_berg),max(SpcH_mean)))
plt.xlim(min(T),max(T))

plt.plot(temp_paper_berg,SpcH_paper_berg,color='green',label='Theoretical data')
plt.scatter(T,SpcH_mean,marker='x',color='black',label=r'$C_v$')

plt.xlabel(r'Temperature $T$ [$J/k_\mathrm{B}$]')
plt.ylabel(r'Specific heat $C_v$ [$J^2/k_\mathrm{B}^2$]')
plt.grid()
plt.vlines(0.89295,min(SpcH_mean),max(max(SpcH_paper_berg),max(SpcH_mean)),ls='-.',color='blue',label=r'Theoretical $T_{KT}$')
plt.vlines(1.167,min(SpcH_mean),max(max(SpcH_paper_berg),max(SpcH_mean)),ls='-.',color='red',label=r'Theoretical $T_c$')
plt.legend(loc='upper left')
plt.savefig('Specific_heat.png',dpi=300)

plt.clf()

plt.scatter(T,M_mean,marker='x',color='black',label='$\langle M\\rangle$ for $L=16$')
plt.plot(temp_BH,magnetization_BH,color='green',label='Theoretical data for $L=32$')
#plt.scatter(temp_paper,mean_mag2_paper,marker='v',color='red')
plt.vlines(0.89295,min(min(M_mean),min(magnetization_BH)),max(max(M_mean),max(magnetization_BH)),ls='-.',color='blue',label=r'Theoretical $T_{KT}$')
plt.vlines(1.167,min(min(M_mean),min(magnetization_BH)),max(max(M_mean),max(magnetization_BH)),ls='-.',color='red',label=r'Theoretical $T_c$')
plt.legend(loc='lower left')
plt.ylim(min(min(M_mean),min(magnetization_BH)),max(max(M_mean),max(magnetization_BH)))
plt.xlim(min(min(T),min(temp_BH)),max(max(T),max(temp_BH)))
plt.xlabel(r'Temperature $T$ [$J/k_\mathrm{B}$]')
plt.ylabel(r'Mean magnetisation $\langle M \rangle$ per site [$\mu$]')
plt.grid()
plt.savefig('Mean_magnetisation.png',dpi=300)
plt.clf()

#


plt.scatter(T,M_sus_mean,marker='x',color='black',label=r'$\chi$')
plt.vlines(0.89295,min(M_sus_mean),max(max(M_sus_mean),max(M_sus_mean)),ls='-.',color='blue',label=r'Theoretical $T_{KT}$')
plt.vlines(1.167,min(M_sus_mean),max(max(M_sus_mean),max(M_sus_mean)),ls='-.',color='red',label=r'Theoretical $T_c$')
plt.legend(loc='upper left')
plt.xlabel(r'Temperature $T$ [$J/k_\mathrm{B}$]')
plt.ylabel(r'Magnetic susceptibility $\chi$ [$\mu/k_\mathrm{B}$]')
plt.grid()
plt.ylim(min(min(M_sus_mean),min(M_sus_mean)),max(max(M_sus_mean),max(M_sus_mean)))
plt.xlim(min(min(T),min(T)),max(max(T),max(T)))
plt.savefig('Magnetic_susceptibility.png',dpi=300)
plt.clf()
