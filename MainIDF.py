# -*- coding: utf-8 -*-
"""
Created on Mon Oct 05 09:54:42 2020

@author: Andres
"""
import os
import numpy as np
import pandas as pd
from Modules.IDF_Func import Wilches
from Modules.IDF_Func import Pulgarin
from Modules.IDF_Func import VargasDiazGranados
from Modules.IDF_Func import IDEAM
from Modules.Utils import Listador
from Modules import Read

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors

################################   INPUT IDF   #################################

Path_series = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Datos/Series/'))
Path_IDF    = os.path.abspath(os.path.join(os.path.dirname(__file__), 'IDF'))

def MaxAnual(Esta, Path_series):
    """
    Read estation to extract the anual max value
    """
    Dat = Read.EstacionCSV_np(Esta, Esta.split('.csv')[0],Path_series)
    Max = Dat.groupby(lambda y : y.year).max()

    return Max[~np.isnan(Max.values)].values.ravel()/24

def GraphIDF(Int, duration, frecuency, cmap_name='jet', name='IDF', pdf=True, png=False, PathFigs=Path_IDF,):
    """
    Graph of month diurnal cycles

    INPUTS
    Int       : 2D array with the Intesity [mm/hour] with shape=(len(durations), len(frecuency))
    duration  : 1D array with durations [min]
    frecuency : 1D array with reuturn periods [years]
    cmap_name : color map name
    name      : stringo for save the figure
    Path      : abtolute Path to save files
    """

    # define some random data that emulates your indeded code:
    NCURVES = len(frecuency)
    plt.close('all')
    fig = plt.figure(figsize=(12.8,8))
    ax = fig.add_subplot(111)
    cNorm  = colors.Normalize(vmin=0, vmax=NCURVES)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap(cmap_name))

    lines = []
    for idx in range(NCURVES):
        line = Int[:, idx]
        colorVal  = scalarMap.to_rgba(idx)
        colorText = str(frecuency[idx])+' years'
        retLine,  = ax.plot(duration,line, linewidth=2,
                            color=colorVal,
                            label=colorText)
        lines.append(retLine)
    #added this to get the legend to work
    handles,labels = ax.get_legend_handles_labels()

    # # Shrink current axis by 20%
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width*1.0, box.height])
    ax.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5),
              fancybox=False, shadow=False)

    ax.set_xlabel('Duration [minutes]',)
    ax.set_ylabel('Intensity [mm/hour]')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    if pdf == True:
        plt.savefig(os.path.join(PathFigs, name+'.pdf'), format='pdf', transparent=True)
        if png == True:
            plt.savefig(os.path.join(PathFigs, name+'.png'), transparent=True)
    elif png == True:
        plt.savefig(os.path.join(PathFigs, name+'.png'), transparent=True)
    else:
        print("Graph not saved. To save it at least one of png or pdf parameters must be True.")


Estaciones = Listador(Path_series, final='.csv')

# Return periods
Tr = np.array([2.33, 5, 10, 25, 50, 100, 200, 500, 1000])
theta = -0.82
#
# for i in range(len(Estaciones)):
#     Nombre = Estaciones[i].split('.csv')[0]
#
#     data = MaxAnual(Estaciones[i], Path_series)
#     dP, Idq = Pulgarin(data, Tr, theta)
#     GraphIDF(Idq, dP, Tr, cmap_name='jet', name=Nombre+'IDF', pdf=True, png=False, PathFigs=Path_IDF,)
#     IDF = pd.DataFrame(Idq, index=dP, columns=Tr)
#     IDF.to_csv(os.path.join(Path_IDF,Estaciones[i]))
#

# Compare Wilches with VargasDiazGranados
Tr = np.array([2,3,5,10,25,50,100])

Nombre = 'CARMEN DE BOLIVAR [29015020]'
# IDEAM params
C1 = [6798.308,9998.576,14882.323,23468.705,39184.485,55085.160,75025.218]
X0 = [27.895,32.735,37.828,43.764,50.544,55.091,59.234]
C2 = [1.112,1.159,1.207,1.263,1.325,1.366,1.403]

data = MaxAnual(Nombre+'.csv', Path_series)
dP, Idq = Pulgarin(data, Tr, theta)
dV, IdV = VargasDiazGranados(data, Tr, Region=2)
dI, Ida = IDEAM(Tr, X0,C1,C2)
DiP = Ida-Idq
DiV = Ida-IdV
GraphIDF(Idq, dP, Tr, cmap_name='jet', name=Nombre+'IDF_Pulgarin',   pdf=True, png=False, PathFigs=Path_IDF,)
GraphIDF(IdV, dV, Tr, cmap_name='jet', name=Nombre+'IDF_Vargas',     pdf=True, png=False, PathFigs=Path_IDF,)
GraphIDF(Ida, dI, Tr, cmap_name='jet', name=Nombre+'IDF_IDEAM',      pdf=True, png=False, PathFigs=Path_IDF,)
GraphIDF(DiV, dV, Tr, cmap_name='jet', name=Nombre+'IDF_DifVargas',  pdf=True, png=False, PathFigs=Path_IDF,)
GraphIDF(DiP, dV, Tr, cmap_name='jet', name=Nombre+'IDF_DifPulgarin',pdf=True, png=False, PathFigs=Path_IDF,)
IDF_P = pd.DataFrame(Idq, index=dP, columns=Tr)
IDF_V = pd.DataFrame(IdV, index=dV, columns=Tr)
IDF_A = pd.DataFrame(Ida, index=dI, columns=Tr)
IDF_p = pd.DataFrame(DiP, index=dP, columns=Tr)
IDF_v = pd.DataFrame(DiV, index=dV, columns=Tr)
IDF_P.to_csv(os.path.join(Path_IDF,Nombre+'_Purlgarin.csv'))
IDF_V.to_csv(os.path.join(Path_IDF,Nombre+'_Vargas.csv'))
IDF_A.to_csv(os.path.join(Path_IDF,Nombre+'_IDEAM.csv'))
IDF_p.to_csv(os.path.join(Path_IDF,Nombre+'_DifVargas.csv'))
IDF_v.to_csv(os.path.join(Path_IDF,Nombre+'_DifPulgarin.csv'))


cmap_name = 'jet'
pdf = True
png = False
PathFigs = Path_IDF
name=Nombre+'IDF_all'
# define some random data that emulates your indeded code:
NCURVES = len(Tr)
plt.close('all')
fig = plt.figure(figsize=(12.8,8))
ax = fig.add_subplot(111)
cNorm  = colors.Normalize(vmin=0, vmax=NCURVES)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap(cmap_name))

lines = []
for idx in range(NCURVES):
    line = Ida[:, idx]
    colorVal  = scalarMap.to_rgba(idx)
    colorText = f'IDEAM    {Tr[idx]} years'
    retLine,  = ax.plot(dI,line, linewidth=2,
                        color=colorVal,
                        label=colorText)
    lines.append(retLine)

    ax.scatter(dV, IdV[:, idx], marker='*', s=10 ,c=colorVal, label=f'Pulgarin {Tr[idx]} years')
    ax.scatter(dP, Idq[:, idx], marker='o', s=10 ,c=colorVal, label=f'Vargas   {Tr[idx]} years')
#added this to get the legend to work
handles,labels = ax.get_legend_handles_labels()

# # Shrink current axis by 20%
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width*1.0, box.height])
ax.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5),
          fancybox=False, shadow=False)

ax.set_xlabel('Duration [minutes]',)
ax.set_ylabel('Intensity [mm/hour]')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
if pdf == True:
    plt.savefig(os.path.join(PathFigs, name+'.pdf'), format='pdf', transparent=True)
    if png == True:
        plt.savefig(os.path.join(PathFigs, name+'.png'), transparent=True)
elif png == True:
    plt.savefig(os.path.join(PathFigs, name+'.png'), transparent=True)
else:
    print("Graph not saved. To save it at least one of png or pdf parameters must be True.")


# # Load annual 24-h maximum rainfall series
# data = np.genfromtxt('BUENAVISTA - AUT [25020520].txt', delimiter = '	')
# data = data[:,1]
# data = data[~np.isnan(data)]/24.





# # Export individual figures
# Namefig = None
# #######################   Ejecutar Wilches IDF (2001)   ########################
# # Scaling exponents
# theta1 = -0.85              # 105 < d < 1440
# theta2 = -0.64              #  45 < d < 105
#
# dW, IdqA, IdqB = Wilches(data, Tr, theta1, theta2, Namefig)
#
# ####################   Ejecutar Pulgarin & Poveda IDF (2008)   #################
#
# theta = -0.82
# dP, Idq = Pulgarin(data, Tr, theta, Namefig)


# import pylab as plt
# import matplotlib
# from matplotlib import gridspec
# matplotlib.rc('text', usetex = False)
# #font = {'family':'serif', 'serif': ['computer modern roman']}
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# #matplotlib.rcParams['font.family'] = 'STIXGeneral'
# font = {'family':'serif', 'serif': ['Times New Roman']}
# plt.rc('font', **font)
# matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
#
# fontsize = 15
# fig = plt.figure(figsize=(9, 5))
# gs = gridspec.GridSpec(1, 1)
# gs.update(left = 0.075, right = 0.955, top = 0.965, bottom = 0.1, hspace=0.06, wspace = 0.04)
#
# ax0 = plt.subplot(gs[0])
# for i in range(len(Tr)):
#     plt.plot(dW, IdqB[:,i], color = 'b', lw = 2.2)
#     plt.plot(dW, IdqA[:,i], color = 'r', ls = 'dashed', lw = 2.2)
#     plt.plot(dP, Idq[:,i], color = 'k', ls = 'dashed', lw = 1.5)
# plt.ylabel('Intensidad [mm/h]', fontsize = fontsize, labelpad = 0.)
# plt.xlabel(u'Duración [min]', fontsize = fontsize, labelpad = 0.)
# plt.xlim([0,350])
# plt.tick_params(axis='x', which='both', labelsize=fontsize)
# plt.tick_params(axis='y', which='both', labelsize=fontsize)
# ax0.tick_params('both', length=5, width = 1.5, which='major')
# for axis in ['top','bottom','left','right']:
#      ax0.spines[axis].set_linewidth(1.5)
#
# plt.plot(np.NaN, np.NaN, color = 'r', ls = 'dashed', lw = 1.5, label = 'Wilches (2001) - Modelo A')
# plt.plot(np.NaN, np.NaN, color = 'b', lw = 2.2, label = 'Wilches (2001) - Modelo B')
# plt.plot(np.NaN, np.NaN, color = 'k', lw = 1.5, label = 'Pulgarin & Poveda (2008)')
#
# ax0.legend(loc = 'center', bbox_to_anchor=(0.7, 0.8), ncol = 1, columnspacing=0.5,
#            handletextpad=0.1, numpoints = 1, handlelength=1.5, fontsize = fontsize,
#            scatterpoints = 1, frameon = False)
# plt.savefig(NameFig+'Curvas_IDF.png', dpi = 400)



#import scipy.stats as stats
#y = np.linspace(data.min(), data.max(), 100)
#gumbel_args = stats.gumbel_r.fit(data)
#gumbel_model_pdf=stats.gumbel_r.pdf(y,*gumbel_args)
#gumbel_model_cdf=stats.gumbel_r.cdf(y,*gumbel_args)
#gumbel_model_cdf_PWM=stats.gumbel_r.cdf(y,*(mu, alpha))
#
##mud = gumbel_args[0]
##sigmad = gumbel_args[1]
#
#
#
#import pylab as plt
#import matplotlib
#
#matplotlib.rc('text', usetex = False)
#matplotlib.rcParams['mathtext.fontset'] = 'stix'
#font = {'family':'serif', 'serif': ['Times New Roman']}
#plt.rc('font', **font)
#matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
#
#figure = plt.figure(figsize=(9,5))
#ax2 = figure.add_subplot(111)
#plt.hist(data,bins=30,normed=True,alpha=0.5,cumulative=True)
#ax2.set_ylabel('CDF',size=17)
#ax2.set_xlabel('Caudal [m3/s]',size=17)
#plt.plot(y,gumbel_model_cdf,linewidth=3,color='r',label='Gumbel_ML')
#plt.plot(y,gumbel_model_cdf_PWM,linewidth=3,color='b',label='Gumbel_PWM')
#plt.grid()
#plt.tick_params(axis='x', which='both', labelsize=17, direction = 'in')
#plt.tick_params(axis='y', which='both', labelsize=17, direction = 'in')
#ax2.tick_params('both', length=8, width = 1.5, which='major')
#ax2.tick_params('both', length=5, width = 1.2, which='minor')
#for axis in ['top','bottom','left','right']:
#     ax2.spines[axis].set_linewidth(1.8)









#######################   Fit Gumbel distribution   #########################
#import scipy.stats as stats
#gumbel_args = stats.gumbel_r.fit(data)
#mud = gumbel_args[0]
#sigmad = gumbel_args[1]
#
#y = -np.log(-np.log(1. - 1./Tr))
#
#################################   Duration   ##############################
#
#dlim1 = 60
#dlim2 = 1440
#d = np.arange(dlim1, dlim2, 5)
#
## IDF curves
#Idq = np.zeros((len(d), len(Tr)))
#
#for i in range(len(Tr)):
#    Idq[:,i] = (mud + sigmad*y[i])*(d/1440.)**theta


## L-moments
##import lmoments3 as lmom
#import lmoments3
#from lmoments3 import distr
#params = distr.nor.lmom_fit(data)

#import scipy.stats as stats
#x = np.linspace(data.min(), data.max(), 100)
#gumbel_args=stats.gumbel_r.fit(data)
#gumbel_model_pdf=stats.gumbel_r.pdf(x,*gumbel_args)
#gumbel_model_cdf=stats.gumbel_r.cdf(x,*gumbel_args)
#
#norm_args=stats.norm.fit(data)
#norm_model_pdf=stats.norm.pdf(x,*norm_args)
#norm_model_cdf=stats.norm.cdf(x,*norm_args)
