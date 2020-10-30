# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 18:00:30 2020

@author: Andres
"""

import pandas as pd
import numpy as np
import scipy.stats as st
import pylab as plt

############################   L-moments functions   ###########################

from Lmoments import Lmom, Lmom1
from Lmoments import L_ratiodiagram
from Lmoments import Lnorm, Lexp, Lgumbel, LGPA, LGEV, LGLO, LLOG3, LP3
from Lmoments import EXPcdf, GUMcdf, NORMcdf, GPAcdf, GEVcdf, GLOcdf, LLOG3cdf, LP3cdf
from Lmoments import EXPq, GUMq, NORMq, GPAq, GEVq, GLOq, LLOG3q, LP3q

dist_names = ['norm', 'expon', 'gumbel_r', 'genpareto',
              'genextreme', 'genlogistic', 'lognorm', 'pearson3']
names = ['Norm.', 'Exp.', 'Gumbel', 'GPA', 'GEV',  'GLO',  'LOGN3', 'P3']
colors = ['k'   ,   'g' ,   'gold',   'b',   'r', 'lime', 'orange',  'm']

Ldist = [Lnorm, Lexp, Lgumbel, LGPA, LGEV, LGLO, LLOG3, LP3]
LCDF = [NORMcdf, EXPcdf, GUMcdf, GPAcdf, GEVcdf, GLOcdf, LLOG3cdf, LP3cdf]
Lq = [NORMq, EXPq, GUMq, GPAq, GEVq, GLOq, LLOG3q, LP3q]

################################   INPUT   #####################################

Est = 'SanMarcos'
data = pd.read_csv(Est + '.csv', index_col = 0, header = 0, low_memory=False)
#dates = data.index
#times = [plt.date2num(dt.datetime.strptime(d,'%Y-%m-%d')) for d in dates]
#dates = [dt.datetime.strptime(d,'%Y-%m-%d') for d in dates]
try:
    serie = np.asarray(data.iloc[:,1])
except:
    serie = np.asarray(data.iloc[:,0])
serie = serie[~np.isnan(serie)]*1.079

lmom, lmomA = Lmom(serie)
lmom1, t3, t4 = Lmom1(serie)

L_ratiodiagram(lmom, Est)

################################   FIGURE   ####################################

import matplotlib
matplotlib.rc('text', usetex = False)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
#matplotlib.rcParams['font.family'] = 'STIXGeneral'
font = {'family':'serif', 'serif': ['Times New Roman']}
plt.rc('font', **font)
matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']

# Start figure and subfigures
from matplotlib import gridspec
fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(1, 2)
gs.update(left = 0.05, right = 0.98, top = 0.95, bottom = 0.11, hspace=0.12, wspace = 0.04)

# Plot parameters
fontsize = 17

#################################   L-Moments   ################################

paramsLM = np.zeros((len(dist_names), 3))*np.NaN
ks_LM = np.zeros(len(dist_names))
pvalue_LM = np.zeros(len(dist_names))

ax1 = plt.subplot(gs[1])
plt.hist(serie, bins = 30, density = True, alpha = 0.5, cumulative = True)
# ax1.set_ylabel('CDF', fontsize = fontsize, labelpad = 0)
# ax1.set_xlabel('Caudal [m$^3$/s]', fontsize = fontsize, labelpad = 0)
plt.title('L-momentos', fontsize = fontsize)


x = np.linspace(np.min(serie), np.max(serie), 100)

for i in range(len(dist_names)):
    dist_name = dist_names[i]
    dist = getattr(st, dist_name)

    if i <= 2:
        Lfunc = Ldist[i]
        loc, scale = Lfunc(lmom)
        cdfi = LCDF[i]
        cdf_fit = cdfi(x, loc, scale)
        cdf = lambda x: cdfi(x, loc, scale)
        ks_LM[i] = st.kstest(serie, cdf)[0]
        pvalue_LM[i] = st.kstest(serie, cdf)[1]
        paramsLM[i][:] = np.array([loc, scale, np.NaN])
    else:
        Lfunc = Ldist[i]
        loc, scale, shape = Lfunc(lmom)
        cdfi = LCDF[i]
        cdf_fit = cdfi(x, loc, scale, shape)
        cdf = lambda x: cdfi(x, loc, scale, shape)
        ks_LM[i] = st.kstest(serie, cdf)[0]
        pvalue_LM[i] = st.kstest(serie, cdf)[1]
        paramsLM[i][:] = np.array([loc, scale, shape])

    plt.plot(x, cdf_fit, lw = 2, color = colors[i], label = names[i])

plt.xlim([np.min(serie), np.max(serie)])
plt.ylim([0,1])
plt.tick_params(axis='x', which='both', labelsize=fontsize, direction = 'in')
plt.tick_params(axis='y', which='both', labelsize=fontsize, direction = 'in')
ax1.tick_params('both', length=5, width = 1.5, which='major')
plt.setp(ax1.get_yticklabels(), visible=False)
for axis in ['top','bottom','left','right']:
     ax1.spines[axis].set_linewidth(1.8)
#plt.legend(loc='best')

###########################   Maximum Likelihood   #############################

paramsMEL = np.zeros((len(dist_names), 3))
ks_MEL = np.zeros(len(dist_names))
pvalue_MEL = np.zeros(len(dist_names))

ax0 = plt.subplot(gs[0], sharex = ax1)
plt.hist(serie, bins = 30, density = True, alpha = 0.5, cumulative = True, label = 'Mediciones')
ax0.set_ylabel('CDF', fontsize = fontsize, labelpad = 0)
ax0.set_xlabel('Caudal [m$^3$/s]', fontsize = fontsize, labelpad = 0, x = 1.05)

plt.title(u'Máxima verosimilitud', fontsize = fontsize)
x = np.linspace(np.min(serie), np.max(serie), 100)
for i in range(len(dist_names)):
    dist_name = dist_names[i]
    dist = getattr(st, dist_name)
    param = dist.fit(serie, loc = paramsLM[i][0], scale = paramsLM[i][1])
    if i <= 2:
        loc, scale = param
        paramsMEL[i][:2] = np.array([loc, scale])
    else:
        shape, loc, scale = param
        paramsMEL[i][:] = np.array([loc, scale, shape])
    cdf_fit = dist.cdf(x, *param[:-2], loc = param[-2], scale = param[-1])
    plt.plot(x, cdf_fit, lw = 2, color = colors[i], label = names[i])

    # K-S test
    ks_MEL[i] = st.kstest(serie, dist_name, param)[0]
    pvalue_MEL[i] = st.kstest(serie, dist_name, param)[1]

plt.xlim([np.min(serie), np.max(serie)])
plt.ylim([0,1])
plt.tick_params(axis='x', which='both', labelsize=fontsize, direction = 'in')
plt.tick_params(axis='y', which='both', labelsize=fontsize, direction = 'in')
ax0.tick_params('both', length=5, width = 1.5, which='major')
# plt.setp(ax0.get_xticklabels(), visible=False)
for axis in ['top','bottom','left','right']:
     ax0.spines[axis].set_linewidth(1.8)
plt.legend(loc='best', fontsize = fontsize-1)

plt.savefig('Ajustes_' + Est + 'A.png', dpi = 400)

df = pd.DataFrame({'ks_MEL': ks_MEL, 'ks_LM': ks_LM}, index = names)
df.to_excel('Smirnov-Kolmogorov'+Est+'.xls')


################################################################################
##############################   Best distribution   ###########################
################################################################################

best_LM  = np.where(pvalue_LM==np.nanmax(pvalue_LM))[0][0]
best_MEL = np.where(pvalue_LM==np.nanmax(pvalue_LM))[0][0]
if best_LM == best_MEL:
    index = best_LM
else:
    print(f"Choose between index {best_LM} and {best_MEL}")

# twoparams = 1
locMEL, scaleMEL, shapeMEL = paramsMEL[index,:]
locLM,  scaleLM,  shapeLM  = paramsLM [index,:]
distMEL = getattr(st, dist_names[index])
distLM  = LCDF[index]

try:
    cdf_fitMEL = distMEL.cdf(x, locMEL, scaleMEL)
    cdf_fitLM  = distLM(x, locLM, scaleLM)
except:
    cdf_fitMEL = distMEL.cdf(x, shapeMEL, locMEL, scaleMEL)
    cdf_fitLM  = distLM(x, locLM, scaleLM, shapeLM)

fig = plt.figure(figsize=(7, 5))
gs = gridspec.GridSpec(1, 1)
gs.update(left = 0.10, right = 0.96, top = 0.98, bottom = 0.13, hspace=0.12, wspace = 0.04)

# Plot parameters
fontsize = 15
ax0 = plt.subplot(gs[0])
plt.hist(serie, bins = 30, density = True, alpha = 0.5, cumulative = True, label = 'Mediciones')
plt.plot(x, cdf_fitMEL, lw = 2, color = 'b', label = u'Máxima verosimilitud')
plt.plot(x, cdf_fitLM, lw = 2, color = 'r', label = u'L-momentos')
plt.xlim([np.min(serie), np.max(serie)])
plt.ylim([0,1])
plt.ylabel('CDF', fontsize = fontsize, labelpad = 0)
# plt.xlabel('Caudal [m$^3$/s]', fontsize = fontsize, labelpad = 0)
plt.xlabel('Nivel [m.s.n.m]', fontsize = fontsize, labelpad = 0)
plt.tick_params(axis='x', which='both', labelsize=fontsize, direction = 'in')
plt.tick_params(axis='y', which='both', labelsize=fontsize, direction = 'in')
ax0.tick_params('both', length=5, width = 1.5, which='major')
#plt.setp(ax0.get_xticklabels(), visible=False)
for axis in ['top','bottom','left','right']:
     ax0.spines[axis].set_linewidth(1.8)
plt.legend(loc='lower right', fontsize = fontsize-1)

plt.savefig('Best_fit_' + Est + 'B.png', dpi = 400)


################################################################################
######################   Quantiles by best distribution   ######################
################################################################################

best_LM  = np.where(pvalue_LM==np.nanmax(pvalue_LM))[0][0]
best_MEL = np.where(pvalue_LM==np.nanmax(pvalue_LM))[0][0]
if best_LM == best_MEL:
    index = best_LM
else:
    print(f"Choose between index {best_LM} and {best_MEL}")

# twoparams = 1
locMEL, scaleMEL, shapeMEL = paramsMEL[index,:]
locLM, scaleLM, shapeLM = paramsLM[index,:]
distMEL = getattr(st, dist_names[index])
distLM  = Lq[index]
Tr = np.array([2.33, 5, 10, 25, 50, 100, 200, 500, 1000])
q = 1. - 1./Tr

try:
    # Quantiles MEL
    quant_MEL = distMEL.ppf(q, loc = locMEL, scale = scaleMEL)
    # Quantiles LM
    quant_LM = distLM(q, locLM, scaleLM)
except:
    # Quantiles MEL
    quant_MEL = distMEL.ppf(q, shapeMEL, loc = locMEL, scale = scaleMEL)
    # Quantiles LM
    quant_LM = distLM(q, locLM, scaleLM, shapeLM)


####################   FIGURE

fig = plt.figure(figsize=(7, 5))
gs = gridspec.GridSpec(1, 1)
gs.update(left = 0.11, right = 0.96, top = 0.97, bottom = 0.13, hspace=0.12, wspace = 0.04)

# Plot parameters
fontsize = 15
ymin = 0.0
ymax = 0.5
xmin = 0.0
xmax = 0.5

# Trpadec = np.array([2, 5, 10, 25, 50, 100, 500, 1000])
# Qpadec = np.array([2465, 3280, 3820, 4508, 5018, 5524, 6695, 7198])

ax0 = plt.subplot(gs[0])
plt.plot(Tr, quant_MEL, 'o-', mfc = 'r', mec = 'k', mew = 1.2, color = 'r',
         lw = 2.5, label = u'Máxima verosimilitud', clip_on = False, zorder = 4)
#plt.plot(Tr, quant_MEL+300., 'o-', mfc = 'r', mec = 'k', mew = 1.2, color = 'r',
#         lw = 2.5, label = u'Máxima verosimilitud + 2.33 Pte Comuna', clip_on = False, zorder = 4)
plt.plot(Tr, quant_LM, 's-', mfc = 'b', mec = 'k', mew = 1.2, color = 'b',
         lw = 1.5, label = 'L-momentos', clip_on = False, zorder = 4)
#plt.plot(Tr, quant_LM+300., 's-', mfc = 'b', mec = 'k', mew = 1.2, color = 'b',
#         lw = 1.5, label = 'L-momentos + 2.33 Pte Comuna', clip_on = False, zorder = 4)
# plt.plot(Trpadec, Qpadec, 'v-', mfc = 'g', mec = 'k', mew = 1.2, color = 'g',
#          lw = 1.5, label = 'PADEC (2016)', clip_on = False, zorder = 5)
#plt.ylim([0, 3500])
plt.xlabel(u'Periodo de retorno [años]', fontsize = fontsize, labelpad = 0)
# plt.ylabel('Caudal [m$^3$/s]', fontsize = fontsize, labelpad = 0)
plt.ylabel('Nivel [m.s.n.m]', fontsize = fontsize, labelpad = 0)
plt.tick_params(axis='x', which='both', labelsize=fontsize, direction = 'in')
plt.tick_params(axis='y', which='both', labelsize=fontsize, direction = 'in')
ax0.tick_params('both', length=7, width = 1.5, which='major')
ax0.tick_params('both', length=4, width = 1.3, which='minor')
plt.xscale('log')

#plt.setp(ax0.get_xticklabels(), visible=False)
for axis in ['top','bottom','left','right']:
     ax0.spines[axis].set_linewidth(1.8)
plt.legend(loc='best', numpoints = 1, fontsize = fontsize-1)

plt.savefig('Cuantiles_' + Est + 'B.png', dpi = 400)


cuantiles = pd.DataFrame( np.array([quant_LM,quant_MEL]).T, index = Tr, columns=['LM', 'MEL'])
cuantiles.to_csv('Cuantiles_' + Est + 'B.csv')
