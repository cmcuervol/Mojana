#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
import datetime as dt

from Modules import Read
from Modules.Utils import Listador
from Modules.FitStats import BestFit, QuantilBestFit

Path_yearly = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Datos/Anuales/'))
Est_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'CleanData'))
Path_out = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Balance'))

Tr = np.array([2.33, 5, 10, 25, 50, 100, 200, 500, 1000])

Magan_d = Read.EstacionCSV_pd('25027680.csv', 'MAGANGUE',    path=Est_path)
Barbo_d = Read.EstacionCSV_pd('25027530.csv', 'BARBOSA',     path=Est_path)
Armen_d = Read.EstacionCSV_pd('25027360.csv', 'ARMENIA',     path=Est_path)
Cruz3_d = Read.EstacionCSV_pd('25027640.csv', 'CRUCES',      path=Est_path)
Monte_d = Read.EstacionCSV_pd('25017010.csv', 'MONTELIBANO', path=Est_path)

Magan_sum = Magan_d.groupby(lambda y : y.year).sum()
Barbo_sum = Barbo_d.groupby(lambda y : y.year).sum()
Armen_sum = Armen_d.groupby(lambda y : y.year).sum()
Cruz3_sum = Cruz3_d.groupby(lambda y : y.year).sum()
Monte_sum = Monte_d.groupby(lambda y : y.year).sum()

Magan_max = Magan_d.groupby(lambda y : y.year).max()
Barbo_max = Barbo_d.groupby(lambda y : y.year).max()
Armen_max = Armen_d.groupby(lambda y : y.year).max()
Cruz3_max = Cruz3_d.groupby(lambda y : y.year).max()
Monte_max = Monte_d.groupby(lambda y : y.year).max()


Magan_LM, Magan_MEL, Magan_dist = QuantilBestFit(Magan_max.dropna().values, Tr=Tr)
Barbo_LM, Barbo_MEL, Barbo_dist = QuantilBestFit(Barbo_max.dropna().values, Tr=Tr)
Armen_LM, Armen_MEL, Armen_dist = QuantilBestFit(Armen_max.dropna().values, Tr=Tr)
Cruz3_LM, Cruz3_MEL, Cruz3_dist = QuantilBestFit(Cruz3_max.dropna().values, Tr=Tr)
Monte_LM, Monte_MEL, Monte_dist = QuantilBestFit(Monte_max.dropna().values, Tr=Tr)

Magan_q =  np.max([Magan_LM, Magan_MEL], axis=0)
Barbo_q =  np.max([Barbo_LM, Barbo_MEL], axis=0)
Armen_q =  np.max([Armen_LM, Armen_MEL], axis=0)
Cruz3_q =  np.max([Cruz3_LM, Cruz3_MEL], axis=0)
Monte_q =  np.max([Monte_LM, Monte_MEL], axis=0)

SJorg1 = Magan_q - Barbo_q
SJorg2 = Magan_q - Armen_q - Cruz3_q

import matplotlib
matplotlib.use ("Agg")
import matplotlib.pyplot as plt

rojo    = '#d7191c'
naranja = '#e66101'
verdeos = '#018571'
azul    = '#2c7bb6'
verde   = '#1a9641'
morado  = '#5e3c99'


plt.close('all')
figure =plt.figure(figsize=(7.9,4.8))
ax = figure.add_subplot(1,1,1)

ax.plot(Magan_sum,color=morado, lw=2, label='Magangue')
ax.plot(Barbo_sum,color=naranja,lw=2,label='Barbosa')
ax.plot(Monte_sum,color=azul,   lw=2, label='Montelibano')
ax.plot(Armen_sum,color=rojo,   lw=2, label='Armenia')
ax.plot(Cruz3_sum,color=verde,  lw=2, label='Tres Cruces')
ax.set_ylabel('Anual Flow [m$^{3}$ s$^{-1}$ year]')
ax.set_xlabel('Year')
ax.legend(loc=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(os.path.join(Path_out,'Balance.pdf'),format='pdf', transparent=True)
plt.savefig(os.path.join(Path_out,'Balance.png'),format='png', transparent=True)



plt.close('all')
figure =plt.figure(figsize=(12.8,8))
ax = figure.add_subplot(1,1,1)
ax.plot(Tr, Magan_q,color=morado, lw=2, label='Magangue')
ax.plot(Tr, Barbo_q,color=naranja,lw=2, label='Barbosa')
ax.plot(Tr, Monte_q,color=azul,   lw=2, label='Montelibano')
ax.plot(Tr, Armen_q,color=rojo,   lw=2, label='Armenia')
ax.plot(Tr, Cruz3_q,color=verde,  lw=2, label='Tres Cruces')
ax.set_xscale('log')
ax.legend(loc=0)
ax.set_xlabel('Return period [years]')
ax.set_ylabel(u'Flow [m$^{3}$ s$^{-1}$]')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.savefig(os.path.join(Path_out,'BalanceTr.pdf'),format='pdf', transparent=True)
plt.savefig(os.path.join(Path_out,'BalanceTr.png'),format='png', transparent=True)



plt.close('all')
figure =plt.figure(figsize=(12.8,8))
ax = figure.add_subplot(1,1,1)
ax.plot(Tr, SJorg1,color=morado, lw=2, label='Magangue - Barbosa')
ax.plot(Tr, SJorg2,color=rojo,   lw=2, label='Magangue - Armenia - Tres Cruces')
ax.set_xscale('log')
ax.legend(loc=0)
ax.set_xlabel('Return period [years]')
ax.set_ylabel(u'Flow [m$^{3}$ s$^{-1}$]')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.savefig(os.path.join(Path_out,'BalanceSanJorgeTr.pdf'),format='pdf', transparent=True)
plt.savefig(os.path.join(Path_out,'BalanceSanJorgeTr.png'),format='png', transparent=True)
