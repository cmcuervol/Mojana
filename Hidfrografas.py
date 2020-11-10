#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
import datetime as dt

from Modules import Read
from Modules.Utils import Listador

Path_input = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Datos/Hidrografas/'))

Estaciones = Listador(Path_input, final='.csv')
dat = np.genfromtxt(os.path.join(Path_input,Estaciones[0]),delimiter=',')
dat[0,0] = 0

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

ax.plot(dat[:,0],dat[:,1],color=azul,lw=2)
ax.fill_between(dat[:,0],dat[:,1],color=verde,alpha=0.6)
ax.set_ylabel('Q [m$^{3}$ s$^{-1}$]')
ax.set_xlabel('Time [hours]')

ax.set_title(Estaciones[0].split('.csv')[0])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
plt.savefig(os.path.join(Path_input,Estaciones[0].split('.csv')[0]+'.pdf'),format='pdf', transparent=True)
plt.savefig(os.path.join(Path_input,Estaciones[0].split('.csv')[0]+'.png'),format='png', transparent=True)
