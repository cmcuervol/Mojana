#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
import datetime as dt

from Modules import Read
from Modules.FitStats import BestFit, QuantilBestFit
from Modules.Utils import Listador, Salto, HistogramValues

import matplotlib
matplotlib.use ("Agg")
import matplotlib.pyplot as plt

rojo    = '#d7191c'
naranja = '#e66101'
verdeos = '#018571'
azul    = '#2c7bb6'
verde   = '#1a9641'
morado  = '#5e3c99'

Path_yearly = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Datos/Anuales/'))
Path_out    = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Hydrograms'))
Est_path    = os.path.abspath(os.path.join(os.path.dirname(__file__), 'CleanData'))
def GraphHistrogram(Values, bins=10, label='', title='', name='Histogram', pdf=True, png=False, PathFigs=Path_out,):
    """
    Grahp histogram
    INPUTS
    Values   : List or array to graph the histogram
    bins     : integer of the number of bins to calculate the histogram
    label    : string of the label
    title    : Figure title
    name     : Name to save figure
    pdf      : Boolean to save figure in pdf format
    png      : Boolean to save figure in png format
    PathFigs : Aboslute route to directory where figure will be save
    """
    histo = HistogramValues(Values, bins)
    plt.close('all')
    figure =plt.figure(figsize=(7.9,4.8))
    ax = figure.add_subplot(1,1,1)

    ax.plot(histo[1],histo[0],color=azul,lw=2)
    ax.fill_between(histo[1],histo[0],color=verde,alpha=0.6)
    ax.set_ylabel('Relative frecuency')
    ax.set_xlabel(label)

    # ax.plot(times,flow, linewidth=2,color=azul)
    # ax.set_xlabel('Time [hours]')
    # ax.set_ylabel(u'U [m$^{3}$ s$^{-1}$ mm$^{-1}$]')
    ax.set_title(title)
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


Estaciones = Listador(Path_yearly, final='csv')

Q_LM = np.empty(len(Estaciones), dtype=float)
Q_MEL= np.empty(len(Estaciones), dtype=float)
dist = np.empty(len(Estaciones), dtype='<U15')

# ['mean', 'max', 'min', 'sum', 'count']

Hydrogram = []
for i in range(len(Estaciones)):

    S_yearly = pd.read_csv(os.path.join(Path_yearly,Estaciones[i]), index_col = 0, header = 0)

    Q_LM[i], Q_MEL[i], dist[i] = QuantilBestFit(S_yearly['max'].dropna().values, Tr=2.33)

    Q = np.max((Q_LM[i], Q_MEL[i]))

    code = Estaciones[i].split('[')[-1].split(']')[0]
    S_diurnal = Read.EstacionCSV_pd(code+'.csv', Estaciones[i].split('.csv')[0], Est_path)
    # S_diurnal = S_diurnal.sort_index

    idh = np.where(S_diurnal.values >Q)[0] # superan umbral

    ide = Salto(idh, 1, condition='>') #identificar eventos
    ide = np.insert(ide, 0, 0) # add 0 to make it easy
    evn = []
    Qp_dur = []
    Qp_vol = []
    for e in range(len(ide)-1):
        Event = S_diurnal.iloc[idh[ide[e]:ide[e+1]]]

        evn.append([Event.mean() .values[0],
                    Event.max()  .values[0],
                    Event.min()  .values[0],
                    Event.sum()  .values[0],
                    Event.count().values[0],
                    ])
        Qp_dur.append(Event.max().values[0]/Event.count().values[0])
        Qp_vol.append(Event.max().values[0]/Event.sum()  .values[0])
    GraphHistrogram(Qp_dur,bins=20,
                    label=u'$\mathrm{Q_p/duration}$ [m$^{3}$ s$^{-1}$ day$^{-1}$]',
                    title=Estaciones[i].split('.csv')[0],
                    name='Peak_duration_'+Estaciones[i].replace(' ', '').split('.csv')[0] )
    GraphHistrogram(Qp_vol,bins=20,
                    label=u'$\mathrm{Q_p/Vol}$ [day s$^{-1}$]',
                    title=Estaciones[i].split('.csv')[0],
                    name='Peak_vol_'+Estaciones[i].replace(' ', '').split('.csv')[0] )

    Hydrogram.append(evn)
