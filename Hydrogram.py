#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
import datetime as dt
from scipy import stats

from Modules import Read

from Modules.FitStats import BestFit, QuantilBestFit
from Modules.Utils import Listador, Salto

Path_yearly = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Datos/Anuales/'))
Est_path    = os.path.abspath(os.path.join(os.path.dirname(__file__), 'CleanData'))


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
    for e in range(len(ide)-1):
        Event = S_diurnal.iloc[idh[ide[e]:ide[e+1]]]

        evn.append([Event.mean() .values[0],
                    Event.max()  .values[0],
                    Event.min()  .values[0],
                    Event.sum()  .values[0],
                    Event.count().values[0],
                    ])
    Hydrogram.append(evn)
