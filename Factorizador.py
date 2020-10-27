#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import datetime as dt

# from Utils import Listador

Path_Q = os.path.join(os.getcwd(),'Datos/Caudales/')
Path_N = os.path.join(os.getcwd(),'Datos/Niveles/')
Path_n = os.path.join(os.getcwd(),'Datos/Niveles-1/')

def ReadEstacion(name, col_name, path=Path_Q):
    """
    Read flow or level data
    INPUTS
    name : station name
    path : folder where are save the data

    OUTPUTS
    D : DataFrame with the data, index are datetime
    """
    data  = np.genfromtxt(os.path.join(path,name),delimiter=',')
    index = np.genfromtxt(os.path.join(path,name),delimiter=',', dtype=str)
    D = pd.DataFrame(data[:,1].ravel(), index=pd.DatetimeIndex(index[:,0]), columns=[col_name])

    return D

def ReturnPeriod(serie, value):
    """
    Calculate the return period of a value in a serie
    INPUTS
    serie : pandas series with the data
    value : float to cacule the return period
    """
    idx = np.where((~np.isnan(serie.values))&(serie.values>=value))[0]
    if len(idx) == 0: # is a NaN value
        return np.nan
    else:
        excedence = len(idx)/len(serie)
        return 1/excedence

def FindQPR(serie, events):
    """
    Find flow and return period in some events
    INPUTS
    serie  : pandas series with the data
    events : indexes or dates of the events
    """
    q  = np.zeros(len(events))
    pr = np.zeros(len(events))
    for i in range(len(events)):
        ide = np.where(serie.index == events[i])[0][0]
        q [i] = serie.iloc[ide].values[0]
        pr[i] = ReturnPeriod(serie, q[i])

    Q  = pd.DataFrame(q,  index=events, columns=serie.columns)
    PR = pd.DataFrame(pr, index=events, columns=serie.columns)
    return Q, PR

#series to use

Magan = ReadEstacion('MAGANGUE [25027680].csv',           'MAGANGUE')
Barbo = ReadEstacion('BARBOSA [25027530].csv',            'BARBOSA')
Armen = ReadEstacion('ARMENIA [25027360].csv',            'ARMENIA')
Cruz3 = ReadEstacion('TRES CRUCES [25027640].csv',        'CRUCES')
Nechi = ReadEstacion('LA ESPERANZA NECHI [27037010].csv', 'NECHI')
Monte = ReadEstacion('MONTELIBANO - AUT [25017010].csv',  'MONTELIBANO')
Coque = ReadEstacion('LA COQUERA - AUT [26247020].csv',   'COQUERA')

Beirt = ReadEstacion('BEIRUT - AUT [25027760].csv',     'BEIRUT', path=Path_N)
Marco = ReadEstacion('SAN MARCOS - AUT [25027220].csv', 'MARCOS', path=Path_N)
Jegua = ReadEstacion('JEGUA [25027240].csv',            'JEGUA',  path=Path_N)

Anton = pd.read_csv(os.path.join(Path_n, 'Bolivar_NivMax_Mes.csv'),sep=';')
Anton = pd.DataFrame(Anton['Valor'].values, index=pd.DatetimeIndex(Anton['Fecha'],dayfirst=True), columns=['ANTONIO'])
#eventos
idx   = np.where((~np.isnan(Magan.values))&(Magan.values>=10000))[0]
dates = Magan.iloc[idx].index

series = [Armen, Cruz3, Nechi, Monte, Coque, Barbo, Magan, Beirt, Marco, Jegua, Anton]

for i in range(len(series)):
    q, pr = FindQPR(series[i], dates)
    if i == 0:
        Q  = q
        PR = pr
    else:
        Q  = Q .join(q,  how='inner')
        PR = PR.join(pr, how='inner')

# Q = Q.join(Magan.iloc[idx], how='inner')

Q .to_csv(os.path.join(os.getcwd(), 'CaudalesEventsMax.csv'), sep=',')
PR.to_csv(os.path.join(os.getcwd(), 'ReturnPeriodEventsMax.csv'), sep=',')
