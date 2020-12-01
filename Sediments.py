#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
import datetime as dt
from scipy import stats
import pymannkendall as mk

from Modules import Read
from Modules.Utils import Listador, FindOutlier, Cycles
from Modules.Graphs import GraphSerieOutliers, GraphDataFrames

Path_out = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Sedimentos'))
Est_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'CleanSedimentos'))


Res_est = Listador(Est_path, inicio='resolidos', final='.csv')

Armen_r = pd.read_csv(os.path.join(Est_path, Res_est[0]), index_col=0)
Coque_r = pd.read_csv(os.path.join(Est_path, Res_est[1]), index_col=0)
Esper_r = pd.read_csv(os.path.join(Est_path, Res_est[2]), index_col=0)
Rayar_r = pd.read_csv(os.path.join(Est_path, Res_est[3]), index_col=0)
Magan_r = pd.read_csv(os.path.join(Est_path, Res_est[4]), index_col=0)
Monte_r = pd.read_csv(os.path.join(Est_path, Res_est[5]), index_col=0)
Palen_r = pd.read_csv(os.path.join(Est_path, Res_est[6]), index_col=0)
Armen_r.index = pd.DatetimeIndex(Armen_r.index)
Coque_r.index = pd.DatetimeIndex(Coque_r.index)
Esper_r.index = pd.DatetimeIndex(Esper_r.index)
Rayar_r.index = pd.DatetimeIndex(Rayar_r.index)
Magan_r.index = pd.DatetimeIndex(Magan_r.index)
Monte_r.index = pd.DatetimeIndex(Monte_r.index)
Palen_r.index = pd.DatetimeIndex(Palen_r.index)

names = ['Armernia',
         'La Coquera',
         'La Esperanza',
         'La Raya',
         'Magangué',
         'Montelibano',
         'Palenquito']

Dfs = [Armen_r, Coque_r, Esper_r, Rayar_r, Magan_r, Monte_r, Palen_r]

GraphDataFrames(Dfs, names, col=0,label='Nivel [cm]',
                name='Niveles', pdf=False, png=True,PathFigs=Path_out)
GraphDataFrames(Dfs, names, col=1,label=u'Caudal liquido [m$^{3}$ s$^{-1}$]',
                name='Caudal',pdf=False, png=True,PathFigs=Path_out)
GraphDataFrames(Dfs, names, col=2, label= 'Gasto solido [kg s$^{-1}$]',
                name='Gasto',pdf=False, png=True,PathFigs=Path_out)
GraphDataFrames(Dfs, names, col=3, label=u'Concetración media [kg m$^{-3}$]',
                name='Concentra_med',pdf=False, png=True,PathFigs=Path_out)
GraphDataFrames(Dfs, names, col=4, label=u'Concetración superficial [kg m$^{-3}$]',
                name='Concentra_sup',pdf=False, png=True,PathFigs=Path_out)
