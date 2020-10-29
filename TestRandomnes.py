#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
import datetime as dt
from scipy import stats

def SingChange(Serie):
    """
    Count times where are sing change
    INPUTS
    Serie : list or array of with the data
    """
    if isinstance(Serie, list) == True:
        Serie = np.array(Serie)

    sing = np.zeros(len(Serie),dtype=int) +1
    sing[np.array(Serie)<0] = -1
    # return sum((x ^ y)<0 for x, y in zip(Serie, Serie[1:])) # only works for integers
    return sum((x ^ y)<0 for x, y in zip(sing, sing[1:]))


def RunsTest(Serie, significance=5E-2):
    """
    Make  run test (Rachas) for a series
    INPUTS
    Serie : list or array with the data
    significance : level of significance to acept or reject the null hypothesis
    OUTPUTS
    hip : boolean with the aceptance of rejection of the null hypothesis
    """
    S_median = np.median(Serie)
    runs = SingChange(Serie-S_median)
    n1 = np.where(Serie>=S_median)[0].shape[0]
    n2 = np.where(Serie< S_median)[0].shape[0]

    runs_exp = ((2*n1*n2)/(n1+n2))+1
    stan_dev = np.sqrt((2*n1*n2*(2*n1*n2-n1-n2))/ \
                       (((n1+n2)**2)*(n1+n2-1)))

    z = (runs-runs_exp)/stan_dev
    Z_norm = stats.norm.ppf(1-significance/2,loc=0,scale=1)

    hip = abs(z)<Z_norm
    return hip
