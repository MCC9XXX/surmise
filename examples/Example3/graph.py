#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 01:13:58 2021

@author: justinchen
"""

import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt 
from surmise.emulation import emulator
from surmise.calibration import calibrator
from numpy import random
from scipy.stats import gaussian_kde

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)


def graph(calibration, whichtheta = [0], n = 1000, method = ['histogran'], subplot = [1,1]):
    fig, axs = plt.subplots(subplot[0],subplot[1], figsize=(14,4))
    cal_theta = calibration.theta.rnd(n)
    for x in range(len(subplot)):
        if subplot[x] == 1:
            length = len(method)*len(whichtheta)
            for ss in range(len(method)):
                for s in range(len(whichtheta)):
                    if method[ss] == 'histogram':
                        axs[ss].hist(cal_theta[:,whichtheta[s]])
                    elif method[ss] == 'boxplot':
                        axs[ss].boxplot(cal_theta[:,whichtheta[s]])
                    elif method[ss] == 'density':
                        density = gaussian_kde(whichtheta[s])
                        z = np.linspace(0,20)
                        density.covariance_factor = lambda : .5
                        density._compute_covariance()
                        axs[ss].plot(z,density(z))
        else:
            for ii in range(len(method)):
                for i in range(len(whichtheta)):
                    if method[ii] == 'histogram':
                        axs[i,ii].hist(cal_theta[:, whichtheta[i]])
                    elif method[ii] == 'boxplot':
                        axs[i,ii].boxplot(cal_theta[:,whichtheta[i]])
                    elif method[ii] == 'density':
                        density = gaussian_kde(cal_theta[whichtheta[i]])
                        z = np.linspace(0,20)
                        density.covariance_factor = lambda : .5
                        density._compute_covariance()
                        axs[i,ii].plot(z,density(z))
                    
    plt.show()