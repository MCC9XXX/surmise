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

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)


def graph(calibration, whichtheta, n, method, subplot):
    fig, axs = plt.subplots(subplot[0],subplot[1], figsize=(14,4))
    cal_theta = calibration.theta.rnd(n)    
    for i in range(len(method)):
        if method[i] == 'histogram':
            axs[i].hist(cal_theta[:, whichtheta])
            
        elif method[i] == 'boxplot':
            axs[i].boxplot(cal_theta[:,whichtheta])
    plt.show()
    
    

        
        