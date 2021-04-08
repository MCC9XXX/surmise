# comparison via LMC

import numpy as np
from random import sample
import scipy.stats as sps
from ml_methods import fit_logisticRegression
from surmise.emulation import emulator
from surmise.calibration import calibrator
from visualization_tools import boxplot_compare
from visualization_tools import plot_pred_interval
from visualization_tools import plot_model_data
from visualization_tools import plot_pred_errors
# Read data 
real_data = np.loadtxt('real_observations.csv', delimiter=',')
description = np.loadtxt('observation_description.csv', delimiter=',',dtype='object')
param_values = np.loadtxt('param_values.csv', delimiter=',')
func_eval = np.loadtxt('func_eval.csv', delimiter=',')

# Get the random sample of 100
<<<<<<< HEAD
rndsample = sample(range(0, 2000), 500)
=======
rndsample = sample(range(0, 2000), 1000)
>>>>>>> mlbayes
func_eval_rnd = func_eval[rndsample, :]
param_values_rnd = param_values[rndsample, :]

# (No Filter) Observe computer model outputs
plot_model_data(description, np.sqrt(func_eval_rnd), np.sqrt(real_data), param_values_rnd)

# Filter out the data
T0 = 50
T1 = 3000
par_in = param_values_rnd[np.logical_and.reduce((func_eval_rnd[:, 25] < 350,
                                                 func_eval_rnd[:, 100] > T0,
                                                 func_eval_rnd[:, 100] < T1)), :]
par_out = param_values_rnd[np.logical_or.reduce((func_eval_rnd[:, 25] > 350,
                                                 func_eval_rnd[:, 100] < T0,
                                                 func_eval_rnd[:, 100] > T1)), :]
func_eval_in = func_eval_rnd[np.logical_and.reduce((func_eval_rnd[:, 25] < 350,
                                                    func_eval_rnd[:, 100] > T0,
                                                    func_eval_rnd[:, 100] < T1)), :]

# (Filter) Observe computer model outputs
plot_model_data(description, np.sqrt(func_eval_in), np.sqrt(real_data), par_in)

# Get the x values 
<<<<<<< HEAD
keeptimepoints = np.arange(10, description.shape[0], step=5)
=======
keeptimepoints = np.arange(10, description.shape[0], step=20)
>>>>>>> mlbayes
#keeptimepoints = np.concatenate((np.arange(0, 150), np.arange(0, 150) + 192, np.arange(0, 150) + 2*192))

func_eval_in_tr = func_eval_in[:, keeptimepoints]
real_data_tr = real_data[keeptimepoints]
real_data_test = np.delete(real_data, keeptimepoints, axis=0) 
x = description
xtr = description[keeptimepoints, :]
xtest = np.delete(description, keeptimepoints, axis=0)

# (Filter) Fit an emulator via 'PCGP'
emulator_f_PCGPwM = emulator(x=x,
                             theta=par_in,
                             f=(func_eval_in)**(0.5),
                             method='PCGPwM')


    
# Define a class for prior of 10 parameters
class prior_covid:
    """ This defines the class instance of priors provided to the method. """
    #sps.uniform.logpdf(theta[:, 0], 3, 4.5)
    def lpdf(theta):
<<<<<<< HEAD
        return (sps.beta.logpdf((theta[:, 0]-1)/4, 3,3) +
                sps.beta.logpdf((theta[:, 1]-0.1)/4.9, 3,3) +
                sps.beta.logpdf((theta[:, 2]-1)/6, 3,3) +
                sps.beta.logpdf((theta[:, 3]-1)/6, 3,3)).reshape((len(theta), 1))
    def rnd(n):
        return np.vstack((1+4*sps.beta.rvs(3,3, size=n),
                          0.1+4.9*sps.beta.rvs(3,3, size=n),
                          1+6*sps.beta.rvs(3,3, size=n),
                          1+6*sps.beta.rvs(3,3, size=n))).T
=======
        return (sps.beta.logpdf((theta[:, 0]-1)/4, 2, 2) +
                sps.beta.logpdf((theta[:, 1]-0.1)/4.9, 2, 2) +
                sps.beta.logpdf((theta[:, 2]-1)/6, 2, 2) +
                sps.beta.logpdf((theta[:, 3]-1)/6, 2, 2)).reshape((len(theta), 1))
    def rnd(n):
        return np.vstack((1+4*sps.beta.rvs(2, 2, size=n),
                          0.1+4.9*sps.beta.rvs(2, 2, size=n),
                          1+6*sps.beta.rvs(2, 2, size=n),
                          1+6*sps.beta.rvs(2, 2, size=n))).T
>>>>>>> mlbayes

# class prior_covid:
#     """ This defines the class instance of priors provided to the method. """
#     #sps.uniform.logpdf(theta[:, 0], 3, 4.5)
#     def lpdf(theta):
#         return (sps.uniform.logpdf(theta[:, 0], 1, 4) +
#                 sps.uniform.logpdf(theta[:, 1], 0.1, 4.9) +
#                 sps.uniform.logpdf(theta[:, 2], 1, 6) +
#                 sps.uniform.logpdf(theta[:, 3], 1, 6)).reshape((len(theta), 1))
#     def rnd(n):
#         return np.vstack((sps.uniform.rvs(1, 4, size=n),
#                           sps.uniform.rvs(0.1, 4.9, size=n),
#                           sps.uniform.rvs(1, 6, size=n),
#                           sps.uniform.rvs(1, 6, size=n))).T
    
# Fit a classification model
classification_model = fit_logisticRegression(func_eval, param_values, T0, T1)

<<<<<<< HEAD
obsvar = 0.01*real_data_tr
=======
obsvar = 0.04*real_data_tr
>>>>>>> mlbayes


cal_f = calibrator(emu = emulator_f_PCGPwM,
                   y = np.sqrt(real_data_tr),
                   x = xtr,
                   thetaprior = prior_covid,
<<<<<<< HEAD
                   method = 'mlbayeswoodbury',
=======
                   method = 'directbayeswoodbury',
>>>>>>> mlbayes
                   yvar = obsvar,
                   args = {'usedir': True,
                           'sampler':'LMCv2'})

#plot_pred_interval(cal_f, xtr, np.sqrt(real_data_tr))
cal_f_theta = cal_f.theta.rnd(500)
#boxplot_param(cal_f_theta)
plot_pred_errors(cal_f, xtest, np.sqrt(real_data_test))


cal_f_ml = calibrator(emu = emulator_f_PCGPwM,
                   y = np.sqrt(real_data_tr),
                   x = xtr,
                   thetaprior = prior_covid,
                   method = 'mlbayeswoodbury',
                   yvar = obsvar,
                   args = {'usedir': True,
                           'clf_method': classification_model, 
                           'sampler':'LMCv2'})

#plot_pred_interval(cal_f_ml, xtr, np.sqrt(real_data_tr))
cal_f_ml_theta = cal_f_ml.theta.rnd(500)
#boxplot_param(cal_f_ml_theta)
plot_pred_errors(cal_f_ml, xtest, np.sqrt(real_data_test))

boxplot_compare(cal_f_theta, cal_f_ml_theta)
print('script done!!!!!!!')