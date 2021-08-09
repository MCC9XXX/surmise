import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as sps
from surmise.emulation import emulator
from surmise.calibration import calibrator
from surmise.plot import plotting

# 1D Example
def model1(x, theta):
    f = np.zeros((len(x), len(theta)))
    
    for k in range(0, theta.shape[0]):
        f[:, k] = ((2*np.sin(x*12)*np.sqrt(x) + np.cos(x*62)*x**2 + np.exp(2/3*x) - 1) + theta[k, :]).reshape(-1)

    return f

# Data
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
sigma_noise = 0.1
x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])[:,None]
ndat = len(y)
theta_true = np.array([[1]])
y = model1(x, theta_true) + (np.random.randn(ndat) * sigma_noise).reshape((ndat, 1))

class prior1:
    def lpdf(theta):
        return (sps.norm.logpdf(theta[:, 0], 0, 1)).reshape((len(theta), 1)) 

    def rnd(n):
        return np.reshape(sps.norm.rvs(0, 1, size=n), (-1,1)) 

# Build an emulator 
theta_sample = prior1.rnd(100)
xnew = np.reshape(np.array([list(range(0, 101))])/100, (101, 1))
f1 = model1(xnew, theta_sample)

emu_1 = emulator(x=xnew, 
                 theta=theta_sample, 
                 f=f1, 
                 method='PCGP') 

obsvar1 = np.maximum(0.1, 0.3*y)

# Build a calibrator 
cal_1 = calibrator(emu=emu_1,
                   y=y,
                   x=x,
                   thetaprior=prior1, 
                   method='directbayes',
                   yvar=obsvar1)

# NOTE TO JUSTIN: Please include all kinds of things that we can do with your plotting module
plot_cal_1 = plotting(cal_1)
plot_cal_1.plot(['boxplot', 'histogram'], whichtheta = [0])
plot_cal_1.plot(['density']) 
plot_cal_1.traceplot()

fig_1, axs_1 = plt.subplots(1,3)
plot_cal_1.plot(['boxplot', 'histogram', 'density'], fig = fig_1, axs = axs_1)




# 2D Example
def model2(x, theta):
    f = np.zeros((len(x), len(theta)))
    
    for k in range(0, theta.shape[0]):
        f[:, k] = ((2*np.sin(x*12)*np.sqrt(x) + np.cos(x*theta[k, 1])*x**2 + np.exp(2/3*x) - 1) + theta[k, 0]).reshape(-1)

    return f

# Data
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
sigma_noise = 0.1
x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])[:,None]
ndat = len(y)
theta_true = np.array([[1, 62]])
y = model2(x, theta_true) + (np.random.randn(ndat) * sigma_noise).reshape((ndat, 1))

class prior2:
    def lpdf(theta):
        return (sps.norm.logpdf(theta[:, 0], 0, 1) + 
                sps.norm.logpdf(theta[:, 1], 62, 1)).reshape((len(theta), 1)) 

    def rnd(n):
        return np.vstack((sps.norm.rvs(0, 1, size=n),  # initial height deviation
                          sps.norm.rvs(62, 1, size=n))).T 

# Build an emulator
theta_sample = prior2.rnd(100)
xnew = np.reshape(np.array([list(range(0, 101))])/100, (101, 1))
f2 = model2(xnew, theta_sample)
emu_2 = emulator(x=xnew, 
                 theta=theta_sample, 
                 f=f2, 
                 method='PCGP') 

obsvar2 = np.maximum(0.1, 0.3*y)

# Build a calibrator 
cal_2 = calibrator(emu=emu_2,
                        y=y,
                        x=x,
                        thetaprior=prior2, 
                        method='directbayes',
                        yvar=obsvar2)

# Observe posterior distribution
plot_cal = plotting(cal_2)
plot_cal.plot(['boxplot', 'histogram'], whichtheta = [0, 1])
plot_cal.plot(['boxplot', 'histogram', 'density'], whichtheta = [0, 1])
plot_cal.plot(['boxplot', 'histogram', 'density'])

# Observe trace plots
plot_cal.traceplot()
plot_cal.traceplot(whichtheta = [1])

# Observe auto correlation plots
plot_cal.autocorr(lags=5)
plot_cal.autocorr(lags = 5, whichtheta = [1])
