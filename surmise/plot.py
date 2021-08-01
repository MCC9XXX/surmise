import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

class plotting:
    def __init__(self, cal):
        self.cal = cal

    def traceplot(self, **kwargs):
        
        theta = self.cal.info['thetarnd']
        whichtheta = range(np.shape(theta)[1])
        if kwargs:
            for key in kwargs:
                if key == 'whichtheta':
                    whichtheta = kwargs[key]
        fig, axs = plt.subplots(1, len(whichtheta))
        if len(whichtheta) == 1:
            axs.plot(self.cal.info['thetarnd'][:, whichtheta])
        else:
            for t in whichtheta:
                axs[t].plot(self.cal.info['thetarnd'][:, whichtheta[t]])
        plt.show()
    
    def autocorr(self, lags, **kwargs):

        theta = self.cal.info['thetarnd']
        whichtheta = range(np.shape(theta)[1])
        if kwargs:
            for key in kwargs:
                if key == 'whichtheta':
                    whichtheta = kwargs[key]
        fig, axs = plt.subplots(1, len(whichtheta))
        if len(whichtheta) == 1:
            axs.acorr(self.cal.info['thetarnd'][:, whichtheta[0]], maxlags = lags)
        else:
            for t in whichtheta:
                axs[t].acorr(self.cal.info['thetarnd'][:, whichtheta[t]], maxlags = lags)
        plt.show()
    
    def predictintvr(self,x, y, xscale = None, alpha = 0.5,  **kwargs):

        import matplotlib.pyplot as plt
        rndm_m = self.cal.info['thetarnd']
        whichtheta = range(np.shape(rndm_m)[1])
        if kwargs:
            for key in kwargs:
                if key == 'whichtheta':
                    whichtheta = kwargs[key]
        fig, axs = plt.subplots(1, len(whichtheta))
                
        post = self.cal.predict()
        rndm_m = post.rnd(s = 1000)
            
        upper = np.percentile(rndm_m, 100 - alpha/2, axis = 0)
        lower = np.percentile(rndm_m, alpha/2, axis = 0)
        median = np.percentile(rndm_m, 50, axis = 0)
            
        if xscale == None:
            axs.plot(x, median, color = 'black')
            axs.fill_between(x, lower, upper, color = 'grey')
            axs.plot(x, y, 'ro', markersize = 5, color='red')
                
        else:
            axs.plot(xscale, median, color = 'black')
            axs.fill_between(xscale, lower, upper, color = 'grey')
            axs.plot(xscale, y, 'ro', markersize = 5, color='red')


        
    def plot(self, method = ['histogram','boxplot','density'], **kwargs):
         
        cal_theta = self.cal.info['thetarnd']
        whichtheta = range(len(cal_theta[0]))
        subplot = "default"
        length_w = len(whichtheta)
        length_m = len(method)
        
        
        if kwargs:
            for key in kwargs:
                if key == 'whichtheta':
                    whichtheta = kwargs[key]
                    length_w = len(whichtheta)
                    fig, axs = plt.subplots(length_w, length_m, figsize=(length_w * 6,length_m * 3))
                    
                if key == 'subplot':
                    subplot = kwargs[key] 
                    if subplot == "transpose":
                        fig, axs = plt.subplots(length_m, length_w, figsize=(length_m * 6,length_w * 6))
                        
                if key == 'axs':
                    fig, axs = kwargs[key]
        else:
            fig, axs = plt.subplots(length_w, length_m, figsize=(length_w * 6,length_m * 3))
        
            
        if length_w == 1 or length_m == 1:
            length = length_m*length_w-1
            cc = 0;
            while cc <= length:
                for ss in range(length_m):
                    for s in range(length_w):
                        if method[ss] == 'histogram':
                            axs[cc].hist(cal_theta[:, whichtheta[s]])
                            
                            lab_x = "$\\theta_{}$".format(cc) 
                            axs[cc].set_xlabel(lab_x, fontsize = 12)
                            cc += 1
                        elif method[ss] == 'boxplot':
                            axs[cc].boxplot(cal_theta[:, whichtheta[s]])
                            
                            lab_x = "$\\theta_{}$".format(cc) 
                            axs[cc].set_xlabel(lab_x, fontsize = 12)
                            cc += 1
                            
                        elif method[ss] == 'density':
                            density = gaussian_kde(cal_theta[:, whichtheta[s]])
                            
                            ll = min(cal_theta[whichtheta[s]]) * 0.9
                            ul = max(cal_theta[whichtheta[s]]) * 1.1
                            z = np.linspace(ll,ul)
                            density.covariance_factor = lambda : .5
                            density._compute_covariance()
                            axs[cc].plot(z, density(z))
                            
                            lab_x = "$\\theta_{}$".format(cc) 
                            axs[cc].set_xlabel(lab_x, fontsize = 12)
                            cc += 1
                
        elif subplot == "transpose":
            for ii in range(length_m):
                for i in range(length_w):
                    if method[ii] == 'histogram':
                        axs[ii, i].hist(cal_theta[:, whichtheta[i]])
                        
                        lab_x = "$\\theta_{}$".format(i) 
                        axs[ii,i].set_xlabel(lab_x, fontsize = 12)
                        
                    elif method[ii] == 'boxplot':
                        axs[ii, i].boxplot(cal_theta[:,whichtheta[i]])
                        
                        lab_x = "$\\theta_{}$".format(i) 
                        axs[ii,i].set_xlabel(lab_x, fontsize = 12)
                        
                    elif method[ii] == 'density':
                        density = gaussian_kde(cal_theta[:, whichtheta[i]])
                        
                        ll = min(cal_theta[:, whichtheta[i]]) * 0.9
                        ul = max(cal_theta[:, whichtheta[i]]) * 1.1
                        
                        z = np.linspace(ll, ul)
                        density.covariance_factor = lambda : .5
                        density._compute_covariance()
                        axs[ii,i].plot(z, density(z))
                        lab_x = "$\\theta_{}$".format(i) 
                        axs[ii,i].set_xlabel(lab_x, fontsize = 12)
                        
        
        else:
            for ii in range(len(method)):
                for i in range(len(whichtheta)):
                    if method[ii] == 'histogram':
                        axs[i,ii].hist(cal_theta[:, whichtheta[i]])
                        
                        lab_x = "$\\theta_{}$".format(i) 
                        axs[i,ii].set_xlabel(lab_x, fontsize = 12)
                        
                      
                    elif method[ii] == 'boxplot':
                        axs[i,ii].boxplot(cal_theta[:, whichtheta[i]])
                        
                        lab_x = "$\\theta_{}$".format(i) 
                        axs[i,ii].set_xlabel(lab_x, fontsize = 12)
                        
                    elif method[ii] == 'density':
                        density = gaussian_kde(cal_theta[:, whichtheta[i]])
                        
                        ll = min(cal_theta[:, whichtheta[i]]) * 0.9
                        ul = max(cal_theta[:, whichtheta[i]]) * 1.1
                        
                        z = np.linspace(ll,ul)
                        
                        density.covariance_factor = lambda : .5
                        density._compute_covariance()
                        axs[i,ii].plot(z,density(z))
                        
                        lab_x = "$\\theta_{}$".format(i) 
                        axs[i,ii].set_xlabel(lab_x, fontsize = 12)
                        
        plt.show()
    
class diagnostics:
        def __init__(self, cal):
            self.cal = cal
        def score(self, y, alpha):
             data = self.cal.info['thetarnd']
             no_of_rows = np.shape(data)[0]
             no_of_columns = np.shape(data)[1]
             max_array = [[0 for col in range(no_of_columns)] for row in range(no_of_rows)]
             
             for i in range(no_of_rows):
                 maxi = 0
                 for j in range(no_of_columns):
                     if data[i][j] > maxi:
                         maxi = data[i][j]
                         max_array[i][j] = data[i][j]
                 
             upper = np.percentile(data, 100 - alpha/2, axis = 0)
             lower = np.percentile(data, alpha/2, axis = 0)
             
             score = -(upper - lower) - (2/alpha)*(lower - max_array) - (2/alpha)[max_array - upper]
             
             return score
            
    
    



        
