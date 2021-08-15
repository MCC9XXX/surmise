import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

class plotting:
    def __init__(self, cal):
        self.cal = cal

    def traceplot(self, **kwargs):
        '''
        Creates a traceplot given a calibrator.

        Parameters
        ----------
        **kwargs : dict, optional
            Optional dictionary containing options you would like to pass to
            fit function. Can edit which theta's to show'


        '''
        
        theta = self.cal.info['thetarnd']
        whichtheta = range(np.shape(theta)[1])
        if kwargs:
            for key in kwargs:
                if key == 'whichtheta':
                    whichtheta = kwargs[key]
                    for x in range(len(whichtheta)):
                        if int(whichtheta[x]) > int(theta.shape[1]):
                            raise ValueError('The theta you chose to observe is out of bounds of the chossen calibration')
        fig = plt.figure(constrained_layout = True)
        ax = fig.add_gridspec(3, 3*len(whichtheta))
        
        if len(whichtheta) == 1:
            axs = fig.add_subplot(ax[0:2,0:2])
            axs.plot(theta[:, whichtheta])
            lab_x = "$\\theta_{}$".format(whichtheta[0]) 
            axs.set_xlabel(lab_x, fontsize = 12)
            axs.set_xticks(np.arange(0, len(theta)+1, len(theta)*0.25))
            axs.set_yticks(np.arange(min(theta[:, whichtheta[0]])*0.95, max(theta[:, whichtheta[0]])*1.05, (max(theta[:, whichtheta[0]])-min(theta[:, whichtheta[0]]))*0.2))
            x_left, x_right = axs.get_xlim()
            y_low, y_high = axs.get_ylim()
            axs.set_aspect(abs((x_right - x_left)/(y_low-y_high)))
            
        else:
            for t in whichtheta:
                axs1 = fig.add_subplot(ax[0:2,(2*t):2*(t+1)])
                axs1.plot(self.cal.info['thetarnd'][:, whichtheta[t]])
                lab_x = "$\\theta_{}$".format(whichtheta[t]) 
                axs1.set_xlabel(lab_x, fontsize = 12)
                axs1.set_xticks(np.arange(0, len(theta)+1, len(theta)*0.25))
                axs1.set_yticks(np.arange(min(theta[:, whichtheta[t]])*0.95, max(theta[:, whichtheta[t]])*1.05, (max(theta[:, whichtheta[t]])-min(theta[:, whichtheta[t]]))*0.2))
                x_left, x_right = axs1.get_xlim()
                y_low, y_high = axs1.get_ylim()
                axs1.set_aspect(abs((x_right - x_left)/(y_low-y_high)))
                
                
        plt.show()
    
    def autocorr(self, lags, **kwargs):
        '''
        Creates an autocorrelation plot given a calibrator

        Parameters
        ----------
        lags : int
            An integer of time series
        **kwargs : dict, optional
            Optional dictionary containing options you would like to pass to
            fit function. Can edit which thetas' to show'


        '''

        theta = self.cal.info['thetarnd']
        whichtheta = range(np.shape(theta)[1])
        if kwargs:
            for key in kwargs:
                if key == 'whichtheta':
                    whichtheta = kwargs[key]
                    for x in range(len(whichtheta)):
                        if int(whichtheta[x]) > int(theta.shape[1]):
                            raise ValueError('The theta you chose to observe is out of bounds of the chossen calibration')
        fig, axs = plt.subplots(1, len(whichtheta))
        if len(whichtheta) == 1:
            axs.acorr(self.cal.info['thetarnd'][:, whichtheta[0]], maxlags = lags)
            lab_x = "$\\theta_{}$".format(whichtheta[0]) 
            axs.set_xlabel(lab_x, fontsize = 12)
        else:
            for t in whichtheta:
                axs[t].acorr(self.cal.info['thetarnd'][:, whichtheta[t]], maxlags = lags)
                lab_x = "$\\theta_{}$".format(whichtheta[t]) 
                axs[t].set_xlabel(lab_x, fontsize = 12)
        plt.show()
    
    def predictintvr(self,x, y, xscale = None, alpha = 0.5,  **kwargs):
        '''
        Creates a Prediction interval graph. Plots each data point and the prediction interval on the same graph

        Parameters
        ----------
        x : surmise.calibration.calibrator
           A calibrator object that contains information about calibration.
        y : array
            The data being plotted against.
        xscale : TYPE, optional
            DESCRIPTION. The default is None.
        alpha : int, optional
            An int that adjusts the percentiles that are graphed. The default is 0.5.
        **kwargs : dict, optional
            Optional dictionary containing options you would like to pass to
            fit function

        '''

        import matplotlib.pyplot as plt
        rndm_m = self.cal.info['thetarnd']
        whichtheta = range(np.shape(rndm_m)[1])
        if kwargs:
            for key in kwargs:
                if key == 'whichtheta':
                    whichtheta = kwargs[key]
                    for x in range(len(whichtheta)):
                        if int(whichtheta[x]) > int(rndm_m.shape[1]):
                            raise ValueError('The theta you chose to observe is out of bounds of the chossen calibration')
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
        '''
        

        Parameters
        ----------
        method : list, optional
            Specify which graphs you want to plot. The default is ['histogram','boxplot','density'].
        **kwargs : dict, optional
            Optional dictionary containing options you would like to pass to
            fit function.

        '''
         
        cal_theta = self.cal.info['thetarnd']
        whichtheta = range(len(cal_theta[0]))
        length_w = len(whichtheta)
        length_m = len(method)
        
        
        if kwargs:
            for key, value in kwargs.items():
                if key == 'whichtheta':
                    whichtheta = kwargs.get("whichtheta")
                    length_w = len(whichtheta)
                    for x in range(len(whichtheta)):
                        if int(whichtheta[x]) > int(cal_theta.shape[1]):
                            raise ValueError('The theta you chose to observe is out of bounds of the chossen calibration')
                    
                if key == 'fig':
                    fig = kwargs.get('fig')
                    
            if 'fig' not in kwargs:
                fig = plt.figure(constrained_layout = True)
                ax = fig.add_gridspec(5*len(method), 4*len(whichtheta))
                
                    
                    
        else:
            fig = plt.figure(constrained_layout = True)
            ax = fig.add_gridspec(4*len(method), 4*len(whichtheta))
        
            
        if (length_w == 1) != (length_m == 1):
            length = length_m*length_w-1
            cc = 0;
            while cc <= length:
                for ss in range(length_m):
                    for s in range(length_w):
                        if method[ss] == 'histogram':
                            axs = fig.add_subplot(ax[0:2,(2*s):2*(s+1)])
                            axs.hist(cal_theta[:, whichtheta[s]])
                            
                            lab_x = "$\\theta_{}$".format(whichtheta[s]) 
                            axs.set_xlabel(lab_x, fontsize = 12)
                            axs.set_xticks(np.arange(min(cal_theta[:, whichtheta[s]])*0.95, max(cal_theta[:, whichtheta[s]])*1.05, (max(cal_theta[:, whichtheta[s]])-min(cal_theta[:, whichtheta[s]]))*0.2))
                            
                            x_left, x_right = axs.get_xlim()
                            y_low, y_high = axs.get_ylim()
                            axs.set_aspect(abs((x_right - x_left)/(y_low-y_high)))
                            
                            cc += 1
                        elif method[ss] == 'boxplot':
                            axs = fig.add_subplot(ax[0:2,(2*s):2*(s+1)])
                            axs.boxplot(cal_theta[:, whichtheta[s]])
                            
                            lab_x = "$\\theta_{}$".format(whichtheta[s]) 
                            axs.set_xlabel(lab_x, fontsize = 12)
                            
                            x_left, x_right = axs.get_xlim()
                            y_low, y_high = axs.get_ylim()
                            axs.set_aspect(abs((x_right - x_left)/(y_low-y_high)))
                            axs.set_yticks(np.arange(min(cal_theta[:, whichtheta[s]])*0.95, max(cal_theta[:, whichtheta[s]])*1.05, (max(cal_theta[:, whichtheta[s]])-min(cal_theta[:, whichtheta[s]]))*0.2))
                            
                            cc += 1
                            
                        elif method[ss] == 'density':
                            axs = fig.add_subplot(ax[0:2,(2*s):2*(s+1)])
                            density = gaussian_kde(cal_theta[:, whichtheta[s]])
                            
                            ll = min(cal_theta[whichtheta[s]]) * 0.9
                            ul = max(cal_theta[whichtheta[s]]) * 1.1
                            z = np.linspace(ll,ul)
                            density.covariance_factor = lambda : .5
                            density._compute_covariance()
                            axs.plot(z, density(z))
                            
                            lab_x = "$\\theta_{}$".format(whichtheta[s]) 
                            axs.set_xlabel(lab_x, fontsize = 12)
                            
                            x_left, x_right = axs[cc].get_xlim()
                            y_low, y_high = axs[cc].get_ylim()
                            axs.set_aspect(abs((x_right - x_left)/(y_low-y_high)))
                            
                            cc += 1
                            
        elif length_w == 1 and length_m == 1:
            if method[0] == 'histogram':
                axs = fig.add_subplot(ax[0:2,0:2])
                axs.hist(cal_theta[:,whichtheta[0]])
                
                lab_x = "$\\theta_{}$".format(whichtheta[0]) 
                axs.set_xlabel(lab_x, fontsize = 12)
                axs.set_xticks(np.arange(min(cal_theta[:, whichtheta[0]])*0.95, max(cal_theta[:, whichtheta[0]])*1.05, (max(cal_theta[:, whichtheta[s]])-min(cal_theta[:, whichtheta[s]]))*0.2))
                
                x_left, x_right = axs.get_xlim()
                y_low, y_high = axs.get_ylim()
                axs.set_aspect(abs((x_right - x_left)/(y_low-y_high)))
                
            elif method[0] == 'boxplot':
                axs = fig.add_subplot(ax[0:2,0:2])
                axs.boxplot(cal_theta[:, whichtheta[0]])
                            
                lab_x = "$\\theta_{}$".format(whichtheta[0]) 
                axs.set_xlabel(lab_x, fontsize = 12)
                
                x_left, x_right = axs.get_xlim()
                y_low, y_high = axs.get_ylim()
                axs.set_aspect(abs((x_right - x_left)/(y_low-y_high)))
                            
            elif method[0] == 'density':
                axs = fig.add_subplot(ax[0:2,0:2])
                density = gaussian_kde(cal_theta[:, whichtheta[0]])
                            
                ll = min(cal_theta[whichtheta[0]]) * 0.9
                ul = max(cal_theta[whichtheta[0]]) * 1.1
                z = np.linspace(ll,ul)
                density.covariance_factor = lambda : .5
                density._compute_covariance()
                axs.plot(z, density(z))
                            
                lab_x = "$\\theta_{}$".format(whichtheta[0]) 
                axs.set_xlabel(lab_x, fontsize = 12)
                
                x_left, x_right = axs.get_xlim()
                y_low, y_high = axs.get_ylim()
                axs.set_aspect(abs((x_right - x_left)/(y_low-y_high)))
               
                
        elif length_w > length_m:
            for ii in range(length_m):
                for i in range(length_w):
                    if method[ii] == 'histogram':
                        axs = fig.add_subplot(ax[(3*ii):3*(ii+1),(3*i):3*(i+1)])
                        axs.hist(cal_theta[:, whichtheta[i]])
                        
                        lab_x = "$\\theta_{}$".format(whichtheta[i]) 
                        axs.set_xlabel(lab_x, fontsize = 12)
                        axs.set_xticks(np.arange(min(cal_theta[:, whichtheta[s]])*0.95, max(cal_theta[:, whichtheta[s]])*1.05, (max(cal_theta[:, whichtheta[s]])-min(cal_theta[:, whichtheta[s]]))*0.2))
                        
                        x_left, x_right = axs.get_xlim()
                        y_low, y_high = axs.get_ylim()
                        axs.set_aspect(abs((x_right - x_left)/(y_low-y_high)))
                    elif method[ii] == 'boxplot':
                        axs = fig.add_subplot(ax[(3*ii):3*(ii+1),(3*i):3*(i+1)])
                        axs.boxplot(cal_theta[:,whichtheta[i]])
                        
                        lab_x = "$\\theta_{}$".format(whichtheta[i]) 
                        axs.set_xlabel(lab_x, fontsize = 12)
                        
                        x_left, x_right = axs.get_xlim()
                        y_low, y_high = axs.get_ylim()
                        axs.set_aspect(abs((x_right - x_left)/(y_low-y_high)))
                        
                    elif method[ii] == 'density':
                        axs = fig.add_subplot(ax[(3*ii):3*(ii+1),(3*i):3*(i+1)])
                        density = gaussian_kde(cal_theta[:, whichtheta[i]])
                        
                        ll = min(cal_theta[:, whichtheta[i]]) * 0.9
                        ul = max(cal_theta[:, whichtheta[i]]) * 1.1
                        
                        z = np.linspace(ll, ul)
                        density.covariance_factor = lambda : .5
                        density._compute_covariance()
                        axs.plot(z, density(z))
                        lab_x = "$\\theta_{}$".format(whichtheta[i]) 
                        axs.set_xlabel(lab_x, fontsize = 12)
                        
                        x_left, x_right = axs.get_xlim()
                        y_low, y_high = axs.get_ylim()
                        axs.set_aspect(abs((x_right - x_left)/(y_low-y_high)))
                       
        else:
            for ii in range(len(method)):
                for i in range(len(whichtheta)):
                    if method[ii] == 'histogram':
                        axs = fig.add_subplot(ax[(3*i):3*(i+1),(3*ii):3*(ii+1)])
                        axs.hist(cal_theta[:, whichtheta[i]])
                        
                        lab_x = "$\\theta_{}$".format(whichtheta[i]) 
                        axs.set_xlabel(lab_x, fontsize = 12)
                        
                        axs.set_xticks(np.arange(min(cal_theta[:, whichtheta[i]])*0.95, max(cal_theta[:, whichtheta[i]])*1.05, (max(cal_theta[:, whichtheta[i]])-min(cal_theta[:, whichtheta[i]]))*0.2))
                        
                        
                        
                      
                    elif method[ii] == 'boxplot':
                        axs = fig.add_subplot(ax[(3*i):3*(i+1),(3*ii):3*(ii+1)])
                        axs.boxplot(cal_theta[:, whichtheta[i]])
                        
                        lab_x = "$\\theta_{}$".format(whichtheta[i])
                        axs.set_xlabel(lab_x, fontsize = 12)
                        
                        
                       
                        
                    elif method[ii] == 'density':
                        axs = fig.add_subplot(ax[(3*i):3*(i+1),(3*ii):3*(ii+1)])
                        density = gaussian_kde(cal_theta[:, whichtheta[i]])
                        
                        ll = min(cal_theta[:, whichtheta[i]]) * 0.9
                        ul = max(cal_theta[:, whichtheta[i]]) * 1.1
                        
                        z = np.linspace(ll,ul)
                        
                        density.covariance_factor = lambda : .5
                        density._compute_covariance()
                        axs.plot(z,density(z))
                        
                        lab_x = "$\\theta_{}$".format(whichtheta[i]) 
                        
                        
                        axs.set_xlabel(lab_x, fontsize = 12)
                        
                      
                        
                        
        plt.show()
    
class diagnostics:
        def __init__(self, cal):
            self.cal = cal
            
        def score(self, y, alpha = 0.5):
            '''
            calculates the interval score of given data

            Parameters
            ----------
            y : surmise.calibration.calibrator
                 A calibrator object that contains information about calibration.
            alpha : int, optional
                 An int that adjusts the percentiles that are graphed. The default is 0.5.

            Returns
            -------
            score : int
                Returns your interval score.

            '''
            
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

        def rmse(self):
            """Return the root mean squared error between the data and the mean of the calibrator."""
            ypred = self.cal.predict().mean()
        
            error = np.sqrt(np.mean((ypred - self.cal.y)**2))
            return error
        
        
        def energyScore(self, S = 1000):
            """Return the empirical energy score between the data and the predictive distribution. """
            
            ydist = self.cal.predict().rnd(S)  # shape = (num of samples, dimension of x)
        
            def norm(y1, y2):
                return np.sqrt(np.sum((y1 - y2)**2))
        
            lin_part = np.array([norm(ydist[i], self.cal.y) for i in range(S)]).mean()
            G = ydist @ ydist.T
            D = np.diag(G) + np.diag(G).reshape(1000, 1) - 2*G
            quad_part = -1/2 * np.mean(np.sqrt(D))
        
            score = lin_part + quad_part
        
            return score
        
        
        def energyScore_naive(self, S = 1000):
            """Return the empirical energy score between the data and the predictive distribution. """
            
            ydist = self.cal.predict().rnd(S)  # shape = (num of samples, dimension of x)
        
            def norm(y1, y2):
                return np.sqrt(np.sum((y1 - y2)**2))
        
            lin_part = np.array([norm(ydist[i], self.cal.y) for i in range(S)]).mean()
            quad_part = 0
            for s in range(S):
                quad_part += -1/2 * 1/S * np.array([norm(ydist[i], ydist[s]) for i in range(S)]).mean()
        
            score = lin_part + quad_part
        
            return score
            
    
    



        
