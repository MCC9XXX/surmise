import numpy as np
import scipy.optimize as spo

'''
Metropolis-adjusted Langevin algorithm or Langevin Monte Carlo (LMC)
'''


def sampler(logpostfunc, options):
    '''

    Parameters
    ----------
    logpostfunc : function
        A function call describing the log of the posterior distribution.
            If no gradient, logpostfunc should take a value of an m by p numpy
            array of parameters and theta and return
            a length m numpy array of log posterior evaluations.
            If gradient, logpostfunc should return a tuple.  The first element
            in the tuple should be as listed above.
            The second element in the tuple should be an m by p matrix of
            gradients of the log posterior.
    options : dict
        a dictionary contains the output of the sampler.
        Required -
            theta0: an m by p matrix of initial parameter values.  p must be larger
            than 10m for the code to work.
        Optional -
            numsamp: the number of samplers you want from the posterior.
            Default is 2000.

    Returns
    -------
    TYPE
        numsamp by p of sampled parameter values

    '''

    if 'theta0' in options.keys():
        theta0 = options['theta0']
        if theta0.shape[0] < 10*theta0.shape[1]:
            raise ValueError('Supply more initial thetas!')
    else:
        raise ValueError('Unknown theta0')

    # Initialize
    if 'numsamp' in options.keys():
        numsamp = options['numsamp']
    else:
        numsamp = 2000

    # Minimum effective sample size (ESS) desired in the returned samples
    tarESS = np.max((150, 10 * theta0.shape[1]))

    # Test
    testout = logpostfunc(theta0[0:2, :])
    if type(testout) is tuple:
        if len(testout) != 2:
            raise ValueError('log density does not return 1 or 2 elements')
        if testout[1].shape[1] is not theta0.shape[1]:
            raise ValueError('derivative appears to be the wrong shape')

        logpostf = logpostfunc

        def logpostf_grad(theta):
            return logpostfunc(theta)[1]

        try:
            testout = logpostfunc(theta0[10, :], return_grad=False)
            if type(testout) is tuple:
                raise ValueError('Cannot stop returning a grad')

            def logpostf_nograd(theta):
                return logpostfunc(theta, return_grad=False)

        except Exception:
            def logpostf_nograd(theta):
                return logpostfunc(theta)[0]
    else:
        logpostf_grad = None
        logpostf = logpostfunc
        logpostf_nograd = logpostfunc

    if logpostf_grad is None:
        rho = 2 / theta0.shape[1] ** (1/2)
        taracc = 0.25
    else:
        rho = 2 / theta0.shape[1] ** (1/6)
        taracc = 0.60


    thetaop = theta0[:10, :]
    thetastart = theta0
    thetac = np.mean(theta0, 0)
    thetas = np.maximum(np.std(theta0, 0), 10 ** (-8) * np.std(theta0))


    theta0 = shrinkaroundcenter(theta0, logpostf_nograd)


    thetac = np.mean(theta0, 0)
    thetas = np.maximum(np.std(theta0, 0), 10 ** (-8) * np.std(theta0))
    def neglogpostf_nograd(thetap):
        theta = thetac + thetas * thetap
        return -logpostf_nograd(theta.reshape((1, len(theta))))[0]
    if logpostf_grad is not None:
        def neglogpostf_grad(thetap):
            theta = thetac + thetas * thetap
            return -thetas * logpostf_grad(theta.reshape((1, len(theta))))

    boundL = np.maximum(-10*np.ones(theta0.shape[1]),
                        np.min((theta0 - thetac)/thetas, 0))
    boundU = np.minimum(10*np.ones(theta0.shape[1]),
                        np.max((theta0 - thetac)/thetas, 0))
    bounds = spo.Bounds(boundL, boundU)

    # begin preoptimizer
    thetaop = np.zeros((10,thetaop.shape[1]))
    for k in range(0, 10):
        if logpostf_grad is None:
            opval = spo.minimize(neglogpostf_nograd,
                                 (thetaop[k, :] - thetac) / thetas,
                                 method='L-BFGS-B',
                                 bounds=bounds)
            thetaop[k, :] = thetac + thetas * opval.x
        else:
            opval = spo.minimize(neglogpostf_nograd,
                                 (thetaop[k, :] - thetac) / thetas,
                                 method='L-BFGS-B',
                                 jac=neglogpostf_grad,
                                 bounds=bounds)
            thetaop[k, :] = thetac + thetas * opval.x
    # end Preoptimizer

    theta0 = np.vstack((theta0,thetaop))
    theta0 = shrinkaroundcenter(theta0, logpostf_nograd)

    thetas = np.maximum(np.std(theta0, 0), 10 ** (-8) * np.std(theta0))
    covmat0 = np.diag(thetas)

    temps =np.concatenate((np.linspace(10,1,40),np.ones(10)))
    temps = np.array(temps,ndmin=2).T
    rho = 1/np.sqrt(theta0.shape[0])

    maxiters = 10
    numsamppc = 200
    numchain = 50

    thetac = theta0[np.random.choice(range(0, theta0.shape[0]),
                                        size=numchain), :]
    if logpostf_grad is not None:
        fval, dfval = logpostf(thetac)
        fval = fval/temps
        dfval = dfval/temps
    else:
        fval = logpostf_nograd(thetac)
        fval = fval/temps


    thetasave = np.zeros((numchain, numsamppc, thetac.shape[1]))
    numtimes = 0
    hc = np.diag(thetas)
    covmat0 = np.diag(thetas**2)


    numtimes = 0
    tau = -1
    rho = 2 * (1 + (np.exp(2 * tau) - 1) / (np.exp(2 * tau) + 1))
    adjrho = rho*temps**(1/3)
    #autotune

    tunesteps = 100
    numtimes = 0
    for k in range(0, 100+numsamppc):
        rvalo = np.random.normal(0, 1, thetac.shape)
        rval = np.sqrt(2) * adjrho * (rvalo @ hc)
        thetap = thetac + rval

        if logpostf_grad is not None:
            diffval = (adjrho ** 2) * (dfval @ covmat0)
            thetap += diffval
            fvalp, dfvalp = logpostf(thetap)
            fvalp = fvalp / temps
            dfvalp = dfvalp / temps
            term1 = rvalo / np.sqrt(2)
            term2 = (adjrho / 2) * ((dfval + dfvalp) @ hc)
            qadj = -(2 * np.sum(term1 * term2, 1) + np.sum(term2**2, 1))
        else:
            fvalp = logpostf_nograd(thetap)
            fvalp = fvalp / temps
            qadj = np.zeros(fvalp.shape)
        swaprnd = np.log(np.random.uniform(size=fval.shape[0]))
        whereswap = np.where(np.squeeze(swaprnd)
                             < np.squeeze(fvalp - fval)
                             + np.squeeze(qadj))[0]
        if whereswap.shape[0] > 0:
            numtimes = numtimes + (whereswap.shape[0]/numchain)
            thetac[whereswap, :] = 1*thetap[whereswap, :]
            fval[whereswap] = 1*fvalp[whereswap]
            if logpostf_grad is not None:
                dfval[whereswap, :] = 1*dfvalp[whereswap, :]
        for rt in range(1,numchain):
            rhoh = temps[rt-1]/temps[rt]
            if((fval[rt-1]*(rhoh-1)+fval[rt]*(1/rhoh-1))>
                np.log(np.random.uniform(size=1))):
                fvaltemp = temps[rt-1]/temps[rt] * fval[rt - 1]
                fval[rt-1] = temps[rt]/temps[rt-1] * fval[rt]
                fval[rt] = 1*fvaltemp
                thetatemp = 1*thetac[rt-1,:]
                thetac[rt-1,:] = 1*thetac[rt,:]
                thetac[rt,:] = 1*thetatemp
                if logpostf_grad is not None:
                    dfvaltemp = temps[rt- 1]/temps[rt ]  * dfval[rt - 1,:]
                    dfval[rt-1,:] = temps[rt ] / temps[rt- 1] * dfval[rt,:]
                    dfval[rt,:] = 1*dfvaltemp
        if (k < tunesteps) and (k % 5 == 0):
            tau = tau + 1 / np.sqrt(1 + k/5) * \
                  ((numtimes / 5) - taracc)
            rho = 2 * (1 + (np.exp(2 * tau) - 1) / (np.exp(2 * tau) + 1))
            adjrho = rho*(temps**(1/3))
            numtimes = 0
        elif(k >= tunesteps):
            thetasave[:, k-tunesteps, :] = 1 * thetac

    thetasave = np.reshape(thetasave[40:thetasave.shape[1],:,:], (-1, thetac.shape[1]))
    theta = thetasave[np.random.choice(range(0, thetasave.shape[0]),
                                       size=numsamp), :]
    sampler_info = {'theta': theta, 'logpost': logpostf_nograd(theta)}

    return sampler_info

def shrinkaroundcenter(theta, lpostf):
    theta0 = 1*theta
    keepgoing = True
    theta0 = np.unique(theta0, axis=0)
    iteratttempt = 0
    while keepgoing:
        logpost = lpostf(theta0)/4
        mlogpost = np.max(logpost)
        logpost -= (mlogpost + np.log(np.sum(np.exp(logpost - mlogpost))))
        post = np.exp(logpost)
        post = post/np.sum(post)
        thetaposs = theta0[np.random.choice(range(0, theta0.shape[0]),
                                            size=1000,
                                            p=post.reshape((theta0.shape[0],
                                                            ))), :]
        if np.any(np.std(thetaposs, 0) < 10 ** (-8) * np.min(np.std(theta0,
                                                                    0))):
            thetastar = theta0[np.argmax(logpost), :]
            theta0 = thetastar + (theta0 - thetastar) / 2
            iteratttempt += 1
        else:
            theta0 = thetaposs
            keepgoing = False
        if iteratttempt > 10:
            raise ValueError('Could not find any points to vary.')
        return theta0
