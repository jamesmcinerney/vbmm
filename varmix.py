'''
Created on 14 May 2013

@author: James McInerney
'''

#implementation of variational Gaussian mixture models

from numpy import *
from matplotlib.pyplot import *
from numpy.linalg.linalg import inv, det
from scipy.special.basic import digamma
import time
from viz import create_cov_ellipse

ROOT = 'PATH_TO_WHERE_YOU_WANT_YOUR_ANIMATIONS_TO_BE_SAVED_TO'

def gen(K,N,XDim):
    #K: number of components
    #N: number of data points
    
    mu = array([random.multivariate_normal(zeros(XDim),10*eye(XDim)) for _ in range(K)])
    cov = [0.1*eye(XDim) for _ in range(K)]
    q = random.dirichlet(ones(K)) #component coefficients
    X = zeros((N,XDim)) #observations
    Z = zeros((N,K)) #latent variables
    for n in range(N):
        #decide which component has responsibility for this data point:
        Z[n,:] = random.multinomial(1, q)
        k = Z[n,:].argmax()
        X[n,:] =  random.multivariate_normal(mu[k,:],cov[k])
    return X


def run(X,K,VERBOSE=True):
    #X: observations
    (N,XDim) = shape(X)
    
    #hyperparams:
    alpha0 = 0.1 #prior coefficient count (for Dir)
    beta0 = (1e-20)*1. #variance of mean (smaller: broader the means)
    v0 = XDim+1. #2. #degrees of freedom in inverse wishart
    m0 = zeros(XDim) #prior mean
    W0 = (1e0)*eye(XDim) #prior cov (bigger: smaller covariance)
    
    #params:
    #Z = ones((N,K))/float(K) #uniform initial assignment
    Z = array([random.dirichlet(ones(K)) for _ in range(N)])

    ion()    
    fig = figure(figsize=(10,10))
    ax_spatial = fig.add_subplot(1,1,1) #http://stackoverflow.com/questions/3584805/in-matplotlib-what-does-111-means-in-fig-add-subplot111
    circs = []
                
    
    itr, max_itr = 0, 200
    while itr < max_itr:
        #M-like-step
        NK = Z.sum(axis=0)
        vk = v0 + NK + 1.
        xd = calcXd(Z,X)
        S = calcS(Z,X,xd,NK)
        betak = beta0 + NK
        m = calcM(K,XDim,beta0,m0,NK,xd,betak)
        W = calcW(K,W0,xd,NK,m0,XDim,beta0,S)

        #E-like-step
        mu = Muopt(X,XDim,NK,betak,m,W,xd,vk,N,K) #eqn 10.64 Bishop
        invc = Invcopt(W,vk,XDim,K) #eqn 10.65 Bishop
        pik = Piopt(alpha0,NK) #eqn 10.66 Bishop
        Z = Zopt(XDim, pik, invc, mu, N, K) #eqn 10.46 Bishop
        
        if VERBOSE:
            print 'itr %i'%itr
            print 'means',m
            print 'Z',Z
            print 'mu',mu
            print 'invc',invc
            print 'exp(pik)',exp(pik)
            print 'NK',NK
            if itr==0:
                sctX = scatter(X[:,0],X[:,1])
                sctZ = scatter(m[:,0],m[:,1],color='r')
            else:
                #ellipses to show covariance of components
                for circ in circs: circ.remove()
                circs = []
                for k in range(K):
                    circ = create_cov_ellipse(S[k], m[k,:],color='r',alpha=0.3) #calculate params of ellipses (adapted from http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals)
                    circs.append(circ)
                    #add to axes:
                    ax_spatial.add_artist(circ)
                    #make sure components with NK=0 are not visible:
                    if NK[k]<=alpha0: m[k,:] = m[NK.argmax(),:] #put over point that obviously does have assignments
                sctZ.set_offsets(m)
            draw()
            #time.sleep(0.1)
            savefig(ROOT + 'animation/%04d.png'%itr)
        itr += 1
    
    if VERBOSE:
        #keep display:    
        time.sleep(360)
    
    return m,invc,pik,Z
    
    
def calcXd(Z,X):
    #weighted means (by component responsibilites)
    (N,XDim) = shape(X)
    (N1,K) = shape(Z)
    NK = Z.sum(axis=0)
    assert N==N1
    xd = zeros((K,XDim))
    for n in range(N):
        for k in range(K):
            xd[k,:] += Z[n,k]*X[n,:]
    #safe divide:
    for k in range(K):
        if NK[k]>0: xd[k,:] = xd[k,:]/NK[k]
    
    return xd

def calcS(Z,X,xd,NK):
    (N,K)=shape(Z)
    (N1,XDim)=shape(X)
    assert N==N1
    
    S = [zeros((XDim,XDim)) for _ in range(K)]
    for n in range(N):
        for k in range(K):
            B0 = reshape(X[n,:]-xd[k,:], (XDim,1))
            L = dot(B0,B0.T)
            assert shape(L)==shape(S[k]),shape(L)
            S[k] += Z[n,k]*L
    #safe divide:
    for k in range(K):
        if NK[k]>0: S[k] = S[k]/NK[k]
    return S

def calcW(K,W0,xd,NK,m0,XDim,beta0,S):
    Winv = [None for _ in range(K)]
    for k in range(K): 
        Winv[k]  = inv(W0) + NK[k]*S[k]
        Q0 = reshape(xd[k,:] - m0, (XDim,1))
        q = dot(Q0,Q0.T)
        Winv[k] += (beta0*NK[k] / (beta0 + NK[k]) ) * q
        assert shape(q)==(XDim,XDim)
    W = []
    for k in range(K):
        try:
            W.append(inv(Winv[k]))
        except linalg.linalg.LinAlgError:
            print 'Winv[%i]'%k, Winv[k]
            raise linalg.linalg.LinAlgError()
    return W

def calcM(K,XDim,beta0,m0,NK,xd,betak):
    m = zeros((K,XDim))
    for k in range(K): m[k,:] = (beta0*m0 + NK[k]*xd[k,:]) / betak[k]
    return m    

def Muopt(X,XDim,NK,betak,m,W,xd,vk,N,K):
    Mu = zeros((N,K))
    for n in range(N):
        for k in range(K):
            A = XDim / betak[k] #shape: (k,)
            B0 = reshape((X[n,:] - m[k,:]),(XDim,1))
            B1 = dot(W[k], B0)
            l = dot(B0.T, B1)
            assert shape(l)==(1,1),shape(l)
            Mu[n,k] = A + vk[k]*l #shape: (n,k)
    
    return Mu

def Piopt(alpha0,NK):
    alphak = alpha0 + NK
    pik = digamma(alphak) - digamma(alphak.sum())
    return pik

def Invcopt(W,vk,XDim,K):
    invc = [None for _ in range(K)]
    for k in range(K):
        dW = det(W[k])
        print 'dW',dW
        if dW>1e-30: ld = log(dW)
        else: ld = 0.0
        invc[k] = sum([digamma((vk[k]+1-i) / 2.) for i in range(XDim)]) + XDim*log(2) + ld
    return invc
        
def Zopt(XDim, exp_ln_pi, exp_ln_gam, exp_ln_mu, N, K):
    Z = zeros((N,K)) #ln Z
    for k in range(K):
        Z[:,k] = exp_ln_pi[k] + 0.5*exp_ln_gam[k] - 0.5*XDim*log(2*pi) - 0.5*exp_ln_mu[:,k]
    #normalise ln Z:
    Z -= reshape(Z.max(axis=1),(N,1))
    Z1 = exp(Z) / reshape(exp(Z).sum(axis=1), (N,1))
    return Z1
    
if __name__ == "__main__":
    #generate synthetic data:
    X = gen(30,200,2)
    
    #run VB on the data:
    K1 = 20 # num components in inference
    mu,invc,pik,Z = run(X,K1)
    print 'mu',mu
    print 'NK',Z.sum(axis=0)
    
