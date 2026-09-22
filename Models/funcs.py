import numpy as np
import itertools
import functools
from numpy import random as random

rng = random.default_rng()

def corrs(sigma):
    ''' 
    Input:
        - sigma: a (B,N) matrix of the response of N neurons at B time indices, where sigmat[b,n] corresponds to the response of the nth neuron at time b

    Output:
        - corrs: a (B,N*(N-1)/2) matrix of the correlation of the N neurons at B indices, for all corrs[b,i*j/2] where i < j
    '''

    return jnp.apply_along_axis(func1d=lambda x : jnp.outer(x,x)[jnp.tril_indices(x.shape[0]-1)],axis=1,arr=sigma)

def pt(sigma,**X):
    '''
    Input:
        - sigma: a (N+N*(N-1)/2,) vector of the response of N neurons at a particular time t, where sigma[i] corresponds to the response of the ith neuron, concatenated with the cross-correlations of each neuron at that time
            - sigma can also be (B,N+N*(N-1)/2)
        - X: a (N + N*(N-1)/2 , ) vector containing:
            - h: (B*N,) vector of the time-dependent field, where h[N*t + n] corresponds to the time dependent field of the nth neuron at time t
            - J: (N(N-1)/2,) vector of the fixed couplings between two neurons

    Output:
        pt: the probability of that state at that time, given the parameters
    '''
    
    if 'X' in X.keys():
        X_ = X['X']
    else:
        X_ = jnp.concatenate((X['h'],X['J']))
    
    if 'Z' in X.keys():
        Z_ = X['Z']
    else:
        Z_ = 1

    pt = jnp.exp(-1*jnp.matmul(sigma,X_)) / Z_
    
    return pt

def observables(sigma):
    '''
    Input:
        - sigma: an (B,N) vector of binarized neural responses at time b; sigmab[n] is the response of neuron n
    
    Output:
        - observables: (B, N + N*(N-1)/2) vector of the observables at that point in time
    '''
    corr = corrs(sigma)
    return jnp.concatenate((sigma,corr),axis=1)

def P_bar(sigma):
    ''' 
    Input:
        sigma: a (B,N) matrix of binarized neural responses, where sigma[b,n] is the response of neuron n at time b

    Output:
        P_b: a (N + N*(N-1)/2 , ) vector of the average of the observables given the neural data
    '''
    P_b = observables(sigma).mean(axis=0)

    return P_b

def Q(X,combs_obs,Z):
    '''
    Input:
        - X: a (N + N*(N-1)/2 , ) vector of the parameters of the model
    
    Output:
        Q: a (N + N*(N-1)/2 , ) vector of the model averages of the observables.
    '''

    weighted_observables = combs_obs*pt(combs_obs,X=X).reshape(-1,1)

    Q = weighted_observables.sum(axis=0)

    return Q

def QMC(sigma,M):
    '''
    Input:
        - sigma: a (B,N) matrix of binarized neural responses, where sigma[b,n] is the response of neuron n at time b
        - M: the number of times to sample from sigma
    
    Output:
        QMC: a montecarlo approximation of Q
    '''
    B,N = sigma.shape
    MC = rng.choice(sigma,M,replace=True)
    QMC = observables(MC)
    QMC = QMC.mean(axis=0)

    return QMC

def susc_bar(sigma):
    '''
    Input:
        - sigma: a (B,N) matrix of binarized neural responses, where sigma[b,n] is the response of neuron n at time b
    
    Output:
        - X_bar: a (D,D) matrix consisting of the mean of the products of observables, subtracted from the product of the mean of each corresponding observable
            where D = N + N*(N-1)/2
    '''
    B,N = sigma.shape
    all_obs = observables(sigma)
    obs_prod_bar = jnp.apply_along_axis(lambda x: jnp.outer(x,x),axis=1,arr=all_obs).mean(axis=0)
    P_ = all_obs.mean(axis=0)
    prod_obs_bar = jnp.outer(P_,P_)
    return obs_prod_bar - prod_obs_bar

def epsilon(P_bar,inv_sus,Q,B):
    D = P_bar.shape[0]
    diff = P_bar - Q
    return jnp.sqrt(jnp.abs((2*B / D)*(diff@inv_sus@diff)))

def all_combs_obs(N):
    combs = jnp.array(np.fromiter(itertools.product(range(2),repeat = N),dtype=np.dtype((jnp.float32,N)),count=2**N))

    combs_obs = observables(combs)

    return combs_obs

def Z(X,combs_obs):
    p_ = pt(combs_obs,X=X)
    return p_.sum()