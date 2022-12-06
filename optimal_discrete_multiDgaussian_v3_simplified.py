# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 09:11:34 2021

@author: turinici

Goal: construct an optimal, in the energy/Radon-Sobolev sense, discrete measure 
which is the sum of K Dirac masses that minimizes the distance to a N-dim
Gaussian

Notations : N = dim of the Gaussian
            K = number of Dirac masses
            alpha_k = coefficients of \delta_{X_k}
            p_k : \alpha_k = e^{p_k} / \sum_l e^{p_l}

REMARK: this can work also for other multi-dimensional measures, 
not only gaussian, but adapted function have to be used.

Reference: Radon-Sobolev distance is that in paper 
https://arxiv.org/abs/1911.13135

Implementation notes: we use a gradient flow of the distance to a N-dim Gaussian

at this moment the alphas are not optimized but set uniform constante = 1/K
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from scipy.special import gamma
from scipy.special import loggamma
from scipy.stats import poisson
from scipy.integrate import odeint
from scipy.optimize import check_grad  
#import seaborn as sns
import sys

from scipy.stats import norm as stats_norm
#%matplotlib auto


nr_args=len(sys.argv)
arg_list=sys.argv



N=23#dimension of the Gaussian
K=23#number of points
#N,K=11,5#this case is more computer intensive

N,K=2,17
N,K=2,10

############################
#
# General routines
#
###########################
def save_last_fig_with_label(label='none'):
    plt.savefig(label+'.pdf')
    plt.savefig(label+'.jpg')
    

###########################
#
#   Radon-Sobolev routines
#
###########################


exact_xi=True 
# if set to false use an approximation as in the paper: sqrt(x**2+c1)+c0
#we obtain qualitatively same solutions, although the distance is not said 
# to be the same !
#exact_xi=False# note: sometime we obtain errors in ODE because time is too long, 
# set a smaller time if necessary

def mean_distance_to_normal_poisson(x,D=8):
    ''' 
    compute mean distance to a normal variable using energy metric
    and the poisson distribution using formula from 
    https://arxiv.org/abs/1911.13135
    '''
#    if(np.abs(x) <=1.e-15):
#        nr_terms=1#for x=0 one term is enough

    #only take relevant indices i.e. those that are ok for the poisson distrib
    i1,i2=poisson.interval(0.999999999999999,mu=x*x/2)
    indices=np.array(range(int(i1),int(i2)+1))
    proba_poisson=poisson.pmf(k=indices,mu=x*x/2)

    values = np.exp(loggamma(D/2+0.5+indices)- loggamma(D/2+indices))
    
    return np.sqrt(2.)*np.sum(values*proba_poisson)

print('mean distance squared among K-samples',
      mean_distance_to_normal_poisson(0,N)/K/np.sqrt(2))

def radon_sobolev_distance_to_normal_sq_poisson(x,D=8):
    if(exact_xi):
        return mean_distance_to_normal_poisson(x,D)-\
               mean_distance_to_normal_poisson(0,D)/np.sqrt(2)
    else:
        xi0pp=np.exp(loggamma(D/2+0.5)- loggamma(1.+D/2.))/np.sqrt(2.)
        xi0=mean_distance_to_normal_poisson(0,D)    
        c0,c1=xi0-1/xi0pp,1/(xi0pp**2)
        return np.sqrt(x*x+c1)+c0

#Next function is ony used to check the gradient
def grad_mean_distance_to_normal_poisson(x,D=8):
    ''' 
    gradient of the distance to a normal variable using energy metric
    '''
    return x*(mean_distance_to_normal_poisson(x,D+2)
              -mean_distance_to_normal_poisson(x,D)
              )

def grad_rs_dist_to_normal_over_x(x,D=8):
    ''' 
    with notations from paper, this function computes : xi'(x)/x
    If approximation is to be used: use as xi = sqrt(x**2+c1)+c0 and return
    1/sqrt(x**2+c1)
    '''
    if(exact_xi):
        return (mean_distance_to_normal_poisson(x,D+2)
                -mean_distance_to_normal_poisson(x,D))
    else:
        xi0pp=np.exp(loggamma(D/2+0.5)- loggamma(1.+D/2.))/np.sqrt(2.)
        c1=1/(xi0pp**2)
        return 1./np.sqrt(x*x+c1)

#for testval in [0,10,38,50,150,250.]:
#   print(testval,mean_distance_to_normal_poisson(testval),
#         grad_mean_distance_to_normal_poisson(testval))


def distance_sq_to_normal_from_discrete_distrib(points,alphas=None):
    '''
    Parameters
    ----------
    points : N x K matrix
        each of the K columns is a point X_k in R^N
    alphas : 1D array of weights of shape [K,1]
        weights of the distribution; default value = uniform = 1/K

    Returns
    -------
    the distance from the distribution sum_k alpha_k X_k to a N-dim normal
    
    The distance squared is computed as 
    \sum_k alpha_k xi(|X_k|) - (1/2)\sum_{k,l} |Xk-Xl|

    '''
    dim,cK=points.shape
    if alphas is None:
        alphas = np.ones((cK))/cK
    nK,=alphas.shape
    assert(nK==cK)
    
    norms= np.linalg.norm(points,ord=2,axis=0)#norm of each X_k
    xi_of_Xk = np.array([radon_sobolev_distance_to_normal_sq_poisson(xkn,dim) 
                         for xkn in norms ])

    #matrix of distances
#    distXX = np.linalg.norm(np.expand_dims(points, 2)
#                            -np.expand_dims(points, 1),ord=2,axis=0)    
    distXX = np.linalg.norm(points[:,:,None]-points[:,None,:],ord=2,axis=0)
    #implementation note: with notation deltaxkl=points[:,:,None]-points[:,None,:]
    #then deltaxkl[:,k,l] = points[:,k]-points[:,l]
    
    return alphas@xi_of_Xk - 0.5*alphas@distXX@alphas

#test  distance_sq_to_normal_from_discrete_distrib(np.random.randn(N,K))


def grad_dist_sq_to_normal_from_discrete_distrib(points,alphas=None,
                                                 output_parts=False):
    '''
    Parameters
    ----------
    points, alphas :same as in distance_sq_to_normal_from_discrete_distrib
    output_parts : if set to True then return a triplet of gradient, 
                    term1 and term2
    Returns
    -------
    the matrix of gradients
    
    Computation details: 
        gradient with respect to X_k is alpha_k X_k \cdot xi'(|X_k|)'/|X_k|  - 
       \frac{X_k-X_l}{|X_k-X_l|} sum_{l \neq k} alpha_k alpha_l

    '''
    dim,cK=points.shape
    if alphas is None:
        alphas = np.ones((cK))/cK
    nK,=alphas.shape
    assert(nK==cK)


    norms= np.linalg.norm(points,ord=2,axis=0)#norm of each X_k
    #matrix of distances
    deltaxkl=points[:,:,None]-points[:,None,:]#dimension N,K,K
    distXX = np.linalg.norm(deltaxkl,ord=2,axis=0)

    #smoothing part : put small term everywhere to be able to divide by it    
    # small_eps=1.e-16
    #smooth_distXX = np.maximum(distXX,small_eps)
    #put 1.0 on the diagonal
    smooth_distXX = distXX
    np.fill_diagonal(smooth_distXX, 1.0)
    #compute Xk_Xl/smoothed_norm times alpha_k alpha_l
    Xk_minus_Xl_over_sm_norm = deltaxkl/smooth_distXX[None,:,:]
    Xk_minus_Xl_over_sm_norm *= (alphas[None,:,None]*alphas[None,None,:])
    # note: the 0.5 term dissapears in the gradient because there are 
    # two terms dist(Xk,Xl) and dist(Xl,Xk)

    alphaxikoverx = np.array([alphak*grad_rs_dist_to_normal_over_x(xkn,dim) 
                     for alphak,xkn in zip(alphas,norms)])

    grad_term_xi=alphaxikoverx[None,:]*points#part concerning xi(X_k)
    grad_term_dist=np.sum(Xk_minus_Xl_over_sm_norm,axis=2)
    if(output_parts):
        return grad_term_xi- grad_term_dist,grad_term_xi,grad_term_dist
    else:
        return grad_term_xi- grad_term_dist

grad_code_check=False
#grad_code_check=True
if(grad_code_check): 
    checkdim=217
    Xt=np.random.randn(checkdim,3)#each point of a column of the NxK matrix
#   Xt[:,2]=Xt[:,0]+1.e-18*np.random.randn(217)
    full_grad,term_xi_grad,term_dist_grad = \
    grad_dist_sq_to_normal_from_discrete_distrib(Xt,output_parts=True)
    Vx=Xt[:,0]
    Vy=Xt[:,1]
    Vz=Xt[:,2]
    
    print(np.linalg.norm(term_dist_grad[:,0]- (Vx-Vy)/np.sqrt(np.sum((Vx-Vy)**2))/(K*K)
        -(Vx-Vz)/np.sqrt(np.sum((Vx-Vz)**2))/(K*K)))
    print(np.linalg.norm(term_dist_grad[:,1]- (Vy-Vz)/np.sqrt(np.sum((Vz-Vy)**2))/(K*K)
        -(Vy-Vx)/np.sqrt(np.sum((Vx-Vy)**2))/(K*K)))
    print(np.linalg.norm(term_dist_grad[:,2]- (Vz-Vy)/np.sqrt(np.sum((Vz-Vy)**2))/(K*K)
        -(Vz-Vx)/np.sqrt(np.sum((Vx-Vz)**2))/(K*K)))
    normVx = np.sqrt(np.sum(Vx**2))
    grad_dn_Vx  = grad_mean_distance_to_normal_poisson(normVx,checkdim)
    grad_dn_Vx2  = grad_rs_dist_to_normal_over_x(normVx,checkdim)
    
    print(np.linalg.norm(term_xi_grad[:,0]- Vx*grad_dn_Vx/normVx/K))
    print(np.linalg.norm(term_xi_grad[:,0]- Vx*grad_dn_Vx2/K))

    #print(normVx*grad_rs_dist_to_normal_over_x(normVx,checkdim) - 
    # grad_mean_distance_to_normal_poisson(normVx,checkdim))

#%%
######################################
#
#         check Radon-Sobolev gradient
#
######################################
check_gradient=False
#check_gradient=True
if(check_gradient):
    X0=np.random.randn(N,K)
    check_grad_result=check_grad(
        lambda x: distance_sq_to_normal_from_discrete_distrib(x.reshape(N,K)),
        lambda x: grad_dist_sq_to_normal_from_discrete_distrib(
            x.reshape(N,K)).flatten(),X0.flatten() )
    print('check grad result=',check_grad_result)    




#%%
######################################
#
#         Grad flow version
#
######################################
grad_flow_version=False
explicit_grad_flow_steps=False
#grad_flow_version=True
#explicit_grad_flow_steps=True
if(grad_flow_version):
    #initialization of points
    X=np.random.randn(N,K)#each point of a column of the NxK matrix
    
    
    #this first implementation keeps alphas constant = 1/K
    
    distrib_weights = np.ones((K))/K #uniform
    time_step=50.
    time_step=10.0
    flow_times=[0.0]
    current_time=0.0
    #plot_or_not=False
    
    time_step,nosteps=0.001,2000
    time_step,nosteps=0.01,500
    #nosteps=10
    
    allX=np.zeros( (nosteps+1,) + X.shape)# to record all iterations
    allX[0,:,:]=X
    
    iter_distances=[distance_sq_to_normal_from_discrete_distrib(allX[0,:,:])]
    print('Initial distance=',iter_distances[0])
    variance_distances=[0]    
    #if(plot_or_not):
    #plt.figure(1,figsize=(10,7))
    
    for kstep in range(nosteps):
        current_time +=time_step
        flow_times.append(current_time)
        if(explicit_grad_flow_steps):
            Xc=allX[kstep,:,:]
            distXc=distance_sq_to_normal_from_discrete_distrib(Xc)
            tmp_grad=grad_dist_sq_to_normal_from_discrete_distrib(Xc)
            norm_grad=np.linalg.norm(tmp_grad)
            print('norm grad=',norm_grad)
            allX[kstep+1,:,:] = allX[kstep,:,:] - \
            time_step*distXc*tmp_grad/(norm_grad**2+1e-14)
        else:
            oldX=allX[kstep,:,:]
            newX1=oldX-time_step*grad_dist_sq_to_normal_from_discrete_distrib(oldX)
            newX2=oldX-time_step*grad_dist_sq_to_normal_from_discrete_distrib(newX1)
            iter_count=1
            while (np.linalg.norm(newX1-newX2)>1.e-6):
                newX1=newX2
                newX2=oldX-time_step*grad_dist_sq_to_normal_from_discrete_distrib(newX1)
                iter_count +=1
            allX[kstep+1,:,:]=newX2.copy()    
        #possible TODO: limit implicit iterations to some max_implicit_iter                                                            

        Xfinal=allX[kstep+1,:,:].copy()

        iter_distances.append(distance_sq_to_normal_from_discrete_distrib(
                Xfinal))
        print('Time=',current_time,' distance=',iter_distances[-1])
        variance_distances.append(np.var(iter_distances))

        plt.figure(3,figsize=(10,5))
        plt.clf()
    
        plt.subplot(1,2,1)
        first_step=np.maximum(int(kstep/2),kstep-10)
        plt.plot(allX[first_step:kstep+1,0,:], 
                 allX[first_step:kstep+1,1,:], '-go', label='discrete')#green dots
    #    plt.figure(1)
        plt.subplot(1,2,2)
        plt.loglog(flow_times,iter_distances,'-b*')
        plt.title('N='+str(N)+' K='+str(K)+' last distance='+str(iter_distances[-1]))
        plt.show()
#        save_last_fig_with_label('result_N'+str(N)+'K'+str(K)+'time'
#                                 +str(np.round(flow_times[-1],2)))

        #represent as brownian motion
        represent_as_brownian=False 
        if(represent_as_brownian):
            plt.figure(32,figsize=(10,5))
            plt.clf()        
            W=np.zeros((N+1,K)) 
            W[0,:]=0
            W[1:,:]=0+np.cumsum(Xfinal,0)
            plt.plot(W)
            plt.xlabel('dimension')
            dataframeXfinal=pd.DataFrame(Xfinal.T)
            tmpfig=dataframeXfinal.hist(figsize = (10,10))
            plt.title('Xfinal')
            dataframeW=pd.DataFrame(W.T)
            _=dataframeW.hist(figsize = (10,10))
            plt.title('Brownian')
#            _=dataframeW.hist(bins=30, figsize = (10,10))
        plt.pause(.5)


    Xfinal=allX[-1,:,:]  
    variance_distances=[0]
    for kstep in range(nosteps):
        variance_distances.append(np.var(iter_distances[0:kstep+2]))

    plt.figure(34,figsize=(10,5))
    plt.clf()
    plt.loglog(flow_times,variance_distances,'-b*')

#%%

#######################################
#
#ode version
#
#######################################
#ode_version=False
ode_version=True
iso_or_exp_flow='iso'#can be 'iso' or 'exp': for 'iso' progress is linear, 
                     #for 'exp' is exponential
iso_or_exp_flow='exp'#can be 'iso' or 'exp'

if(ode_version):

    def iso_grad_flow(X,t):
        '''
        Define the gradient flow: attention this is an iso-improvement flow i.e.
        time is rescaled to have constant improvement in the overall function to
        minimize. In practice this means dividing by the square of the norm of the 
        gradient.
        '''
        Xr = X.reshape(N,K)
        gradX=grad_dist_sq_to_normal_from_discrete_distrib(Xr)
        norm_grad=np.linalg.norm(gradX)        
        return -gradX.reshape(N*K)/(norm_grad**2+1.e-8)

    def exp_grad_flow(X,t):
        '''
        Define the gradient flow: to have exponential improvement 
        in the overall function to minimize. In practice this means 
        dividing by the square of the norm of the 
        gradient and multiplying by the value of the function to minimize.
        '''
        Xr = X.reshape(N,K)
        gradX=grad_dist_sq_to_normal_from_discrete_distrib(Xr)
        norm_grad=np.linalg.norm(gradX)
        dist_value=distance_sq_to_normal_from_discrete_distrib(Xr)        
        return -dist_value*gradX.reshape(N*K)/(norm_grad**2+1.e-14)

    grad_flow_dict={'iso':iso_grad_flow,'exp':exp_grad_flow}
    #choose the exponential version, more stable
    grad_flow=grad_flow_dict['exp']
      

    X0=np.random.randn(N,K)#each point of a column of the NxK matrix
    Tfinal=500*K


    if(iso_or_exp_flow=='iso'):
        if(exact_xi):
            Tfinal=distance_sq_to_normal_from_discrete_distrib(X0)*0.9
        else:#if not exact then distance is not well computed
            Tfinal=distance_sq_to_normal_from_discrete_distrib(X0)*0.5
    else:#exponential progress towards minimum
        Tfinal=10.0

    Npas=100
    
    trange = np.linspace(0, Tfinal, Npas+1,endpoint=True)
#    sol,infodict = odeint(grad_flow, X0.reshape(N*K), trange,rtol=1.e-1,atol=1.e-1,
#                          full_output=True)
#    sol,infodict = odeint(grad_flow, X0.reshape(N*K), trange,full_output=True)
#    iter_distances_ode=[distance_sq_to_normal_from_discrete_distrib(sol[ki,:].reshape(N,K))
#                    for ki in range(Npas+1)]

    #gradual resolution
    Xinit = X0.reshape(N*K)
    iter_distances_ode=[distance_sq_to_normal_from_discrete_distrib(X0)]
    allX=[]
    for cstep in range(Npas):
        sol,infodict = odeint(grad_flow, Xinit, trange[cstep:cstep+2],full_output=True)
        Xinit=sol[-1,:]
        allX.append(Xinit)
        tmpdist=distance_sq_to_normal_from_discrete_distrib(sol[-1,:].reshape(N,K))
        iter_distances_ode.append(tmpdist)
        print('time step=',trange[cstep+1],' distance=',tmpdist)



        Xfinal=sol[-1,:].reshape(N,K)
        plt.figure(4,figsize=(10,5))
        plt.clf()
        plt.subplot(1,2,1)
        xdata=Xfinal[0,:]
        if(N>=2):
            ydata=Xfinal[1,:]
        else:
            ydata=xdata*0
            final_quantiles=np.sort(stats_norm.cdf(xdata))
        plt.plot(xdata,ydata, 
                 'go', label='discrete')  # green dots
        plt.subplot(1,2,2)
    #    plt.loglog(trange,iter_distances_ode,'-o')
        plt.semilogy(trange[0:len(iter_distances_ode)],iter_distances_ode,'-o')
        plt.title('last distance='+str(iter_distances_ode[-1]))
        plt.xlabel('flow time (t)')
        plt.ylabel('distance')    
        plt.show()
        plt.tight_layout()
        plt.pause(0.5)
        save_last_fig_with_label('ode_result_N'+str(N)+'K'+str(K)+'cstep'
                                 +str(cstep))


        if(N>=2):            
            if(N==2):
                plt.figure(44,figsize=(5,5))
                plt.clf()
                xdata=Xfinal[0,:]
                ydata=Xfinal[1,:]
                plt.plot(xdata,ydata, 
                     'go', label='discrete')  # green dots
                save_last_fig_with_label('fig44_ode_result_N'+str(N)+'K'+str(K)+'cstep'
                                     +str(cstep))    
            else:#plot brownian version
                plt.figure('brownian',figsize=(5,5))
                plt.clf()
                brownian_trajectories1=np.cumsum(Xfinal,axis=0)
                brownian_trajectories=np.vstack((np.zeros((1,K)),brownian_trajectories1))
                plt.plot(brownian_trajectories, 
                     '-', label='discrete')  # green dots
                save_last_fig_with_label('brownian_ode_result_N'+str(N)+'K'+str(K)+'cstep'
                                     +str(cstep))
    
        np.savez_compressed('Xfinal_ode_N'+str(N)+'K'+str(K)+'cstep'
                                     +str(cstep)+'.npz',Xfinal)
            
                        
        #test
        if np.abs(iter_distances_ode[-1]-iter_distances_ode[-2])<1.e-15:
            break

np.savez_compressed('run_N'+str(N)+'K'+str(K)+'.npz',allX,iter_distances_ode,Npas)
print('best distance N=',N,' K=',K,'dist=',iter_distances_ode[-1])
