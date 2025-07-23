#12/05/2025
#Giuseppe Speciale


import numpy as np
import er_graph as er
import os
import pickle
import dist_plot_ as dp

np.random.seed(10)
os.system('cls')
debug = 1
save = 0
save_name = 'saved_pickles\zo1_step_2_02.pkl' # REMEMBER TO CHANGE NAME OR ELSE IT REWRITES EVERYTHING

ll = 1 # Tradeoff for Aggregative f.
gg = 10
d = [0,0] # Distance between agents in cost

def cost_new(x,r,s):
    return (gg/2)*np.linalg.norm(x-r)**2 + 0.5*np.linalg.norm(x-s)**2

def grad1n(x,r,s,a,iter):
    return gg*(x-r) + (x-s)
"""
def grad1n(x,r,s,a):
    return gg*(x-r) + (x-s)*(1-1/(2*NN*s))
"""
def grad2n(x,r,s,a,iter):
    return s-x


def grad_s(a):
    return 1/NN


def tzo1(x,d,s,a,iter,ag,sr = 0.0001):
    #u = np.random.uniform(-1, 1, dd)
    update = 0.5*(cost_new(x + sr*u[iter,ag], d,s-sr*u[iter,ag]) - cost_new(x,d,s))/sr 
    return update

def tzo2(x,d,s,a,iter,ag,sr = 0.0001):
    """ u = np.random.uniform(-1, 1, dd)
    v = np.random.uniform(-1, 1, dd) """
    update = 0.5*(cost_new(x, d,s+ sr*v[iter,ag]) - cost_new(x,d,s- sr*v[iter,ag]))/sr 
    return update

def tzo(x,d,s,a,iter,ag,sr = 0.0001):
    return (cost_new(x+sr*u[iter,ag], d,s+ sr*v[iter,ag]) - cost_new(x,d,s))/sr 

# Generate Network - data is from Section V of the paper
NN = 5
dd = 2
MAX_ITERS = 3500

I_NN = np.identity(NN, dtype=int) 
p_ER = 0.6
Adj, Adj_weighted = er.graph(NN,p_ER)


# Initialization
# r_i are the positions of the targets (fixed)
# x_i are the positions of the targets (moving)


# Agents
alfa = np.zeros((NN, dd))
alfa = np.random.uniform(0.7, 0.8, (NN, dd))

rr = np.random.uniform(-10, 10, (NN, 2))

xx_final = np.random.uniform(-10, 10, (NN, 2))
xx_init = np.random.uniform(-10, 10, (NN, 2))
xx = np.zeros((NN,dd,MAX_ITERS))
xx[:,:,0] = xx_init

costs = np.zeros((MAX_ITERS,4))
grad_norms = np.zeros((MAX_ITERS,4))
sigma = np.zeros((MAX_ITERS,dd))
print(xx_init)

steppp = 0.04
const = 400

# Generate full u, v vectors
# Pre-generate u and v for all iterations
u = np.random.uniform(-1, 1, (MAX_ITERS, NN, dd))
v = np.random.uniform(-1, 1, (MAX_ITERS, NN, dd))

def agg_van(xt, gf1, gf2, AA, cost_plot, grad_plot, ss,stepsize):
    upd1=np.zeros((NN,dd))
    upd2=np.zeros((NN,dd))
    store=np.zeros((NN,dd))
    ut = np.zeros((NN,dd,MAX_ITERS))
    st = np.zeros_like(ut)
    yt = np.zeros_like(ut)
    for ii in range(NN):
        st[ii,:,0] = np.copy(xt[ii,:,0])
        yt[ii,:,0] = gf2(xt[ii,:,0],rr[ii],st[ii,:,0],alfa[ii,:],0)
    ss[0,:] = np.zeros(dd)

    #init
    for ii in range(NN):
        ss[0,:] += xt[ii,:,0] / NN
    for ii in range(NN):
        ut[ii,:,0] = 0
        #ut[ii,:,0] = grad_func(xt[ii,:,0],phi[ii],s[0,:],alfa[ii,:])
        store[ii,:] = gf2(xt[ii,:,0],rr[ii],st[ii,:,0],alfa[ii,:],0)
    #alg
    for kk in range(MAX_ITERS-1):
        plot_grad = np.zeros(dd)
        #gradient
        for ii in range(NN):
            upd1[ii,:], upd2[ii,:] = gf1(xt[ii,:,kk],rr[ii],st[ii,:,kk],alfa[ii,:],kk), gf2(xt[ii,:,kk],rr[ii],st[ii,:,kk],alfa[ii,:],kk)
            #
        for ii in range(NN):
            plot_grad+= grad1n(xt[ii,:,kk],rr[ii],ss[kk,:],alfa[ii,:],0) +\
                        grad_s(alfa[ii,:]) * sum(grad2n(xt[jj,:,kk],rr[jj],ss[kk,:],alfa[jj,:],0) for jj in range(NN))
            

        #gt algo
        ss[kk+1,:] = 0
        for ii in range(NN):
            xt[ii,:,kk+1] = xt[ii,:,kk] - stepsize * (upd1[ii,:]+ xt[ii,:,kk]*yt[ii,:,kk])
            #ut[ii,:,kk+1] = ut[ii,:,kk] - stepsize* a1*(upd1[ii,:] + zt[ii,:,kk] + upd2[ii,:])/(1-alfa[ii,:])
            ss[kk+1,:] += xt[ii,:,kk+1] / NN
        for ii in range(NN):
            st[ii,:,kk+1] = sum(AA[ii,jj] * (st[jj,:,kk]) for jj in range(NN)) + xt[ii,:,kk+1] - xt[ii,:,kk]
        for ii in range(NN):

            
            store[ii,:] = gf2(xt[ii,:,kk+1],rr[ii],st[ii,:,kk+1],alfa[ii,:],kk+1)
            yt[ii,:,kk+1] = sum(AA[ii,jj] * (yt[jj,:,kk])for jj in range(NN)) + store[ii,:] - upd2[ii,:]
            
            cost_plot[kk] += cost_new(xt[ii, :, kk], rr[ii], ss[kk,:])
        """  
        if(kk % 1000 == 0):
            print(f"iter {kk} yt = {sum (yt[jj,0,kk] for jj in range(NN))}")
            print(f"iter {kk} g2 = {sum (upd2[jj,0] for jj in range(NN))}") 
        """    
        """ 
        # diminishing stepsize
        stepsize = steppp * 1/(1+kk*(1/const))
         """
        grad_plot[kk] = np.linalg.norm(plot_grad)
        #ut[:,:,kk+1] = ut[:,:,kk] - sterpsize*update
    
    print(f"final cost = {cost_plot[MAX_ITERS-2]}")
    print(f"final grad = {grad_plot[MAX_ITERS-2]}")
    #print(xt[:,:,MAX_ITERS-1])
    #print(s[MAX_ITERS-2,:])
def agg_nb1(xt, gf1, gf2, AA, cost_plot, grad_plot, ss,stepsize):
    #print(rr)
    upd1=np.zeros((NN,dd))
    upd2=np.zeros((NN,dd))
    store=np.zeros((NN,dd))
    ut = np.zeros((NN,dd,MAX_ITERS))
    st = np.zeros_like(ut)
    yt = np.zeros_like(ut)
    for ii in range(NN):
        st[ii,:,0] = np.copy(xt[ii,:,0])
        yt[ii,:,0] = gf2(xt[ii,:,0],rr[ii],st[ii,:,0],alfa[ii,:],0)
    ss[0,:] = np.zeros(dd)

    #init
    for ii in range(NN):
        ss[0,:] += xt[ii,:,0] / NN
    for ii in range(NN):
        ut[ii,:,0] = 0
        #ut[ii,:,0] = grad_func(xt[ii,:,0],phi[ii],s[0,:],alfa[ii,:])
        store[ii,:] = gf2(xt[ii,:,0],rr[ii],st[ii,:,0],alfa[ii,:],0)
    #alg
    for kk in range(MAX_ITERS-1):
        plot_grad = np.zeros(dd)
        #gradient
        for ii in range(NN):
            upd1[ii,:], upd2[ii,:] = gf1(xt[ii,:,kk],rr[ii],st[ii,:,kk],alfa[ii,:],kk,ii)*u[kk,ii], gf2(xt[ii,:,kk],rr[ii],st[ii,:,kk],alfa[ii,:],kk)
            #
        for ii in range(NN):
            plot_grad+= grad1n(xt[ii,:,kk],rr[ii],ss[kk,:],alfa[ii,:],0) +\
                        grad_s(alfa[ii,:]) * sum(grad2n(xt[jj,:,kk],rr[jj],ss[kk,:],alfa[jj,:],0) for jj in range(NN))
            

        #gt algo
        ss[kk+1,:] = 0
        for ii in range(NN):
            xt[ii,:,kk+1] = xt[ii,:,kk] - stepsize * (upd1[ii,:]+ xt[ii,:,kk]*yt[ii,:,kk])
            #ut[ii,:,kk+1] = ut[ii,:,kk] - stepsize* a1*(upd1[ii,:] + zt[ii,:,kk] + upd2[ii,:])/(1-alfa[ii,:])
            ss[kk+1,:] += xt[ii,:,kk+1] / NN
        for ii in range(NN):
            st[ii,:,kk+1] = sum(AA[ii,jj] * (st[jj,:,kk]) for jj in range(NN)) + xt[ii,:,kk+1] - xt[ii,:,kk]
        for ii in range(NN):

            
            store[ii,:] = gf2(xt[ii,:,kk+1],rr[ii],st[ii,:,kk+1],alfa[ii,:],kk+1)
            yt[ii,:,kk+1] = sum(AA[ii,jj] * (yt[jj,:,kk])for jj in range(NN)) + store[ii,:] - upd2[ii,:]
            
            cost_plot[kk] += cost_new(xt[ii, :, kk], rr[ii], ss[kk,:])
        """  
        if(kk % 1000 == 0):
            print(f"iter {kk} yt = {sum (yt[jj,0,kk] for jj in range(NN))}")
            print(f"iter {kk} g2 = {sum (upd2[jj,0] for jj in range(NN))}") 
        """    
        """ 
        # diminishing stepsize
        stepsize = steppp * 1/(1+kk*(1/const))
         """
        grad_plot[kk] = np.linalg.norm(plot_grad)
        #ut[:,:,kk+1] = ut[:,:,kk] - sterpsize*update
    
    print(f"final cost = {cost_plot[MAX_ITERS-2]}")
    print(f"final grad = {grad_plot[MAX_ITERS-2]}")
    #print(xt[:,:,MAX_ITERS-1])
    #print(s[MAX_ITERS-2,:])
def agg_mod(xt, gf1, gf2, AA, cost_plot, grad_plot, ss,stepsize):
    #print(rr)
    delta1=np.zeros((NN,dd))
    delta2=np.zeros((NN,dd))
    store=np.zeros((NN,dd))
    ut = np.zeros((NN,dd,MAX_ITERS))
    st = np.zeros_like(ut)
    yt = np.zeros_like(ut)
    for ii in range(NN):
        st[ii,:,0] = np.copy(xt[ii,:,0])
        yt[ii,:,0] = gf2(xt[ii,:,0],rr[ii],st[ii,:,0],alfa[ii,:],0,ii)
    ss[0,:] = np.zeros(dd)

    #init
    for ii in range(NN):
        ss[0,:] += xt[ii,:,0] / NN
    for ii in range(NN):
        ut[ii,:,0] = 0
        #ut[ii,:,0] = grad_func(xt[ii,:,0],phi[ii],s[0,:],alfa[ii,:])
        store[ii,:] = gf2(xt[ii,:,0],rr[ii],st[ii,:,0],alfa[ii,:],0,ii)
    #alg
    for kk in range(MAX_ITERS-1):
        plot_grad = np.zeros(dd)
        #gradient
        for ii in range(NN):
            delta1[ii,:] , delta2[ii,:] = gf1(xt[ii,:,kk],rr[ii],st[ii,:,kk],alfa[ii,:],kk,ii), \
                                          gf2(xt[ii,:,kk],rr[ii],st[ii,:,kk],alfa[ii,:],kk,ii),
            #
        for ii in range(NN):
            plot_grad+= grad1n(xt[ii,:,kk],rr[ii],ss[kk,:],alfa[ii,:],0) +\
                        grad_s(alfa[ii,:]) * sum(grad2n(xt[jj,:,kk],rr[jj],ss[kk,:],alfa[jj,:],0) for jj in range(NN))
            

        #gt algo
        ss[kk+1,:] = 0
        for ii in range(NN):
            xt[ii,:,kk+1] = xt[ii,:,kk] - stepsize * (delta1[ii,:]*u[kk,ii] + xt[ii,:,kk]*yt[ii,:,kk])
            #ut[ii,:,kk+1] = ut[ii,:,kk] - stepsize* a1*(upd1[ii,:] + zt[ii,:,kk] + upd2[ii,:])/(1-alfa[ii,:])
            ss[kk+1,:] += xt[ii,:,kk+1] / NN
        for ii in range(NN):
            st[ii,:,kk+1] = sum(AA[ii,jj] * (st[jj,:,kk]) for jj in range(NN)) + xt[ii,:,kk+1] - xt[ii,:,kk]
        for ii in range(NN):

            
            store[ii,:] = gf2(xt[ii,:,kk+1],rr[ii],st[ii,:,kk+1],alfa[ii,:],kk+1,ii)
            yt[ii,:,kk+1] = sum(AA[ii,jj] * (yt[jj,:,kk])for jj in range(NN)) + store[ii,:]*v[kk+1,ii] - delta2[ii,:]*v[kk,ii] 
            
            cost_plot[kk] += cost_new(xt[ii, :, kk], rr[ii], ss[kk,:])
        """  
        if(kk % 1000 == 0):
            print(f"iter {kk} yt = {sum (yt[jj,0,kk] for jj in range(NN))}")
            print(f"iter {kk} g2 = {sum (upd2[jj,0] for jj in range(NN))}") 
        """    
        """ 
        # diminishing stepsize
        stepsize = steppp * 1/(1+kk*(1/const))
         """
        grad_plot[kk] = np.linalg.norm(plot_grad)
        #ut[:,:,kk+1] = ut[:,:,kk] - sterpsize*update
    
    print(f"final cost = {cost_plot[MAX_ITERS-2]}")
    print(f"final grad = {grad_plot[MAX_ITERS-2]}")
    #print(xt[:,:,MAX_ITERS-1])
    #print(s[MAX_ITERS-2,:])
def agg_new(xt, gf1, gf2, AA, cost_plot, grad_plot, ss,stepsize):
    #print(rr)
    delta=np.zeros((NN,dd))
    store=np.zeros((NN,dd))
    ut = np.zeros((NN,dd,MAX_ITERS))
    st = np.zeros_like(ut)
    yt = np.zeros_like(ut)
    for ii in range(NN):
        st[ii,:,0] = np.copy(xt[ii,:,0])
        yt[ii,:,0] = gf2(xt[ii,:,0],rr[ii],st[ii,:,0],alfa[ii,:],0,ii)
    ss[0,:] = np.zeros(dd)

    #init
    for ii in range(NN):
        ss[0,:] += xt[ii,:,0] / NN
    for ii in range(NN):
        ut[ii,:,0] = 0
        #ut[ii,:,0] = grad_func(xt[ii,:,0],phi[ii],s[0,:],alfa[ii,:])
        store[ii,:] = gf2(xt[ii,:,0],rr[ii],st[ii,:,0],alfa[ii,:],0,ii)
    #alg
    for kk in range(MAX_ITERS-1):
        plot_grad = np.zeros(dd)
        #gradient
        for ii in range(NN):
            delta[ii,:] = store[ii,:].copy()
            #
        for ii in range(NN):
            plot_grad+= grad1n(xt[ii,:,kk],rr[ii],ss[kk,:],alfa[ii,:],0) +\
                        grad_s(alfa[ii,:]) * sum(grad2n(xt[jj,:,kk],rr[jj],ss[kk,:],alfa[jj,:],0) for jj in range(NN))
            

        #gt algo
        ss[kk+1,:] = 0
        for ii in range(NN):
            xt[ii,:,kk+1] = xt[ii,:,kk] - stepsize * (delta[ii,:]*u[kk,ii] + xt[ii,:,kk]*yt[ii,:,kk])
            #ut[ii,:,kk+1] = ut[ii,:,kk] - stepsize* a1*(upd1[ii,:] + zt[ii,:,kk] + upd2[ii,:])/(1-alfa[ii,:])
            ss[kk+1,:] += xt[ii,:,kk+1] / NN
        for ii in range(NN):
            st[ii,:,kk+1] = sum(AA[ii,jj] * (st[jj,:,kk]) for jj in range(NN)) + xt[ii,:,kk+1] - xt[ii,:,kk]
        for ii in range(NN):

            
            store[ii,:] = gf1(xt[ii,:,kk+1],rr[ii],st[ii,:,kk+1],alfa[ii,:],kk+1,ii)
            yt[ii,:,kk+1] = sum(AA[ii,jj] * (yt[jj,:,kk])for jj in range(NN)) + store[ii,:]*v[kk+1,ii] - delta[ii,:]*v[kk,ii] 
            
            cost_plot[kk] += cost_new(xt[ii, :, kk], rr[ii], ss[kk,:])
        """  
        if(kk % 1000 == 0):
            print(f"iter {kk} yt = {sum (yt[jj,0,kk] for jj in range(NN))}")
            print(f"iter {kk} g2 = {sum (upd2[jj,0] for jj in range(NN))}") 
        """    
        """ 
        # diminishing stepsize
        stepsize = steppp * 1/(1+kk*(1/const))
         """
        grad_plot[kk] = np.linalg.norm(plot_grad)
        #ut[:,:,kk+1] = ut[:,:,kk] - sterpsize*update
    
    print(f"final cost = {cost_plot[MAX_ITERS-2]}")
    print(f"final grad = {grad_plot[MAX_ITERS-2]}")
    #print(xt[:,:,MAX_ITERS-1])
    #print(s[MAX_ITERS-2,:])

agg_van(xx, grad1n, grad2n, Adj_weighted, costs[:,0], grad_norms[:,0], sigma,stepsize=0.005)
agg_nb1(xx, tzo1, grad2n, Adj_weighted, costs[:,1], grad_norms[:,1], sigma,stepsize=0.005)
agg_mod(xx, tzo1, tzo2, Adj_weighted, costs[:,2], grad_norms[:,2], sigma, stepsize=0.005)
agg_new(xx, tzo1, tzo2, Adj_weighted, costs[:,3], grad_norms[:,3], sigma,stepsize=0.005)

n_plots = 4

if(save):
    def save_data():
        data = {
            "NN": NN,
            "MAX_ITERS": MAX_ITERS,
            "costs": costs,
            "grad_norms": grad_norms,
            "xx": xx,
            "rr": rr,
            "sigma": sigma,
            "rr": rr
        }
        save_path = os.path.join(os.getcwd(), save_name)  
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
    save_data()

##########################################################################################
dp.plot_new(NN, MAX_ITERS, xx, sigma, rr, costs, grad_norms, n_plots)

