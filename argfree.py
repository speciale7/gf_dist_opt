#27/05/2025
#Giuseppe Speciale

import numpy as np
import er_graph as er
import os
import pickle
import dist_plot_ as dp

np.random.seed(1)
os.system('cls')
debug = 1
save = 0
save_name = 'saved_pickles\mc_12_06.pkl' # REMEMBER TO CHANGE NAME OR ELSE IT REWRITES EVERYTHING

ll = 1 # Tradeoff for Aggregative f.
gg = 2

def cost_new(x,r,s):
    return (gg/2)*np.linalg.norm(x-r)**2 + 0.5*np.linalg.norm(x-s)**2

def grad1n(x,r,s,a):
    return gg*(x-r) + (x-s)
"""
def grad1n(x,r,s,a):
    return gg*(x-r) + (x-s)*(1-1/(2*NN*s))
"""
def grad2n(x,r,s,a):
    return s-x

def grad_s(a):
    return 1/NN

#####################################################


def damping_matrix(i):
    matrices = []
    for _ in range(i):
        while True:
            A = np.random.randn(2, 2) * 0.1  # small random entries
            A = A + np.eye(2) * 0.95  # center eigenvalues around 0.95
            eigenvalues = np.linalg.eigvals(A)
            if np.all(np.isreal(eigenvalues)) and np.all((eigenvalues > 0.9) & (eigenvalues < 1)):
                matrices.append(A)
                break
    return np.array(matrices)


# Generate Network 
NN = 5
dd = 2
MAX_ITERS = 15000
MC_ITERS = 2

I_NN = np.identity(NN, dtype=int) 
p_ER = 0.6
Adj, Adj_weighted = er.graph(NN,p_ER)


# Initialization
# r_i are the positions of the targets (fixed)
# x_i are the positions of the targets (moving)


# Agents Dynamics (unused)
alfa = np.zeros((NN, dd))
alfa = np.random.uniform(0.7, 0.8, (NN, dd))

rr = np.random.uniform(-10, 10, (NN, 2))

xx_final = np.random.uniform(-10, 10, (NN, 2))
xx_init = np.random.uniform(-10, 10, (NN, 2))
xx = np.zeros((NN,dd,MAX_ITERS))
xx[:,:,0] = xx_init

costs = np.zeros((MAX_ITERS,4))
vg_costs = np.zeros((MAX_ITERS,2))
grad_norms = np.zeros((MAX_ITERS,4))
sigma = np.zeros((MAX_ITERS,dd,2))
est_sigma = np.zeros((NN,dd,MAX_ITERS))
print(xx_init)


# Pre-generate u and v for all iterations

uk = np.random.uniform(-1, 1, (MAX_ITERS, NN, dd))

# Unused
def generate_uv():
    u1 = np.random.uniform(-1, 1, (MAX_ITERS, NN, dd))
    v1 = np.random.uniform(-1, 1, (MAX_ITERS, NN, dd))
    return u1,v1

# Parameters
delta = 1e-6 
step__ = 2e-3
MM = damping_matrix(NN)

def argfree(xt, AA, cost_plot, grad_plot, sig, ss, stepsize, type):
    #print(rr)
    #uk,vk = generate_uv()
    vk = np.random.uniform(-1, 1, (MAX_ITERS, NN, dd)) 
    cost_plot[:] = 0
    grad_plot[:] = 0
    sig[:] = 0
    ut = np.zeros((NN,dd,MAX_ITERS))
    ss[0,:,1] = np.zeros(dd)

    gt = np.zeros_like(ut)#sigma
    st = np.zeros_like(ut)
    ft = np.zeros((NN,MAX_ITERS))#phi
    pt = np.zeros_like(ft)
    upd = np.zeros(dd)

    #init
    for ii in range(NN):
        gt[ii,:,0] = np.copy(xt[ii,:,0])
        #print("init", gt[ii,:,0])
        ft[ii,0] = cost_new(xt[ii,:,0], rr[ii], gt[ii,:,0])


        st[ii,:,0] = np.copy(xt[ii,:,0])
        #st[ii,:,0] = np.copy(xt[ii,:,0]) + delta*u[0,ii]
        pt[ii,0] = cost_new(xt[ii,:,0], rr[ii], st[ii,:,0])
        #pt[ii,:,0] = cost_new(xt[ii,:,0] + delta*u[0,ii], rr[ii], st[ii,:,0])
    for ii in range(NN):
        ss[0,:,1] += xt[ii,:,0] / NN

    #dyn
    for ii in range(NN):
        ut[ii,:,0] = 0

    #alg
    for kk in range(MAX_ITERS-1):
        plot_grad = np.zeros(dd)
        #gradient (obsolete??)
        for ii in range(NN):
            plot_grad+= grad1n(xt[ii,:,kk],rr[ii],ss[kk,:,1],alfa[ii,:]) +\
                        grad_s(alfa[ii,:]) * sum(grad2n(xt[jj,:,kk],rr[jj],ss[kk,:,1],alfa[jj,:]) for jj in range(NN))
            

        #argfree gt algo
        ss[kk+1,:,1] = 0

        xi = 0.4
        ix = 0.8
        uk[kk+1] = np.array([(ix*MM[i] @ uk[kk, i] + xi * vk[kk, i]) for i in range(NN)])

        for ii in range(NN):
            if type == '2p':
                upd = (pt[ii,kk] - ft[ii,kk])
            elif type == '1p':
                upd = (pt[ii,kk]) 
            elif type == '1pr':
                upd = (pt[ii,kk] - pt[ii,kk-1]) 
            xt[ii,:,kk+1] = xt[ii,:,kk] - stepsize * upd *uk[kk,ii] / delta 
            ss[kk+1,:,1] += xt[ii,:,kk+1] / NN

        for ii in range(NN):
            gt[ii,:,kk+1] = sum(AA[ii,jj] * (gt[jj,:,kk]) for jj in range(NN)) + xt[ii,:,kk+1] - xt[ii,:,kk]
            st[ii,:,kk+1] = sum(AA[ii,jj] * (st[jj,:,kk]) for jj in range(NN)) + (xt[ii,:,kk+1] + delta*uk[kk+1,ii]) - (xt[ii,:,kk] + delta*uk[kk,ii])
            ft[ii,kk+1] = sum(AA[ii,jj] * (ft[jj,kk]) for jj in range(NN)) + cost_new(xt[ii,:,kk+1], rr[ii], gt[ii,:,kk+1]) - cost_new(xt[ii,:,kk], rr[ii], gt[ii,:,kk])
            pt[ii,kk+1] = sum(AA[ii,jj] * (pt[jj,kk]) for jj in range(NN)) \
                            + cost_new(xt[ii,:,kk+1]+ delta*uk[kk+1,ii], rr[ii], st[ii,:,kk+1]) - cost_new(xt[ii,:,kk]+ delta*uk[kk,ii], rr[ii], st[ii,:,kk])
            
            cost_plot[kk] += cost_new(xt[ii, :, kk], rr[ii], ss[kk,:,1])

        grad_plot[kk] = np.linalg.norm(plot_grad)
        sig[:,:,kk] = np.copy(gt[:, :, kk])
    
    print(f"final cost = {cost_plot[MAX_ITERS-2]}")
    print(f"final grad = {grad_plot[MAX_ITERS-2]}")
#
def agg_gt(xt, gf1, gf2, AA, cost_plot, grad_plot, sig, ss, stepsize, d = False):
    #print(rr)
    cost_plot[:] = 0
    grad_plot[:] = 0
    sig[:] = 0
    upd1=np.zeros((NN,dd))
    upd2=np.zeros((NN,dd))
    store=np.zeros((NN,dd))
    ut = np.zeros((NN,dd,MAX_ITERS))
    st = np.zeros_like(ut)
    yt = np.zeros_like(ut)
    for ii in range(NN):
        st[ii,:,0] = np.copy(xt[ii,:,0])
        yt[ii,:,0] = gf2(xt[ii,:,0],rr[ii],st[ii,:,0],alfa[ii,:])
    ss[0,:,0] = np.zeros(dd)

    #init
    for ii in range(NN):
        ss[0,:,0] += xt[ii,:,0] / NN
    for ii in range(NN):
        ut[ii,:,0] = 0
        #ut[ii,:,0] = grad_func(xt[ii,:,0],phi[ii],s[0,:],alfa[ii,:])
        store[ii,:] = gf2(xt[ii,:,0],rr[ii],st[ii,:,0],alfa[ii,:])
    #alg
    for kk in range(MAX_ITERS-1):
        plot_grad = np.zeros(dd)
        #gradient
        for ii in range(NN):
            if d:
                upd1[ii,:], upd2[ii,:] = gf1(xt[ii,:,kk],rr[ii],st[ii,:,kk],alfa[ii,:]), store[ii,:].copy()
            else:
                upd1[ii,:], upd2[ii,:] = gf1(xt[ii,:,kk],rr[ii],st[ii,:,kk],alfa[ii,:]), gf2(xt[ii,:,kk],rr[ii],st[ii,:,kk],alfa[ii,:])
            #
        for ii in range(NN):
            plot_grad+= grad1n(xt[ii,:,kk],rr[ii],ss[kk,:,0],alfa[ii,:]) +\
                        grad_s(alfa[ii,:]) * sum(grad2n(xt[jj,:,kk],rr[jj],ss[kk,:,0],alfa[jj,:]) for jj in range(NN))
            

        #gt algo
        ss[kk+1,:,0] = 0
        for ii in range(NN):
            xt[ii,:,kk+1] = xt[ii,:,kk] - stepsize * (upd1[ii,:] + xt[ii,:,kk]*yt[ii,:,kk])
            #ut[ii,:,kk+1] = ut[ii,:,kk] - stepsize* a1*(upd1[ii,:] + zt[ii,:,kk] + upd2[ii,:])/(1-alfa[ii,:])
            ss[kk+1,:,0] += xt[ii,:,kk+1] / NN
        for ii in range(NN):
            st[ii,:,kk+1] = sum(AA[ii,jj] * (st[jj,:,kk]) for jj in range(NN)) + xt[ii,:,kk+1] - xt[ii,:,kk]
        for ii in range(NN):
            store[ii,:] = gf2(xt[ii,:,kk+1],rr[ii],st[ii,:,kk+1],alfa[ii,:])
            yt[ii,:,kk+1] = sum(AA[ii,jj] * (yt[jj,:,kk])for jj in range(NN)) + store[ii,:] - upd2[ii,:]
            
            cost_plot[kk] += cost_new(xt[ii, :, kk], rr[ii], ss[kk,:,0])
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
        sig[:,:,kk] = np.copy(st[:, :, kk])
        #ut[:,:,kk+1] = ut[:,:,kk] - sterpsize*update
    
    print(f"final cost = {cost_plot[MAX_ITERS-2]}")
    print(f"final grad = {grad_plot[MAX_ITERS-2]}")
    #print(xt[:,:,MAX_ITERS-1])
    #print(s[MAX_ITERS-2,:])

agg_gt(xx, grad1n, grad2n, Adj_weighted, costs[:,1], grad_norms[:,1],est_sigma, sigma, stepsize=step__)
#argfree(xx, Adj_weighted, costs[:,0], grad_norms[:,0], sigma, stepsize=step__, type='2p')
#argfree(xx, Adj_weighted, costs[:,1], grad_norms[:,1], sigma, stepsize=step__, type='1pr')
#argfree(xx, Adj_weighted, costs[:,2], grad_norms[:,2], sigma, stepsize=2.5e-5, type='2p')
n_plots = 2

sig_est = np.zeros((NN,dd,MAX_ITERS,2))
sig_est[:,:,:,1] = est_sigma.copy()

# Monte Carlo
if(1):
    x_avg = np.zeros((NN,dd,MAX_ITERS))
    cost_avg = np.zeros((MAX_ITERS,4))
    grad_avg = np.zeros((MAX_ITERS,4))
    cost_total = np.zeros((MC_ITERS,MAX_ITERS))
    cost_2 = np.zeros((MC_ITERS,MAX_ITERS))
    opt= costs[-2,1]
    for mc in range(MC_ITERS):
        argfree(xx, Adj_weighted, costs[:,0], grad_norms[:,0], est_sigma, sigma, stepsize=step__, type='2p')
        x_avg += xx/MC_ITERS
        cost_avg[:,0] += costs[:,0]
        grad_avg[:,0] += grad_norms[:,0]
        sig_est[:,:,:,0] += est_sigma
        cost_total[mc,:] = costs[:,0]
        cost_2[mc,:] = (costs[:,0] - opt) / opt
    std_val = np.std(cost_total, axis=0)
    std_norm = np.std(cost_2, axis=0)
    n_plots = 2
    xx = x_avg
    sig_est[:,:,:,0] /= MC_ITERS
    costs[:,0] = cost_avg[:,0]/MC_ITERS
    grad_norms[:,0] = grad_avg[:,0]/MC_ITERS
    print(f"Average final cost = {costs[-2,0]}")
    #print(f"Average final grad = {grad_norms[MAX_ITERS-2,0]}")
costs[:,2] = (costs[:,0] - opt) / opt
costs[:,3] = (costs[:,1] - opt) / opt



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
            "sig_est": sig_est,
            "std_val": std_val,
            "std_norm": std_norm,
            "n_plots": n_plots,
        }
        save_path = os.path.join(os.getcwd(), save_name)  
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
    save_data()
    print(f"Data saved to {save_name}")

##########################################################################################
dp.plot_27_5(NN, MAX_ITERS, xx, sigma[:,:,1], rr, costs, grad_norms, std_val,std_norm, n_plots)
#dp.static(NN, MAX_ITERS, xx, sigma[:,:,1], rr, costs, grad_norms, n_plots)
#dp.plot_sigma(NN, MAX_ITERS, sigma, sig_est, 2)

