import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import pickle
import os
os.system('cls')

size = 16
mpl.rcParams.update({
    'font.serif': ['Computer Modern Roman'],
    'font.size': size,              # Match main LaTeX font size
    'axes.titlesize': size,
    'axes.labelsize': size,
    'xtick.labelsize': size,
    'ytick.labelsize': size,
    'legend.fontsize': size,
    'figure.titlesize': size      
})

load = 0
#load_file = 'saved_pickles\\noise_multi_07_07.pkl'
load_file ='saved_pickles\\noise_multi_08_07.pkl'
def plot_agg(NN, MAX_ITERS, xx, xx_real, ww, sigma, costs, grad_norms, num_plots=1):
    # Plot cost and norm of the gradient evolution
    fig, (ax1, ax2) = plt.subplots(2, 1)
    namesake = ['VG', 'TZO', 'SZO', 'RF-SZO']
    [ax1.semilogy(range(MAX_ITERS-1), costs[:-1, i], label=f'{namesake[i]} Cost') for i in range(num_plots)]
    ax1.set_xlim(0, MAX_ITERS)
    #ax1.set_ylim(min(costs[:-1, :].min(), 1e-10), costs[:-1, :].max())
    ax1.grid(True)
    ax1.set_xlabel('t')
    ax1.set_ylabel(r'$\sum_{i=1}^N f_i(x_i^t)$')
    ax1.set_title('Cost Evolution')
    ax1.legend()

    [ax2.semilogy(range(MAX_ITERS-1), grad_norms[:-1, i,:], label=f'{namesake[i]} Gradient Norm') for i in range(num_plots)]
    ax2.set_xlim(0, MAX_ITERS)
    #ax2.set_ylim(min(grad_norms[:-1, :].min(), 1e-10), grad_norms[:-1, :].max())
    ax2.grid(True)
    ax2.set_xlabel('iteration t')
    ax2.set_ylabel(r'$\|\sum_{i=1}^N \nabla f_i(x_i^t)\|$')
    ax2.set_title('Gradient Norm Evolution')
    ax2.legend()
    plt.tight_layout()
    plt.show()

    # Plot estimates evolution

    fig, ax = plt.subplots()
    #lines = [ax.plot([], [], linestyle='--', linewidth=1,label = 'Agent Estimates')]
    lines = [ax.plot([], [], linewidth=1)[0] for ii in range(NN)]
    rp = []

    # Plot static points for x_target and ww
    #ax.scatter(rr[:, 0], rr[:, 1], c='blue', marker='x', label='Initial Estimates')
    #ax.scatter(ww[:, 0], ww[:, 1], c='red', marker='x', label='Agents\' Positions')
    #ax.scatter(x_target[0, 0], x_target[1, 0], c='gold', marker='*', s=100, label='Target')

    def init():
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal', 'box')  # Ensure square proportions
        return lines

    def update(frame):
        for ii, line in enumerate(lines):
            line.set_data([xx_real[ii,0], xx[ii, 0, frame]], [xx_real[ii,1], xx[ii, 1, frame]])

        if frame == 0:
            for ii, line in enumerate(lines):
                real_pos = ax.scatter(xx_real[ii,0], xx_real[ii, 1], c=line.get_color(), marker='x', s=50, label='Real Positions')
                rp.append(real_pos)
        real_bar = np.zeros(2)
        for ii in range(NN):
            real_bar += xx_real[ii, :] / NN
        # Create a scatter plot for sigma at the current frame
        bary = ax.scatter(sigma[frame, 0], sigma[frame, 1], c='green', marker='*', s=50, label='Barycenter')
        rb = ax.scatter(real_bar[0], real_bar[1], c='blue', marker='*', s=50, label='Real Barycenter')
        star = ax.scatter(ww[0], ww[1], c='black', marker='*', s=200, label='Target', zorder = 10)
        pos = ax.scatter(xx[:, 0, frame], xx[:, 1, frame], c='red', marker='x', s = 50, label='Agents\' Positions')
        #return rp + [bary, star, pos]
        return lines + rp + [rb,bary, star, pos]

    ani = animation.FuncAnimation(fig, update, frames=MAX_ITERS-1, init_func=init, blit=True, repeat=False, interval=60)

    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.title('Evolution of x_t')

    bary_legend = plt.Line2D([0], [0], marker='x', color='green', label='Barycenter',
                            markerfacecolor='green', markersize=8)
    star_legend = plt.Line2D([0], [0], marker='*', color='w', label='Target',
                            markerfacecolor='black', markersize=12)
    agents_legend = plt.Line2D([0], [0], marker='x', color='red', linestyle='None',
                            label="Agents' Positions")
    plt.legend(handles=[bary_legend, star_legend, agents_legend])

    plt.show()

def plot_all(NN, MAX_ITERS, xx, ww, sigma, phi, costs, grad_norms, num_plots=1):
    # Plot cost and norm of the gradient evolution
    fig, (ax1, ax2) = plt.subplots(2, 1)
    namesake = ['VG', 'TZO', 'SZO', 'RF-SZO']
    [ax1.semilogy(range(MAX_ITERS-1), costs[:-1, i], label=f'{namesake[i]} Cost') for i in range(num_plots)]
    ax1.set_xlim(0, MAX_ITERS)
    #ax1.set_ylim(min(costs[:-1, :].min(), 1e-10), costs[:-1, :].max())
    ax1.grid(True)
    ax1.set_xlabel('t')
    ax1.set_ylabel(r'$\sum_{i=1}^N f_i(x_i^t)$')
    ax1.set_title('Cost Evolution')
    ax1.legend()

    [ax2.semilogy(range(MAX_ITERS-1), grad_norms[:-1, i], label=f'{namesake[i]} Gradient Norm') for i in range(num_plots)]
    ax2.set_xlim(0, MAX_ITERS)
    #ax2.set_ylim(min(grad_norms[:-1, :].min(), 1e-10), grad_norms[:-1, :].max())
    ax2.grid(True)
    ax2.set_xlabel('iteration t')
    ax2.set_ylabel(r'$\|\sum_{i=1}^N \nabla f_i(x_i^t)\|$')
    ax2.set_title('Gradient Norm Evolution')
    ax2.legend()
    plt.tight_layout()
    plt.show()

    # Plot estimates evolution

    fig, ax = plt.subplots()
    #lines = [ax.plot([], [], linestyle='--', linewidth=1,label = 'Agent Estimates')]
    lines = [ax.plot([], [], linewidth=1)[0] for ii in range(NN)]
    circles = [plt.Circle((ww[0], ww[1]), 0, color=line.get_color(), fill=False) for line in lines]
    for circle in circles:
        ax.add_patch(circle)

    # Plot static points for x_target and ww
    #ax.scatter(rr[:, 0], rr[:, 1], c='blue', marker='x', label='Initial Estimates')
    #ax.scatter(ww[:, 0], ww[:, 1], c='red', marker='x', label='Agents\' Positions')
    #ax.scatter(x_target[0, 0], x_target[1, 0], c='gold', marker='*', s=100, label='Target')

    def init():
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal', 'box')  # Ensure square proportions
        return lines

    def update(frame):
        for ii, line in enumerate(lines):
            line.set_data([ww[0], xx[ii, 0, frame]], [ww[1], xx[ii, 1, frame]])
            circles[ii].set_radius(np.sqrt(phi[ii])) 

        if frame == 0:
            for ii, line in enumerate(lines):
                circle = plt.Circle((ww[0], ww[1]), np.sqrt(phi[ii]), color=line.get_color(), fill=False)
                ax.add_patch(circle)
        # Create a scatter plot for sigma at the current frame
        bary = ax.scatter(sigma[frame, 0], sigma[frame, 1], c='green', marker='x', s=50, label='Barycenter')
        star = ax.scatter(ww[0], ww[1], c='black', marker='*', s=200, label='Target', zorder = 10)
        pos = ax.scatter(xx[:, 0, frame], xx[:, 1, frame], c='red', marker='x', s = 50, label='Agents\' Positions')
        return lines + circles + [bary, star, pos]

    ani = animation.FuncAnimation(fig, update, frames=MAX_ITERS-1, init_func=init, blit=True, repeat=False, interval=60)

    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.title('Evolution of x_t')

    bary_legend = plt.Line2D([0], [0], marker='x', color='green', label='Barycenter',
                            markerfacecolor='green', markersize=8)
    star_legend = plt.Line2D([0], [0], marker='*', color='w', label='Target',
                            markerfacecolor='black', markersize=12)
    agents_legend = plt.Line2D([0], [0], marker='x', color='red', linestyle='None',
                            label="Agents' Positions")
    plt.legend(handles=[bary_legend, star_legend, agents_legend])

    plt.show()

def plot_sigma(NN, MAX_ITERS, sigma, sig_est, num_plots=1):
    # Plot cost and norm of the gradient evolution
    sig_est_arg = sig_est[:,:,:,0]
    sig_est_vg = sig_est[:,:,:,1]

    sig_avg_arg = np.mean(sig_est_arg, axis=0)
    sig_avg_vg = np.mean(sig_est_vg, axis=0)
    print(sig_est.shape)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,6))
    # Compute the norm across agents for each iteration (for x-coordinate)
    # Compute the norm across agents for each iteration (for x-coordinate)
    norm_x = np.linalg.norm(sig_est_arg[:, 0, :MAX_ITERS-1] - sig_avg_arg[0, :MAX_ITERS-1], axis=0)
    norm_y = np.linalg.norm(sig_est_arg[:, 1, :MAX_ITERS-1] - sig_avg_arg[1, :MAX_ITERS-1], axis=0)

    norm_x_ = np.linalg.norm(sig_est_vg[:, 0, :MAX_ITERS-1] - sig_avg_vg[0, :MAX_ITERS-1], axis=0)
    norm_y_ = np.linalg.norm(sig_est_vg[:, 1, :MAX_ITERS-1] - sig_avg_vg[1, :MAX_ITERS-1], axis=0)



    ax1.semilogy(
        range(MAX_ITERS-1),
        np.sqrt(norm_x**2 + norm_y**2),
        label=r'$\|\sigma - \mathcal{J}\sigma\|$ (GF)',
        alpha=1
    )

    ax1.semilogy(
        range(MAX_ITERS-1),
        np.sqrt(norm_x_**2 + norm_y_**2),
        label=r'$\|\sigma - \mathcal{J}\sigma\|$ (VG)',
        alpha=1
    )

    ax1.set_xlim(0, MAX_ITERS)
    ax1.grid(True)
    ax1.set_xlabel('iteration')
    ax1.set_ylabel(r'$|\sigma - \overline{\sigma}|$')
    ax1.legend()

    ax2.plot(range(MAX_ITERS-1), sigma[:MAX_ITERS-1, 0, 0], label=r'$\sigma_x^{VG}$')
    ax2.plot(range(MAX_ITERS-1), sigma[:MAX_ITERS-1, 1, 0], label=r'$\sigma_y^{VG}$')
    ax2.plot(range(MAX_ITERS-1), sigma[:MAX_ITERS-1, 0, 1], label=r'$\sigma_x^{GF}$')
    ax2.plot(range(MAX_ITERS-1), sigma[:MAX_ITERS-1, 1, 1], label=r'$\sigma_y^{GF}$')
    ax2.set_xlim(0, MAX_ITERS)
    ax2.grid(True)
    ax2.set_xlabel('iteration')
    ax2.set_ylabel(r'$\sigma$')
    ax2.legend()

    
    #############################
    plt.tight_layout()
    plt.show()
  
def plot_grad(NN, MAX_ITERS, xx, sigma, rr, costs, grad_norms, std, num_plots=1):
    # Plot cost and norm of the gradient evolution
    fig, (ax1) = plt.subplots(1, 1, figsize=(12,3))
    fig.set_size_inches(12, 4)
    namesake = ['GF','VG',  'Norm. ARGFree', 'Norm VG']
    [ax1.plot(range(MAX_ITERS-1), grad_norms[:-1, i], label=f'{namesake[i]} Gradient Norm') for i in range(num_plots)]
    ax1.fill_between(np.arange(MAX_ITERS), grad_norms[:, 0] - std, grad_norms[:, 0] + std, alpha=0.3)
    ax1.set_xlim(0, MAX_ITERS)
    # Set y-axis lower limit to the smallest value of all plotted data (but not below 1e-10 for safety)
   

    ax1.grid(True)
    ax1.set_xlabel('t')
    ax1.set_ylabel(r'$\|\sum_{i=1}^N \nabla f_i(x_i^t)\|$')
    ax1.legend()

    """ 
    
    """


    #############################
    plt.tight_layout()
    plt.show()


    # Plot estimates evolution

    fig, ax = plt.subplots()
    #lines = [ax.plot([], [], linestyle='--', linewidth=1,label = 'Agent Estimates')]
    lines = [ax.plot([], [], linewidth=1)[0] for ii in range(NN)]

    # Plot static points for x_target and ww
    #ax.scatter(rr[:, 0], rr[:, 1], c='blue', marker='x', label='Initial Estimates')
    #ax.scatter(ww[:, 0], ww[:, 1], c='red', marker='x', label='Agents\' Positions')
    #ax.scatter(x_target[0, 0], x_target[1, 0], c='gold', marker='*', s=100, label='Target')

def plot_27_5(NN, MAX_ITERS, xx, sigma, rr, costs, grad_norms, std,std2, num_plots=1):
    # Plot cost and norm of the gradient evolution
    fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(12,6))
    namesake = ['ARGFree','VG',  'Norm. ARGFree', 'Norm VG']
    [ax1.semilogy(range(MAX_ITERS-1), costs[:-1, i], label=f'{namesake[i]} Cost') for i in range(num_plots)]
    ax1.set_xlim(0, MAX_ITERS)
    #ax1.set_ylim(min(costs[:-1, :].min(), 1e-10), costs[:-1, :].max())
    ax1.grid(True)
    ax1.set_xlabel('t')
    ax1.set_ylabel(r'$\sum_{i=1}^N f_i(x_i^t)$')
    ax1.legend()

    # Apply scientific notation to ax1's y-axis
    formatter1 = ticker.ScalarFormatter(useMathText=True)
    formatter1.set_scientific(True)
    formatter1.set_powerlimits((0, 0))  # Always use scientific notation
    ax1.xaxis.set_major_formatter(formatter1)
    ax1.fill_between(np.arange(MAX_ITERS), costs[:, 0] - std, costs[:, 0] + std, alpha=0.3, label='±1 std dev')
    
    [ax2.semilogy(range(MAX_ITERS-1), costs[:-1, i+2], label=f'{namesake[i+2]} Cost') for i in range(num_plots)]
    ax2.set_xlim(0, MAX_ITERS)
    ax2.set_ylim(1e-4, costs[:-1, 2:num_plots+2].max())
    ax2.grid(True)
    ax2.set_xlabel('t')
    ax2.set_ylabel(r'$\dfrac{f(x)-f(x^*)}{f(x^*)}$')
    ax2.legend()
    ax2.xaxis.set_major_formatter(formatter1)
    ax2.fill_between(np.arange(MAX_ITERS), costs[:, 2] - std2, costs[:, 2] + std2, alpha=0.3, label='±1 std dev')

    """ 
    [ax3.semilogy(range(MAX_ITERS-1), grad_norms[:-1, i], label=f'{namesake[i]} Gradient Norm') for i in range(num_plots)]
    ax3.set_xlim(0, MAX_ITERS)
    ax3.grid(True)
    ax3.set_xlabel('k')
    ax3.set_ylabel(r'$\|\sum_{i=1}^N \nabla f_i(x_i^t)\|$')
    ax3.legend()
    ax3.xaxis.set_major_formatter(formatter1)
    """


    #############################
    plt.tight_layout()
    plt.show()


    # Plot estimates evolution

    fig, ax = plt.subplots()
    #lines = [ax.plot([], [], linestyle='--', linewidth=1,label = 'Agent Estimates')]
    lines = [ax.plot([], [], linewidth=1)[0] for ii in range(NN)]

    # Plot static points for x_target and ww
    #ax.scatter(rr[:, 0], rr[:, 1], c='blue', marker='x', label='Initial Estimates')
    #ax.scatter(ww[:, 0], ww[:, 1], c='red', marker='x', label='Agents\' Positions')
    #ax.scatter(x_target[0, 0], x_target[1, 0], c='gold', marker='*', s=100, label='Target')

    def init():
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal', 'box')  # Ensure square proportions
        return lines
    
    # Create empty lists to store text objects
    target_labels = []
    agent_labels = []

    # Create initial text objects for each target and agent
    for i in range(len(rr)):
        r_label = ax.text(rr[i, 0], rr[i, 1], f'$r_{{{i}}}$', fontsize=9, color='blue')
        x_label = ax.text(xx[i, 0, 0], xx[i, 1, 0], f'$x_{{{i}}}$', fontsize=9, color='red')
        target_labels.append(r_label)
        agent_labels.append(x_label)

    # Barycenter label 
    bary_label = ax.text(0, 0, '', fontsize=10, color='green')
    def update(frame):
        
        for ii, line in enumerate(lines):
            line.set_data([rr[ii,0], xx[ii, 0, frame]], [rr[ii,1], xx[ii, 1, frame]])

        # Create a scatter plot for sigma at the current frame
        bary = ax.scatter(sigma[frame, 0], sigma[frame, 1], c='green', marker='o', s=25, label='Barycenter')
        pos = ax.scatter(xx[:, 0, frame], xx[:, 1, frame], c='red', marker='x', s = 25, label='Agents\' Positions')
        tpos = ax.scatter(rr[:, 0], rr[:, 1], c='blue', marker='o', s = 25, label='Targets\' Positions')

        # Update barycenter label
        bary_label.set_position((sigma[frame, 0] + 0.3, sigma[frame, 1] + 0.3))
        bary_label.set_text(r'$\sigma(x)$')
        #annotation.set_text(r'$\sigma = (%.2f,\ %.2f)$' % (sigma[frame, 0], sigma[frame, 1]))

        # Update agent and target labels
        for i in range(len(rr)):
            target_labels[i].set_position((rr[i, 0] + 0.3, rr[i, 1] + 0.3))
            agent_labels[i].set_position((xx[i, 0, frame] + 0.3, xx[i, 1, frame] + 0.3))

        return lines + agent_labels + target_labels + [bary, pos, tpos, bary_label] 

    ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, 100, 20), init_func=init, blit=True, repeat=False, interval=5)
    
    ani.save("animazione.gif", writer="pillow")

    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.title('Evolution of x_t')

    bary_legend = plt.Line2D([0], [0], marker='o', color='green', label='Barycenter',
                            markerfacecolor='green', markersize=8)
    agents_legend = plt.Line2D([0], [0], marker='x', color='red', linestyle='None',
                            label="Agents' Positions")
    targets_legend = plt.Line2D([0], [0], marker='x', color='blue', linestyle='None',
                            label="Targets' Positions")
    plt.legend(handles=[bary_legend, agents_legend, targets_legend])

    plt.show()

def plot_new(NN, MAX_ITERS, xx, sigma, rr, costs, grad_norms, num_plots=1):
    # Plot cost and norm of the gradient evolution
    fig, (ax1, ax2) = plt.subplots(2, 1)
    namesake = ['VG', 'TZO 1', 'TZO 1+2', 'RF-TZO']
    [ax1.semilogy(range(MAX_ITERS-1), costs[:-1, i], label=f'{namesake[i]} Cost') for i in range(num_plots)]
    ax1.set_xlim(0, MAX_ITERS)
    #ax1.set_ylim(min(costs[:-1, :].min(), 1e-10), costs[:-1, :].max())
    ax1.grid(True)
    ax1.set_xlabel('t')
    ax1.set_ylabel(r'$\sum_{i=1}^N f_i(x_i^t)$')
    ax1.set_title('Cost Evolution')
    ax1.legend()

    [ax2.semilogy(range(MAX_ITERS-1), grad_norms[:-1, i], label=f'{namesake[i]} Gradient Norm') for i in range(num_plots)]
    ax2.set_xlim(0, MAX_ITERS)
    #ax2.set_ylim(min(grad_norms[:-1, :].min(), 1e-10), grad_norms[:-1, :].max())
    ax2.grid(True)
    ax2.set_xlabel('iteration t')
    ax2.set_ylabel(r'$\|\sum_{i=1}^N \nabla f_i(x_i^t)\|$')
    ax2.set_title('Gradient Norm Evolution')
    ax2.legend()
    plt.tight_layout()
    plt.show()

    # Plot estimates evolution

    fig, ax = plt.subplots()
    #lines = [ax.plot([], [], linestyle='--', linewidth=1,label = 'Agent Estimates')]
    lines = [ax.plot([], [], linewidth=1)[0] for ii in range(NN)]

    # Plot static points for x_target and ww
    #ax.scatter(rr[:, 0], rr[:, 1], c='blue', marker='x', label='Initial Estimates')
    #ax.scatter(ww[:, 0], ww[:, 1], c='red', marker='x', label='Agents\' Positions')
    #ax.scatter(x_target[0, 0], x_target[1, 0], c='gold', marker='*', s=100, label='Target')

    def init():
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal', 'box')  # Ensure square proportions
        return lines
    
    # Create empty lists to store text objects
    target_labels = []
    agent_labels = []

    # Create initial text objects for each target and agent
    for i in range(len(rr)):
        r_label = ax.text(rr[i, 0], rr[i, 1], f'$r_{{{i}}}$', fontsize=9, color='blue')
        x_label = ax.text(xx[i, 0, 0], xx[i, 1, 0], f'$x_{{{i}}}$', fontsize=9, color='red')
        target_labels.append(r_label)
        agent_labels.append(x_label)

    # Barycenter label (already handled)
    bary_label = ax.text(0, 0, '', fontsize=10, color='green')
    def update(frame):
        
        for ii, line in enumerate(lines):
            line.set_data([rr[ii,0], xx[ii, 0, frame]], [rr[ii,1], xx[ii, 1, frame]])

        # Create a scatter plot for sigma at the current frame
        bary = ax.scatter(sigma[frame, 0], sigma[frame, 1], c='green', marker='o', s=25, label='Barycenter')
        pos = ax.scatter(xx[:, 0, frame], xx[:, 1, frame], c='red', marker='x', s = 25, label='Agents\' Positions')
        tpos = ax.scatter(rr[:, 0], rr[:, 1], c='blue', marker='o', s = 25, label='Targets\' Positions')

        # Update barycenter label
        bary_label.set_position((sigma[frame, 0] + 0.3, sigma[frame, 1] + 0.3))
        bary_label.set_text(r'$\sigma(x)$')
        #annotation.set_text(r'$\sigma = (%.2f,\ %.2f)$' % (sigma[frame, 0], sigma[frame, 1]))

        # Update agent and target labels
        for i in range(len(rr)):
            target_labels[i].set_position((rr[i, 0] + 0.3, rr[i, 1] + 0.3))
            agent_labels[i].set_position((xx[i, 0, frame] + 0.3, xx[i, 1, frame] + 0.3))

        return lines + agent_labels + target_labels + [bary, pos, tpos, bary_label] 

    ani = animation.FuncAnimation(fig, update, frames=MAX_ITERS-1, init_func=init, blit=True, repeat=False, interval=5)

    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.title('Evolution of x_t')

    bary_legend = plt.Line2D([0], [0], marker='o', color='green', label='Barycenter',
                            markerfacecolor='green', markersize=8)
    agents_legend = plt.Line2D([0], [0], marker='x', color='red', linestyle='None',
                            label="Agents' Positions")
    targets_legend = plt.Line2D([0], [0], marker='x', color='blue', linestyle='None',
                            label="Targets' Positions")
    plt.legend(handles=[bary_legend, agents_legend, targets_legend])

    plt.show()

def init(NN, MAX_ITERS, xx, sigma, rr, costs, grad_norms, num_plots=1):
    agent_colors = ['red', 'blue', 'green', 'orange', 'purple', 'teal', 'brown', 'magenta', 'olive', 'cyan']

    fig, ax = plt.subplots()

    for ii in range(NN):
        color = agent_colors[ii % len(agent_colors)]  # Cycle if more agents than colors
        # Only plot the initial position for each agent
        ax.scatter(xx[ii, 0, 0], xx[ii, 1, 0], color=color, marker='x',s =200, label=f'Agent {ii} Initial' if ii == 0 else "")
        ax.scatter(rr[ii, 0], rr[ii, 1], c=color, marker='*', s=200, label='Targets' if ii == 0 else "")
        # Draw a segment from each agent's initial position to its target
        ax.plot([rr[ii, 0], xx[ii, 0, 0]], [rr[ii, 1], xx[ii, 1, 0]], color=color, linestyle='--', linewidth=1)

    ax.scatter(sigma[0, 0], sigma[0, 1], c='black', marker='+', alpha=0.5, s=200, label='Initial Barycenter')

    # Axes and labels
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x-coordinate')
    ax.set_ylabel('y-coordinate')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='x', color='w', label='Initial Position',
            markerfacecolor='gray', markeredgecolor='gray', markersize=8),
        Line2D([0], [0], marker='*', color='w', label='Targets',
            markerfacecolor='black', markersize=12)
    ]
    ax.legend(handles=legend_elements)

    plt.show()

def static(NN, MAX_ITERS, xx, sigma, rr, costs, grad_norms, num_plots=1):

    agent_colors = ['red', 'blue', 'green', 'orange', 'purple', 'teal', 'brown', 'magenta', 'olive', 'cyan']

    fig, ax = plt.subplots()

    for ii in range(NN):
        color = agent_colors[ii % len(agent_colors)]  # Cycle if more agents than colors
        ax.plot(xx[ii, 0, :], xx[ii, 1, :], color=color, linewidth=1, alpha=0.25)
        ax.scatter(xx[ii, 0, 0], xx[ii, 1, 0], color=color, marker='s')  # Initial position
        ax.scatter(xx[ii, 0, -1], xx[ii, 1, -1], color=color, marker='x', s=100)  # Final position
        ax.scatter(rr[ii,0], rr[ii,1], c=color, marker='*', s=100, label='Targets')
        # Draw a segment from each agent's initial position to its target
        ax.plot([rr[ii, 0], xx[ii, 0, -1]], [rr[ii, 1], xx[ii, 1, -1]], color=color, linestyle='--', linewidth=1)
        ax.plot([sigma[-1, 0], xx[ii, 0, MAX_ITERS-1]], [sigma[-1, 1], xx[ii, 1, MAX_ITERS-1]], color=color, linestyle=':', linewidth=1)

        
    ax.scatter(sigma[0, 0], sigma[0, 1],c='black', marker='+',alpha=0.5,s=200)  # Initial barycenter position
    # Plot barycenter path
    ax.plot(sigma[:, 0], sigma[:, 1], c='black', linestyle='--',alpha=0.25,label='Barycenter Path')
    ax.scatter(sigma[-1, 0], sigma[-1, 1], c='black', marker='+', s=200)

    # Plot target(s) — from rr

    # Axes and labels
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x-coordinate')
    ax.set_ylabel('y-coordinate')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', lw=1, label='Agent Trajectory'),
        Line2D([0], [0], marker='s', color='w', label='Initial Position',
            markerfacecolor='gray', markersize=8),
        Line2D([0], [0], marker='x', color='w', label='Final Position',
            markerfacecolor='gray', markeredgecolor='gray', markersize=8),
        Line2D([0], [0], marker='+', color='w', label='Barycenter',
            markerfacecolor='gray', markeredgecolor='gray', markersize=8),
        Line2D([0], [0], marker='*', color='w', label='Targets',
            markerfacecolor='gray', markersize=12)
    ]
    ax.legend(handles=legend_elements)

    plt.show()

def plot_19_5(NN, MAX_ITERS, xx, sigma, rr, costs, grad_norms, sigma_norms, y_norms, num_plots=1):
    # Plot cost and norm of the gradient evolution
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    namesake = ['VG', 'TZO 1', 'TZO 1+2', 'RF-TZO']
    [ax1.semilogy(range(MAX_ITERS-1), costs[:-1, i], label=f'{namesake[i]} Cost') for i in range(num_plots)]
    ax1.set_xlim(0, MAX_ITERS)
    #ax1.set_ylim(min(costs[:-1, :].min(), 1e-10), costs[:-1, :].max())
    ax1.grid(True)
    ax1.set_xlabel('t')
    ax1.set_ylabel(r'$\sum_{i=1}^N f_i(x_i^t)$')
    ax1.set_title('Cost Evolution')
    ax1.legend()

    [ax2.semilogy(range(MAX_ITERS-1), grad_norms[:-1, i], label=f'{namesake[i]} Gradient Norm') for i in range(num_plots)]
    ax2.set_xlim(0, MAX_ITERS)
    #ax2.set_ylim(min(grad_norms[:-1, :].min(), 1e-10), grad_norms[:-1, :].max())
    ax2.grid(True)
    ax2.set_xlabel('iteration t')
    ax2.set_ylabel(r'$\|\sum_{i=1}^N \nabla f_i(x_i^t)\|$')
    ax2.set_title('Gradient Norm Evolution')
    ax2.legend()

    [ax3.semilogy(range(MAX_ITERS-1), sigma_norms[:-1, i], label=f'{namesake[i]} Gradient Norm') for i in range(num_plots)]
    ax3.set_xlim(0, MAX_ITERS)
    #ax3.set_ylim(min(grad_norms[:-1, :].min(), 1e-10), grad_norms[:-1, :].max())
    ax3.grid(True)
    ax3.set_xlabel('iteration t')
    ax3.set_ylabel(r'$\|\sum_{i=1}^N \nabla f_i(x_i^t)\|$')
    ax3.set_title('Gradient Norm Evolution')
    ax3.legend()

    [ax4.semilogy(range(MAX_ITERS-1), y_norms[:-1, i], label=f'{namesake[i]} Gradient Norm') for i in range(num_plots)]
    ax4.set_xlim(0, MAX_ITERS)
    #ax4.set_ylim(min(grad_norms[:-1, :].min(), 1e-10), grad_norms[:-1, :].max())
    ax4.grid(True)
    ax4.set_xlabel('iteration t')
    ax4.set_ylabel(r'$\|\sum_{i=1}^N \nabla f_i(x_i^t)\|$')
    ax4.set_title('Gradient Norm Evolution')
    ax4.legend()
    plt.show()
    # Plot estimates evolution

    fig, ax = plt.subplots()
    #lines = [ax.plot([], [], linestyle='--', linewidth=1,label = 'Agent Estimates')]
    lines = [ax.plot([], [], linewidth=1)[0] for ii in range(NN)]

    # Plot static points for x_target and ww
    #ax.scatter(rr[:, 0], rr[:, 1], c='blue', marker='x', label='Initial Estimates')
    #ax.scatter(ww[:, 0], ww[:, 1], c='red', marker='x', label='Agents\' Positions')
    #ax.scatter(x_target[0, 0], x_target[1, 0], c='gold', marker='*', s=100, label='Target')

    def init():
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal', 'box')  # Ensure square proportions
        return lines
    
    # Create empty lists to store text objects
    target_labels = []
    agent_labels = []

    # Create initial text objects for each target and agent
    for i in range(len(rr)):
        r_label = ax.text(rr[i, 0], rr[i, 1], f'$r_{{{i}}}$', fontsize=9, color='blue')
        x_label = ax.text(xx[i, 0, 0], xx[i, 1, 0], f'$x_{{{i}}}$', fontsize=9, color='red')
        target_labels.append(r_label)
        agent_labels.append(x_label)

    # Barycenter label (already handled)
    bary_label = ax.text(0, 0, '', fontsize=10, color='green')
    def update(frame):
        
        for ii, line in enumerate(lines):
            line.set_data([rr[ii,0], xx[ii, 0, frame]], [rr[ii,1], xx[ii, 1, frame]])

        # Create a scatter plot for sigma at the current frame
        bary = ax.scatter(sigma[frame, 0], sigma[frame, 1], c='green', marker='o', s=25, label='Barycenter')
        pos = ax.scatter(xx[:, 0, frame], xx[:, 1, frame], c='red', marker='x', s = 25, label='Agents\' Positions')
        tpos = ax.scatter(rr[:, 0], rr[:, 1], c='blue', marker='o', s = 25, label='Targets\' Positions')

        # Update barycenter label
        bary_label.set_position((sigma[frame, 0] + 0.3, sigma[frame, 1] + 0.3))
        bary_label.set_text(r'$\sigma(x)$')
        #annotation.set_text(r'$\sigma = (%.2f,\ %.2f)$' % (sigma[frame, 0], sigma[frame, 1]))

        # Update agent and target labels
        for i in range(len(rr)):
            target_labels[i].set_position((rr[i, 0] + 0.3, rr[i, 1] + 0.3))
            agent_labels[i].set_position((xx[i, 0, frame] + 0.3, xx[i, 1, frame] + 0.3))

        return lines + agent_labels + target_labels + [bary, pos, tpos, bary_label] 

    ani = animation.FuncAnimation(fig, update, frames=MAX_ITERS-1, init_func=init, blit=True, repeat=False, interval=10)

    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.title('Evolution of x_t')

    bary_legend = plt.Line2D([0], [0], marker='o', color='green', label='Barycenter',
                            markerfacecolor='green', markersize=8)
    agents_legend = plt.Line2D([0], [0], marker='x', color='red', linestyle='None',
                            label="Agents' Positions")
    targets_legend = plt.Line2D([0], [0], marker='x', color='blue', linestyle='None',
                            label="Targets' Positions")
    plt.legend(handles=[bary_legend, agents_legend, targets_legend])

    plt.show()

if(load):
    with open(load_file, 'rb') as f:
        data = pickle.load(f)

    # Access individual variables
    NN = data["NN"]
    MAX_ITERS = data["MAX_ITERS"]
    costs = data["costs"]
    grad_norms = data["grad_norms"]
    xx = data["xx"]
    rr = data["rr"]
    sigma = data["sigma"]
    sig_est = data["sig_est"]
    std_val = data["std_val"]
    std_grad = data["std_grad"]
    n_plots = data["n_plots"]
    #plot_sigma(NN, MAX_ITERS, sigma, sig_est, n_plots)
    #plot_grad(NN, MAX_ITERS, xx, sigma[:,:,1], rr, costs, grad_norms, std_grad, n_plots)
    #static(NN, MAX_ITERS, xx, sigma[:,:,1], rr, costs, grad_norms, 1)
    
    plot_27_5(NN, MAX_ITERS, xx, sigma[:,:,1], rr, costs, grad_norms, std_val,0, n_plots)
    
    #init(NN, MAX_ITERS, xx, sigma[:,:,1], rr, costs, grad_norms, 1)
