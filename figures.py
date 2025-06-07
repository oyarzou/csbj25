import numpy as np
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rc("axes.spines", top=False, right=False)
new_rc_params = {"svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)

import genephys.sampler as sample_data
    

def plot_sim_sigma():

    scolors = plt.rcParamsDefault['axes.prop_cycle'].by_key()['color'][:3]

    sigmas=[1,5,20]
    N=100
    x = np.linspace(-10,10,N)
    dx = np.diff(x)[0]

    fig, axes = plt.subplots(2,3)
    for i,sigma in enumerate(sigmas):
        y = np.exp(-x ** 2 / sigma)
        axes[0,i].plot(y,c=scolors[0])
        axes[0,i].xaxis.set_visible(False)
        axes[0,i].set_title(r'$\sigma$ = ' + str(sigma))

        sample = np.round(np.random.randn(N) * sigma + N / 2)
        axes[1,i].scatter(sample,np.arange(0,100,1), color=scolors[0], marker='|')
        axes[1,i].set_xlim(0, 100)
        axes[1,i].set_xlabel('time')

        if i > 0:
            axes[0,i].yaxis.set_visible(False)
            axes[1,i].yaxis.set_visible(False)
        else:
            axes[0,i].set_yticks([0, 1])
            axes[1,i].set_yticks([0, 50, 100])
            axes[0,i].set_ylabel('densitiy')
            axes[1,i].set_ylabel('trial')


def plot_sim_rho():

    scolors = plt.rcParamsDefault['axes.prop_cycle'].by_key()['color'][:3]

    spontaneous = {
        "FREQ_RANGE": [.01, math.pi/4],
        "AMP_RANGE": [.5, 2],
        "FREQ_AR_W": .95,
        "AMP_AR_W": .99,
        "MEASUREMENT_NOISE": .5
    }

    evoked = {
        "phase_reset": False,
        "amplitude_modulation": False,
        "additive_response": False,
        "additive_oscillation": True,
        "CHAN_PROB": 1,
        "DELAY": np.array([30,30]),
        'KERNEL_TYPE': ('Exponential','Exponential'),
        'KERNEL_PAR': (10,10),
        #'ADDOF': ADDOF,
        'ADDOA': np.array([2]*2),
        'ADDOP': np.array([3*math.pi/2,0]),
        'DELAY_ABSOLUTE_JITTER': 2
    }

    rhos = [.2, .4, .6]
    fig, axes = plt.subplots(2,3)
    for i,rho in enumerate(rhos):

        evoked_i = {**evoked, "ADDOA": np.array([rho*5]*2)}

        ds = sample_data.DataSampler(100, 1, 2, spontaneous, evoked_i)
        x, _, _, _, _, _, stimulus = ds.sample(300)
        y = 2 * stimulus[10] - 3

        x1 = x[:,y==-1]
        x2 = x[:,y==1]

        axes[0,i].plot(x1[:,0],c=scolors[0])
        axes[0,i].plot(x2[:,0],c=scolors[1])
        axes[0,i].set_ylim(-3, 4.5)
        axes[0,i].xaxis.set_visible(False)
        axes[0,i].spines['left'].set_visible(False)
        axes[0,i].spines['bottom'].set_visible(False)
        axes[0,i].tick_params(left=False, labelleft=False)
        axes[0,i].set_title(r'$\rho$ = ' + str(rho))

        axes[1,i].plot(np.mean(x1,axis=1),c=scolors[0])
        axes[1,i].plot(np.mean(x2,axis=1),c=scolors[1])
        axes[1,i].set_ylim(-1, 3)
        axes[1,i].spines['left'].set_visible(False)
        axes[1,i].tick_params(left=False, labelleft=False)
        axes[1,i].set_xlabel('time')

    axes[0,0].set_ylabel('single trial')
    axes[1,0].set_ylabel('trial average')


def plot_sim_results(data):
    from matplotlib.ticker import FuncFormatter
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#238B45', '#E69F00'])

    def smart_format(x, pos):
        return f'{x:.0f}' if x == int(x) else f'{x:.1f}'

    n_perm, n_conds, n_models = data['acc'].shape

    models = ['ADA', 'tl-KNN']

    fig, axes = plt.subplots(1, n_conds)
    for ax, (xlabel, xvals) in zip(axes, data['manipulations']):

        step = np.mean(np.diff(xvals))
        jitter = step * .1 * np.random.randn(n_perm, n_conds)
        x = np.tile(xvals, (n_perm, 1)) + jitter

        for m in range(models):
            y = np.squeeze(data['acc'][:,:,:,m])
            ax.scatter(x.flatten(), np.squeeze(y).flatten(),s=2)
            ax.plot(xvals, np.mean(y, axis=0), label=models[m])

        ax.set_xticks(np.linspace(xvals[0], xvals[-1], 5))
        ax.set_xlabel(xlabel)
        ax.set_yticks(np.linspace(.5,1,6))
        show_ylabel = ax == axes[0, 0]
        ax.set_ylabel('accuracy' if show_ylabel else '')
        ax.tick_params(left=True, labelleft=show_ylabel)
        ax.xaxis.set_major_formatter(FuncFormatter(smart_format))
        ax.yaxis.set_major_formatter(FuncFormatter(smart_format))



def plt_real_scheme():

    def shift_data(X, sigma, effect=None):
        ttrial, N, _ = X.shape
        if effect is None:
            effect = 0

        I = np.zeros((N, ttrial), dtype=int)

        X_shifted = np.full(X.shape, np.nan)
        for n in range(N):
            J = int(round(sigma * np.abs(np.random.rand())))
            if J + effect >= ttrial:
                I[n, -1] = 1
            else:
                I[n, effect + J] = 1
            ind = np.roll(np.arange(ttrial), J)
            X_shifted[:, n, :] = X[ind, n, :]
        return X_shifted, I

    scolors = plt.rcParamsDefault['axes.prop_cycle'].by_key()['color'][:3]

    spontaneous = {
        "FREQ_RANGE": [.01, math.pi/4],
        "AMP_RANGE": [.5, 2],
        "FREQ_AR_W": .95,
        "AMP_AR_W": .99,
        "MEASUREMENT_NOISE": .5
    }

    evoked = {
        "phase_reset": False,
        "amplitude_modulation": False,
        "additive_response": False,
        "additive_oscillation": True,
        "CHAN_PROB": 1,
        "DELAY": np.array([30,30]),
        'KERNEL_TYPE': ('Exponential','Exponential'),
        'KERNEL_PAR': (10,10),
        #'ADDOF': ADDOF,
        'ADDOA': np.array([2]*2),
        'ADDOP': np.array([3*math.pi/2,0]),
        'DELAY_ABSOLUTE_JITTER': 2
    }

    ds = sample_data.DataSampler(100, 1, 2, spontaneous, evoked_i)
    x, _, _, _, _, _, stimulus = ds.sample(300)
    y = 2 * stimulus[10] - 3

    sigmas = [1,20]
    fig, axes = plt.subplots(2,2, figsize=(3.5,4))
    for i,sigma in enumerate(sigmas):

        x_shifted, I = shift_data(x, sigma)

        x1 = x_shifted[:,y==-1]
        x2 = x_shifted[:,y==1]

        axes[i,0].plot(x1[:,0],c=scolors[0])
        axes[i,0].plot(x2[:,0],c=scolors[1])
        axes[i,0].set_ylim(-3, 6)
        axes[i,0].xaxis.set_visible(False)
        axes[i,0].spines['left'].set_visible(False)
        axes[i,0].spines['bottom'].set_visible(False)
        axes[i,0].tick_params(left=False, labelleft=False)
        axes[i,0].set_ylabel(r'$\sigma$ = ' + str(sigma))

        axes[i,1].plot(np.mean(x1,axis=1),c=scolors[0])
        axes[i,1].plot(np.mean(x2,axis=1),c=scolors[1])
        axes[i,1].set_xlim(0, 100)
        axes[i,1].set_ylim(-1, 3)
        axes[i,1].spines['left'].set_visible(False)
        axes[i,1].tick_params(left=False, labelleft=False)
        axes[i,1].set_xlabel('Time')

    axes[0,0].set_title('single trial')
    axes[0,1].set_title('trial average')


def plot_real_acc(data):
    from matplotlib.ticker import FuncFormatter

    def smart_format(x, pos):
        return f'{x:.0f}' if x == int(x) else f'{x:.1f}'
    
    acc = data['acc']
    sigma_values = data['sigma_values']

    model_labels = ['ADA', 'tl-KNN']

    n_subs, n_conds, n_perms, n_models = acc.shape

    fig, axes = plt.subplots(1, 4, figsize=(9,2), sharey=True, sharex=True)
    for s in range(n_subs):
        ax = axes[s]

        for m, model in enumerate(model_labels):
            linestyle = '-' #if model == 'ADA' else '--'
            c_color = '#238B45' if model == 'ADA' else '#E69F00'

            dist = 7 * m - 3.5
            dat = acc[s,:,:,m]
            d_mean = dat.mean(axis=1)             

            if s == 0:
                ax.plot(sigma_values+dist, d_mean, linestyle=linestyle, alpha=.8, label=model)
                handles, labels = ax.get_legend_handles_labels()
            else:
                ax.plot(sigma_values+dist, d_mean, linestyle=linestyle, alpha=.8)
            
            for i in range(len(sigma_values)):
                jitter = np.random.normal(loc=0, scale=.7, size=n_perms)
                ax.scatter(np.full(n_perms, sigma_values[i] + dist) + jitter, 
                        dat[i,:],
                        color=c_color,
                        alpha=.2,
                        s=5)
            
            ax.axhline(.5, c='grey', alpha=.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks(sigma_values[::2])
            ax.yaxis.set_major_formatter(FuncFormatter(smart_format))
            ax.set_xlabel(r'$\sigma$')
            if s == 0:
                ax.set_ylabel('accuracy')
    fig.legend(handles, labels, loc='upper center', ncol=n_models, frameon=False, title='Model', bbox_to_anchor=(.5,1.1))
    

def plt_real_hhat(data):
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors   

    Hhat = data['Hhat']
    sigma_values = data['sigma_values']
    n_sigma = len(sigma_values)

    n_sub, n_sigma, n_perm, n_win, n_trial = Hhat.shape

    cmap = plt.cm.cividis(np.linspace(0, 1, n_sigma))
    fig, axes = plt.subplots(1, 4, figsize=(10,2), sharey=True, sharex=True)
    axes = axes.flatten()
    for sub in range(n_sub):
        ax = axes[sub]
        row, col = divmod(sub,2)

        for s, sigma in enumerate(sigma_values):
            h_mean = np.mean(Hhat[sub,s], axis=(0,2))
            ax.plot(h_mean, c=cmap[s])

        ax.set_xlabel('window')
        if sub == 0:
            ax.set_ylabel('average $\hat{H}$')
    norm = mcolors.Normalize(vmin=0)
    sm = cm.ScalarMappable(cmap=plt.cm.cividis, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, label='Temporal variability ($\sigma$)')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['min', 'max'])
