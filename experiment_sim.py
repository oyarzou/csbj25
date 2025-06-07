import numpy as np
import os
import sys

from multiprocessing import Pool
from functools import partial

from ada import *
from knn import *


def generate_genephys(spontaneous, evoked):
    import genephys.sampler as sample_data

    n_t = 100
    n_chan = len(evoked['CHAN_PROB'])
    n_trial = 200
    n_cond = 2

    ds = sample_data.DataSampler(n_t, n_chan, n_cond, spontaneous, evoked)
    x, _, _, _, _, _, stimulus = ds.sample(n_trial)
    y = 2 * stimulus[10] - 3

    sorted_ixs = np.argsort(y)
    y = y[sorted_ixs]
    x = x[:, sorted_ixs]

    ds_test = sample_data.DataSampler(n_t, n_chan, n_cond, spontaneous, evoked)
    x_test, _, _, _, _, _, stimulus_test = ds_test.sample(n_trial)
    y_test = 2 * stimulus_test[10] - 3

    sorted_ixs_test = np.argsort(y_test)
    y_test = y_test[sorted_ixs_test]
    x_test = x_test[:, sorted_ixs_test]

    return x, y, x_test, y_test


def experiment(exp_type, exp_params, n_rep=20, n_chan=40):
    import math

    c_prob = .9
    c_relev = 20
    chan_prob = np.array([c_prob]*c_relev + [0]*(n_chan-c_relev))

    cfg_algo = {
        'K': 20,
        'L': 30,
        'lag': 2,
        'step': 4,
        'filtering': .25,
        'th': 4,
        'prediction2': 'Regression'
    }

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
        "CHAN_PROB": chan_prob,
        "DELAY": np.array([40,45]),
        'KERNEL_TYPE': ('Exponential','Exponential'),
        'KERNEL_PAR': (20,20),
        'ADDOA': np.array([2]*2),
        'ADDOP': np.array([-2,0]),
        'DELAY_ABSOLUTE_JITTER': 20
    }

    out_dict = {
        'cfg_algo': cfg_algo,
        'evoked': evoked
    }

    exp_label, overrides = exp_params

    n_conds = len(overrides)

    acc = np.zeros((n_rep, n_conds, 4))
    for i, cond in enumerate(overrides):
        print(f'computing condition {i}/{n_conds}')

        if exp_type == 'signal':
            if exp_label == 'DELAY_ABSOLUTE_JITTER':
                param = cond
            elif exp_label == 'ADDOA':
                param = np.array([1+cond]*2)
            elif exp_label == 'CHAN_PROB':
                param = np.array([c_prob]*cond + [0]*(n_chan-cond))
            evoked_i = {**evoked, exp_label: param}
        else:
            evoked_i = evoked

        if exp_type == 'parameters':
            cfg_i = {**cfg_algo, exp_label: cond}
        else:
            cfg_i = cfg_algo

        for r in range(n_rep):
            X, y, Xtest, ytest = generate_genephys(spontaneous, evoked_i)

            yhat_ada, yhat_win_test, betas, theta, Htrain, Htest = ada(X, None, y, Xtest, None, cfg_i)
            acc_ada = np.mean(ytest == yhat_ada)

            yhat_cknn = knn(X, y, Xtest, cfg_i)
            acc_cknn = 0
            for jj in range(1, yhat_cknn.shape[1]):
                acc_cknn = max(acc_cknn, np.mean(ytest == yhat_cknn[:, jj]))

            cfg_i['method'] = 'average'
            yhat_wknn_avg, _, _ = w_knn(X, y, Xtest, cfg_i)
            acc_wknn_avg = np.mean(ytest == yhat_wknn_avg)

            acc[r,i,:] = [acc_ada, acc_cknn, acc_wknn_avg]

    out_dict['acc'] = acc
    out_dict['manipulations'] = exp_params

    return out_dict


def run_experiment():

    experiments = [
        ('DELAY_ABSOLUTE_JITTER', np.arange(1,20,2)),
        ('ADDOA', np.arange(.3,.7,.05)),
        ('CHAN_PROB', np.arange(0,31,3))
        ]

    run_experiment_partial = partial(experiment)

    with Pool(processes=len(experiments)) as pool:
        results = pool.map(run_experiment_partial, 'signal', experiments)

    return results

