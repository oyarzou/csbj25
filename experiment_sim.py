import numpy as np
import os
import sys

from multiprocessing import Pool
from functools import partial

from ada import *
from classifiers import *


def generate_genephys(spontaneous, evoked):
    import genephys.sampler as sample_data

    # Data dimensions
    n_t = 100
    n_chan = len(evoked['CHAN_PROB'])
    n_trial = 200
    n_cond = 2

    # Train split
    ds = sample_data.DataSampler(n_t, n_chan, n_cond, spontaneous, evoked)
    x, _, _, _, _, _, stimulus = ds.sample(n_trial)
    y = 2 * stimulus[10] - 3

    # Sort trials by label
    sorted_ixs = np.argsort(y)
    y = y[sorted_ixs]
    x = x[:, sorted_ixs]

    # Test split
    ds_test = sample_data.DataSampler(n_t, n_chan, n_cond, spontaneous, evoked)
    x_test, _, _, _, _, _, stimulus_test = ds_test.sample(n_trial)
    y_test = 2 * stimulus_test[10] - 3

    # Sort
    sorted_ixs_test = np.argsort(y_test)
    y_test = y_test[sorted_ixs_test]
    x_test = x_test[:, sorted_ixs_test]

    return x, y, x_test, y_test


def experiment(exp_type, exp_params, n_perm=100, n_chan=40):
    import math

    # Channel relevance prior
    c_prob = .9
    c_relev = 20
    chan_prob = np.array([c_prob]*c_relev + [0]*(n_chan-c_relev))

    # Base algorithm configuration
    cfg_algo = {
        'K': 20,
        'L': 30,
        'lag': 2,
        'step': 4,
        'filtering': .25,
        'th': 4,
        'prediction2': 'Regression'
    }

    # Background process configuration
    spontaneous = {
        "FREQ_RANGE": [.01, math.pi/4],
        "AMP_RANGE": [.5, 2],
        "FREQ_AR_W": .95,
        "AMP_AR_W": .99,
        "MEASUREMENT_NOISE": .5
    }

    # Evoked process configuration
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

    # Initialize output
    out_dict = {
        'cfg_algo': cfg_algo,
        'evoked': evoked,
        'manipulations': exp_params,
        'results': {}
    }

    exp_label, overrides = exp_params
    n_conds = len(overrides)

    for i, cond in enumerate(overrides):
        print(f'computing condition {i}/{n_conds}')
        out_dict['results'][cond] = {}

        # Update config with experiment parameters
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

        for p in range(n_perm):
            x_train, y_train, x_test, y_test = generate_genephys(spontaneous, evoked_i)

            # ADA with multiple base classifiers
            ada_out = {}
            for clf in ['knn', 'svm', 'lda']:
                cfg_ada = {**cfg_algo, 'clf_kind':clf}
                ada_out[clf] = ada(x_train, None, y_train, x_test, None, cfg_ada)

            # KNN
            yhat_knn = knn(x_train, y_train, x_test, cfg_algo)
            acc_knn = np.zeros(yhat_knn.shape[1])
            for jj in range(yhat_knn.shape[1]):
                acc_knn[jj] = np.mean(y_test == yhat_knn[:, jj])
                

            # SVM
            yhat_svm = svm(x_train, y_train, x_test, cfg_algo)
            acc_svm = np.zeros(yhat_svm.shape[1])
            for jj in range(yhat_svm.shape[1]):
                acc_svm[jj] = np.mean(y_test == yhat_svm[:, jj])

            # LDA
            yhat_lda = lda(x_train, y_train, x_test, cfg_algo)
            acc_lda = np.zeros(yhat_lda.shape[1])
            for jj in range(yhat_lda.shape[1]):
                acc_lda[jj] = np.mean(y_test == yhat_lda[:, jj])
        
            # Store results
            out_dict['results'][cond][p] = {
                                                'cond_idx': i,
                                                'cond': cond,
                                                'iteration': p,
                                                'ada': ada_out,
                                                'knn': acc_knn,
                                                'svm': acc_svm,
                                                'lda': acc_lda
                                            }

    return out_dict


def run_experiment():

    experiments = [
        ('DELAY_ABSOLUTE_JITTER', np.arange(1,20,2)),
        ('ADDOA', np.arange(.3,.7,.05)),
        ('CHAN_PROB', np.arange(0,31,3))
        ]

    run_experiment_partial = partial(experiment, 'signal')

    with Pool(processes=len(experiments)) as pool:
        results = pool.map(run_experiment_partial, experiments)

    return results

