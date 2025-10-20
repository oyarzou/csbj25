import numpy as np
import os
import sys
from tqdm import tqdm

from ada import *
from classifiers import *


def sample_trials(X, y, nx, rng):
    # Balanced sampling: take nx trials per class, split for train/test
    ix_c0 = np.where(y == -1)[0]
    ix_c1 = np.where(y == 1)[0]

    n_set = nx // 2
    if len(ix_c0) < nx or len(ix_c1) < nx:
        raise ValueError("Not enough trials per category to sample")

    # Sample without replacement, per class
    samp_c0 = rng.choice(ix_c0, size=nx, replace=False)
    samp_c1 = rng.choice(ix_c1, size=nx, replace=False)

    # Train set
    x_train = np.concatenate((X[:,samp_c0[:n_set]], X[:,samp_c1[:n_set]]), axis=1)
    y_train = np.concatenate(([-1] * n_set, [1] * n_set))

    # Test set
    x_test = np.concatenate((X[:,samp_c0[n_set:]], X[:,samp_c1[n_set:]]), axis=1)
    y_test = np.concatenate(([-1] * n_set, [1] * n_set))

    return x_train, y_train, x_test, y_test


def shift_data(X, sigma, effect=None):
    # Circularly shift each trial using sigma and record shift
    T, N, _ = X.shape

    I = np.round(np.random.normal(0,sigma,size=N)).astype(int)

    X_shifted = np.full(X.shape, np.nan)
    for n in range(N):
        ind = np.roll(np.arange(T), I[n])
        X_shifted[:, n, :] = X[ind, n, :]

    return X_shifted, I


def experiment_real(X, y):
    # Experiment over different temporal jitters (sigma) with repeated resampling

    nx = 104
    n_perm = 100
    sigma_values = np.arange(0, 150, 20)

    # Default decoder configuration
    cfg_algo = {
        'K': 20,
        'L': 30,
        'lag': 2,
        'step': 4,
        'filtering': 0.2,
        'th': 5,
        'metric': 'Cosine',
        'prediction2': 'Regression'
    }

    # Initialize output
    out_dict = {'cfg_default': cfg_algo}
    out_dict['sigma_values'] = sigma_values
    out_dict['results'] = {}

    n_time, n_trial, n_chan = X.shape

    for s,sigma in tqdm(enumerate(sigma_values)):
        out_dict['results'][sigma] = {}
        for p in range(n_perm):
            rng = np.random.default_rng(seed=42 + p + s * 100)

            # Apply temporal jitter to data
            X_shifted, I = shift_data(X, sigma)

            # Train/test split
            x_train, y_train, x_test, y_test = sample_trials(X_shifted, y, nx, rng)

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
            out_dict['results'][sigma][p] = {
                                                'sigma_idx': s,
                                                'sigma': sigma,
                                                'iteration': p,
                                                'ada': ada_out,
                                                'knn': acc_knn,
                                                'svm': acc_svm,
                                                'lda': acc_lda
                                            }

    return out_dict