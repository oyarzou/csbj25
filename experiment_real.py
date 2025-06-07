import numpy as np
import os
import sys
from tqdm import tqdm

from ada import *
from knn import *


def load_mat(filepath):
    from scipy.io import loadmat

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    return loadmat(filepath, struct_as_record=False, squeeze_me=True)


def sample_trials(X, y, nx, rng):
    ix_c0 = np.where(y == -1)[0]
    ix_c1 = np.where(y == 1)[0]

    n_set = nx // 2
    if len(ix_c0) < nx or len(ix_c1) < nx:
        raise ValueError("Not enough trials per category to sample")

    samp_c0 = rng.choice(ix_c0, size=nx, replace=False)
    samp_c1 = rng.choice(ix_c1, size=nx, replace=False)

    x_train = np.concatenate((X[:,samp_c0[:n_set]], X[:,samp_c1[:n_set]]), axis=1)
    y_train = np.concatenate(([-1] * n_set, [1] * n_set))

    x_test = np.concatenate((X[:,samp_c0[n_set:]], X[:,samp_c1[n_set:]]), axis=1)
    y_test = np.concatenate(([-1] * n_set, [1] * n_set))

    return x_train, y_train, x_test, y_test


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


def experiment_real(X, y):

    nx = 104
    n_perm = 100
    sigma_values = np.arange(0, 150, 20)
    n_sigma = len(sigma_values)

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

    out_dict = {'cfg_default': cfg_algo}
    out_dict['sigma_values'] = sigma_values

    n_time, n_trial, n_chan = X.shape

    nwin = len(range(1, n_time - cfg_algo['L'] + 2, cfg_algo['step']))

    out_dict['acc'] = np.full((n_sigma, n_perm, 2), np.nan)
    out_dict['Hhat'] = np.full((n_sigma, n_perm, nwin), np.nan)

    for s,sigma in tqdm(enumerate(sigma_values)):
        for p in range(n_perm):
            rng = np.random.default_rng(seed=42 + p + s * 100)

            X_shifted, I = shift_data(X, sigma)

            x_train, y_train, x_test, y_test = sample_trials(X_shifted, y, nx, rng)

            # ADA decoding
            yhat_ada, yhat_win_test, betas, theta, Htrain, Htest = ada(x_train, None, y_train, x_test, None, cfg_algo)
            acc_ada = np.mean(y_test == yhat_ada)

            # Weighted kNN (averaged method)
            cfg_knn = {**cfg_algo, 'method': 'average'}
            yhat_wknn, theta_wknn, yhat_win_wknn = w_knn(x_train, y_train, x_test, cfg_knn)
            acc_wknn = np.mean(y_test == yhat_wknn)
            
            out_dict['acc'][s,p,:] = [acc_ada, acc_wknn]
            out_dict['acc'][s,p,:,:] = Htrain

    return out_dict