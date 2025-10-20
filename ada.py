import numpy as np
from scipy.stats import spearmanr


def solve_lsqlin(y, yhat_win_train, n_win):
    from scipy.optimize import minimize, LinearConstraint, Bounds
    # Objective function: Minimize ||yhat_win_train * theta - y||^2
    def objective(theta):
        residual = yhat_win_train @ theta - y
        return np.linalg.norm(residual)**2  # Equivalent to sum of squares
    
    # Constraints
    bounds = Bounds(0, 1)  # 0 ≤ theta ≤ 1 for each element
    sum_constraint = LinearConstraint(np.ones((1, n_win)), 1, 1)  # sum(theta) = 1
    
    # Initial guess: uniform distribution
    theta0 = np.full(n_win, 1.0 / n_win)
    
    # Solve constrained least squares
    res = minimize(objective, theta0, method='trust-constr', bounds=bounds, constraints=[sum_constraint])

    return res.x


def pairwise_correlation(X, Y, method="pearson"):
    from scipy.stats import rankdata
    # If Spearman, convert data to ranks
    if method.lower() == "spearman":
        X = np.apply_along_axis(rankdata, axis=0, arr=X)
        Y = np.apply_along_axis(rankdata, axis=0, arr=Y)

    # Normalize each column: (value - mean) / std
    X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
    Y_norm = (Y - Y.mean(axis=0)) / Y.std(axis=0)

    # Compute correlation using matrix multiplication
    C = np.dot(X_norm.T, Y_norm) / X.shape[0]  # (p, q) matrix

    return C


def compute_response_knn(C, y, K=1):
    # Compute KNN for each row using top K similarities
    N = C.shape[0]
    Q = np.zeros(N)
    for n in range(N):
        c = C[n, :]
        if np.all(c == 0):
            continue
        idx = np.argsort(c, kind='stable')[::-1][:K]    # top K indices
        cd = c[idx]                                     # similarity values
        Q[n] = np.dot(cd, y[idx])                       # weighted vote

    return Q


def q_to_yhat(Q, my, regression):
    # Convert continuous Q to class prediction
    N = len(Q)
    if regression:
        return Q
    my = round(N * my)
    yhat = np.ones(N)
    yhat[np.argsort(Q, kind='stable')[:my]] = -1

    return yhat


def make_decision(C, Y, my, K, I=None, Q=None):
    # Aggregate Q across radius and apply decision rule
    if Q is None:
        radius, N, _ = C.shape
        Q = np.zeros((radius, N))
        has_Q = False
    else:
        radius, N = Q.shape
        has_Q = True

    if I is None:
        I = np.ones((radius, N))

    Qadd = np.zeros(N)
    for jj in range(radius):
        if not has_Q:
            if np.all(I[jj, :] == 0):
                continue
            Cj = C[jj]
            Q[jj, :] = compute_response_knn(Cj, Y, K)
        Qadd += np.squeeze(Q[jj, :] * I[jj, :])
    yhat = q_to_yhat(Qadd, my, 0)

    return yhat, Q


def knn_train(D1, Y, trial_idx, nfeatures, K):
    from scipy.stats import rankdata
    # Compute similarity matrix
    C = D1 @ D1.T
    d = np.sqrt(np.diag(C))
    C = (C / d).T / d

    # Zero out negative and self-similarity for same trial index
    C[C < 0] = 0
    for i in range(C.shape[0]):
        C[i, trial_idx == trial_idx[i]] = 0

    # KNN score
    Qsingle = compute_response_knn(C, Y, K)

    return Qsingle


def knn_test(d_test, D2, K, Y1):
    # Compute KNN scores for test samples
    C = d_test @ D2.T
    dx = np.sqrt(np.sum(d_test ** 2, axis=1))[:, None]
    dy = np.sqrt(np.sum(D2 ** 2, axis=1))[None, :]
    C = C / dx
    C = C / dy
    C[C < 0] = 0

    # For each test vector, take top K weighted votes
    Qj = np.array([
        np.dot(np.sort(c)[::-1][:K], Y1[np.argsort(c)[::-1][:K]])
        for c in C
    ])

    return Qj


def svm_train(D1, Y, C=.5, class_weight=None):
    from sklearn.svm import LinearSVC
    # Compute SVM scores for train samples
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LinearSVC(C=C, class_weight=class_weight, max_iter=10000)
    )

    clf.fit(D1, Y)
    Qsingle = clf.decision_function(D1)

    return Qsingle, clf


def lda_train(D1, Y, shrinkage='auto'):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    # Compute LDA scores for train samples
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LinearDiscriminantAnalysis(solver='lsqr', shrinkage=shrinkage)
    )

    clf.fit(D1, Y)
    Qsingle = clf.decision_function(D1)

    return Qsingle, clf


def ada(X, Z, y, Xtest, Ztest, options={}):
    # Unpack options with defaults
    K = options.get('K', 20)
    L = options.get('L', 25)
    lag = options.get('lag', 1)
    step = options.get('step', 2)
    th = options.get('th', 0)
    radius = options.get('radius', 0)
    step2 = options.get('step2', 1)
    metric = options.get('metric', 'Cosine')
    alpha = options.get('alpha', 0.1)
    filtering = options.get('filtering', 0)
    clf_kind = options.get('clf_kind', 'knn')

    # Validate threshold
    if th < 0 or (0 < th < 1):
        raise ValueError("th must be a non-negative integer or 0")

    # Basic dims
    ttrial, N, p = X.shape
    Ntest = Xtest.shape[1]
    nwin = len(range(0, ttrial - L + 1, step))
    winsize = len(range(0, L, lag))
    nfeatures = p * winsize

    # By default, radius = number of windows
    radius = nwin if radius == 0 else radius
    if radius > nwin:
        raise ValueError("radius must be <= nwin")
    if th > radius:
        raise ValueError("th cannot be greater than radius")

    # Sliding window offsets
    tt = list(range(0, nwin - radius + 1, step2)) if radius < nwin else [0]
    nwin2 = len(tt)

    def make_windows(X):
        # Extract windowed features
        D = np.zeros((nwin, N, winsize, p))
        for n in range(N):
            for j in range(p):
                for nw, t in enumerate(range(0, ttrial - L + 1, step)):
                    D[nw, n, :, j] = X[t:t+L:lag, n, j]
        return D

    # Build train and test window representations
    D = make_windows(X)
    DZ = make_windows(Z if Z is not None else X)
    Dtest = make_windows(Xtest)
    DZtest = make_windows(Ztest if Ztest is not None else Xtest)

    # Expand y to match window count
    Y = np.repeat(y, radius)
    my = np.mean(y == -1)
    trial_idx = np.repeat(np.arange(N), radius)

    # Initialize outputs
    beta = np.zeros((nfeatures + 1, nwin2))
    r2 = np.zeros(nwin2)
    yhat_win_train = np.ones((N, nwin2))
    yhat_win_test = np.ones((Ntest, nwin2))
    Htrain = np.zeros((radius, N, nwin2), dtype=bool)
    Htest = np.zeros((radius, Ntest, nwin2), dtype=bool)
    Qtrain = np.zeros((radius, N, nwin2))
    Qtest = np.zeros((radius, Ntest, nwin2))
    Acchat_test = np.zeros((radius, Ntest, nwin2))
    clf_win = [None] * nwin2

    # Train
    for j, t0 in enumerate(tt):
        win_idx = [t0 + i for i in range(radius)]
        D1 = D[win_idx].reshape(radius * N, nfeatures, order='F')
        D1Z = DZ[win_idx].reshape(radius * N, nfeatures, order='F')

        # Train classifier
        if clf_kind == 'knn':
            Qsingle = knn_train(D1, Y, trial_idx, nfeatures, K, metric)
        elif clf_kind == 'svm':
            Qsingle, clf_win[j] = svm_train(D1, Y,
                                C=options.get('C_svm', 0.5),
                                class_weight=options.get('class_weight', None))
            last = list(clf_win[j].named_steps.values())[-1]
            print(f"[{clf_kind}] j={j} -> {type(last).__name__}")
        elif clf_kind == 'lda':
            Qsingle, clf_win[j] = lda_train(D1, Y,
                                shrinkage=options.get('shrinkage', 'auto'))
            last = list(clf_win[j].named_steps.values())[-1]
            print(f"[{clf_kind}] j={j} -> {type(last).__name__}")
        else:
            raise ValueError("options['clf'] must be 'knn', 'svm', or 'lda'")
        
        # Convert scores to label
        yhat_tmp = q_to_yhat(Qsingle, my, False)

        if radius > 1:
            Acctrain = np.abs(Qsingle)
            Acctrain[yhat_tmp != Y] *= -1

            # Optional feature filtering via correlation
            if filtering:
                nsel = round(filtering * nfeatures) if filtering < 1 else filtering
                f = np.argsort(np.abs(pairwise_correlation(D1Z, Acctrain)), kind='stable')[::-1][:nsel]
            else:
                f = np.arange(nfeatures)

            # Add intercept
            D1Z = np.hstack([np.ones((radius * N, 1)), D1Z])
            f = np.hstack([0, f + 1])

            # Ridge regression to learn beta
            R = alpha * np.eye(len(f))
            R[0, 0] = 0
            beta[f, j] = np.linalg.solve(D1Z[:, f].T @ D1Z[:, f] + R, D1Z[:, f].T @ Acctrain)
            r2[j] = 1 - np.sum((D1Z[:, f] @ beta[f, j] - Acctrain) ** 2) / np.sum((np.mean(Acctrain) - Acctrain) ** 2)

            # Convert scores into window selections Htrain
            Acctrain = Acctrain.reshape((radius, N), order='F')
            for n in range(N):
                if th > 0:
                    thresh = np.sort(Acctrain[:, n], kind='stable')[::-1][th-1]
                    sel = Acctrain[:, n] >= thresh
                else:
                    sel = Acctrain[:, n] > 0
                if not np.any(sel):
                    sel[np.random.randint(radius)] = True
                Htrain[:, n, j] = sel
            Qsingle = Qsingle.reshape((radius, N), order='F')
            Qtrain[:, :, j] = Qsingle
            yhat_win_train[:,j], _ = make_decision(None, None, my, K, Htrain[:, :, j], Qtrain)

        else:
            # Single window = trivial selection
            Htrain[:,:,j] = True
            yhat_win_train[:, j] = yhat_tmp

    # Solve for theta weights if superwindows
    theta = solve_lsqlin(y, yhat_win_train, nwin2)

    # Test
    for j, t0 in enumerate(tt):
        win_idx = [t0 + i for i in range(radius)]
        D1 = D[win_idx].reshape(radius * N, nfeatures, order='F')
        D1test = Dtest[win_idx].reshape(radius * Ntest, nfeatures, order='F')
        D1Ztest = DZtest[win_idx].reshape(radius * Ntest, nfeatures, order='F')

        # Select informative windows
        if radius > 1 and th < radius:
            Acchat = np.dot(np.hstack([np.ones((radius * Ntest, 1)), D1Ztest]), beta[:, j]).reshape((radius, Ntest), order='F')
            for n in range(Ntest):
                if th > 0:
                    thresh = np.sort(Acchat[:, n], kind='stable')[::-1][th-1]
                    sel = Acchat[:, n] >= thresh
                else:
                    sel = Acchat[:, n] > 0
                if not np.any(sel):  # safeguard
                    sel[np.random.randint(radius)] = True
                Htest[:, n, j] = sel
        else:
            Htest[:, :, j] = True

        I1 = Htrain[:, :, j].flatten(order='F')
        Y1 = Y[I1] if radius > 1 and th < radius else Y
        D2 = D1[I1, :] if radius > 1 and th < radius else D1

        Qfull = np.zeros((radius, Ntest))

        # Compute scores for test samples
        for n in range(Ntest):
            sel = Htest[:, n, j]
            if not np.any(sel):
                continue
            d_test = D1test[n*radius:(n+1)*radius][sel]

            if clf_kind == 'knn':
                Qfull[sel, n] = knn_test(d_test, D2, K, Y1)
            else:
                Qfull[sel, n] = clf_win[j].decision_function(d_test)

        Qadd = np.sum(Qfull * Htest[:,:,j], axis=0)
        yhat_win_test[:, j] = q_to_yhat(Qadd, my, regression=False)
        Qtest[:,:,j] = Qfull
        Acchat_test[:,:,j] = Acchat

    # Final aggregation
    Qfinal = yhat_win_test @ theta
    yhat = np.ones(Ntest)
    yhat[np.argsort(Qfinal, kind='stable')[:round(Ntest * my)]] = -1

    return yhat, yhat_win_test, beta, theta, Htrain, Htest
