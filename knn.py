import numpy as np


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
    # Solve the constrained least squares problem
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


def compute_response(Q, my):
    N = len(Q)
    my = round(N * my)
    yhat = np.ones(N)
    ord_idx = np.argsort(Q, kind='stable')
    yhat[ord_idx[:my]] = -1
    return yhat


def knn(X, y, Xtest, options=None):
    if options is None:
        options = {}

    K = options.get('K', 10)
    L = options.get('L', 25)
    lag = options.get('lag', 1)
    step = options.get('step', 2)
    metric = options.get('metric', 'Cosine')
    use_proportions = options.get('useProportions', True)

    ttrial, N, p = X.shape
    Ntest = Xtest.shape[1]

    nwin = len(range(1, ttrial - L + 2, step))
    winsize = len(range(0, L, lag))

    # Make the windowed samples
    D = np.zeros((nwin, N, winsize, p))
    for n in range(N):
        for j in range(p):
            nw = 0
            for t in range(0, ttrial - L + 1, step):
                D[nw, n, :, j] = X[t:t+L:lag, n, j]
                nw += 1

    # Class proportions
    my = np.mean(y == -1)

    # Make the windowed samples in testing
    Dtest = np.zeros((nwin, Ntest, winsize, p))
    for n in range(Ntest):
        for j in range(p):
            nw = 0
            for t in range(0, ttrial - L + 1, step):
                Dtest[nw, n, :, j] = Xtest[t:t+L:lag, n, j]
                nw += 1

    yhat_win = np.zeros((Ntest, nwin))
    for j in range(nwin):
        D1test = Dtest[j].reshape(Ntest, p * winsize, order='F').T
        D1 = D[j].reshape(N, p * winsize, order='F').T

        if metric == 'Pearson':
            Ctest = pairwise_correlation(D1test, D1)
        elif metric == 'Spearman':
            Ctest = pairwise_correlation(D1test, D1, method='spearman')
        elif metric == 'Cosine':
            Ctest = D1test.T @ D1
            dx = np.sqrt(np.sum(D1test ** 2, axis=0))
            dy = np.sqrt(np.sum(D1 ** 2, axis=0))
            Ctest = (Ctest.T / dx).T / dy

        Ctest[Ctest < 0] = 0

        Qtest = np.zeros(Ntest)
        for n in range(Ntest):
            c = Ctest[n]
            d = np.argsort(c, kind='stable')[::-1]  # Sort in descending order
            Qtest[n] = np.dot(c[d[:K]], y[d[:K]])

        if use_proportions:
            yhat_win[:, j] = compute_response(Qtest, my)
        else:
            yhat_win[:, j] = np.where(Qtest < 0, -1, 1)

    return yhat_win


def w_knn(X, y, Xtest, options=None):
    from scipy.stats import spearmanr

    if options is None:
        options = {}

    K = options.get('K', 10)
    L = options.get('L', 25)
    lag = options.get('lag', 1)
    step = options.get('step', 2)
    metric = options.get('metric', 'Cosine')
    method = options.get('method', 'Stacking')
    gamma = options.get('gamma', 0)
    use_proportions = options.get('useProportions', True)

    ttrial, N, p = X.shape
    Ntest = Xtest.shape[1]

    nwin = len(range(1, ttrial - L + 2, step))
    winsize = len(range(0, L, lag))

    # Make the windowed samples
    D = np.zeros((nwin, N, winsize, p))
    for n in range(N):
        for j in range(p):
            nw = 0
            for t in range(0, ttrial - L + 1, step):
                D[nw, n, :, j] = X[t:t+L:lag, n, j]
                nw += 1

    # Class proportions
    my = np.mean(y == -1)

    # Work out combination
    yhattrain = np.zeros((N, nwin))
    if not (method == 'average' and gamma == 0):
        for j in range(nwin):
            D1 = D[j].reshape(N, p * winsize, order='F')

            if metric == 'Pearson':
                C = np.corrcoef(D1)
            elif metric == 'Spearman':
                C, _ = spearmanr(D1, axis=1)
            elif metric == 'Cosine':
                C = D1 @ D1.T
                d = np.sqrt(np.diag(C))
                C = (C.T / d).T / d

            C[C < 0] = 0
            np.fill_diagonal(C, 0)

            Q = np.zeros(N)
            for n in range(N):
                c = C[n]
                d = np.argsort(c, kind='stable')[::-1]  # Descending order
                Q[n] = np.dot(c[d[:K]], y[d[:K]])

            if use_proportions:
                yhattrain[:, j] = compute_response(Q, my)
            else:
                yhattrain[:, j] = np.where(Q < 0, -1, 1)

    if method == 'average':
        theta = np.ones(nwin) / nwin
        if gamma > 0:
            theta = np.array([np.mean(y == yhattrain[:, j]) for j in range(nwin)])
            theta = theta**gamma
            theta /= np.sum(theta)
    else:
        theta = solve_lsqlin(y, yhattrain, nwin)

    # Make the windowed samples in testing
    Dtest = np.zeros((nwin, Ntest, winsize, p))
    for n in range(Ntest):
        for j in range(p):
            nw = 0
            for t in range(0, ttrial - L + 1, step):
                Dtest[nw, n, :, j] = Xtest[t:t+L:lag, n, j]
                nw += 1

    yhat_win = np.zeros((Ntest, nwin))
    for j in range(nwin):
        D1test = Dtest[j].reshape(Ntest, p * winsize, order='F').T
        D1 = D[j].reshape(N, p * winsize, order='F').T

        if metric == 'Pearson':
            Ctest = pairwise_correlation(D1test, D1)
        elif metric == 'Spearman':
            Ctest = pairwise_correlation(D1test, D1, method='spearman')
        elif metric == 'Cosine':
            Ctest = D1test.T @ D1
            dx = np.sqrt(np.sum(D1test ** 2, axis=0))
            dy = np.sqrt(np.sum(D1 ** 2, axis=0))
            Ctest = (Ctest.T / dx).T / dy

        Ctest[Ctest < 0] = 0

        Qtest = np.zeros(Ntest)
        for n in range(Ntest):
            c = Ctest[n]
            d = np.argsort(c)[::-1]  # Descending order
            Qtest[n] = np.dot(c[d[:K]], y[d[:K]])

        if use_proportions:
            yhat_win[:, j] = compute_response(Qtest, my)
        else:
            yhat_win[:, j] = np.where(Qtest < 0, -1, 1)

        if method == 'average':
            yhat_win[:, j] *= theta[j]

    Qfinal = np.sum(yhat_win, axis=1) if method=='average' else yhat_win @ theta

    yhat = np.ones(Ntest)
    if use_proportions:
        ord_idx = np.argsort(Qfinal, kind='stable')
        yhat[ord_idx[:round(Ntest * my)]] = -1
    else:
        yhat = np.where(Qtest < 0, -1, 1)

    if use_proportions:
        for j in range(nwin):
            ord_idx = np.argsort(yhat_win[:, j], kind='stable')
            yhat_win[:, j] = 1
            yhat_win[ord_idx[:round(Ntest * my)], j] = -1

    return yhat, theta, yhat_win

