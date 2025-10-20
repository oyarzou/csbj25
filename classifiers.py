import numpy as np


def lda(X, y, Xtest, options=None):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    if options is None:
        options = {}

    L = options.get('L', 25)
    lag = options.get('lag', 1)
    step = options.get('step', 2)

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
        D1test = Dtest[j].reshape(Ntest, p * winsize, order='F')
        D1 = D[j].reshape(N, p * winsize, order='F')

        classifier = LinearDiscriminantAnalysis(solver='lsqr', 
                                                shrinkage='auto')

        classifier.fit(D1, y)
        yhat_win[:,j] = classifier.predict(D1test)

    return yhat_win


def svm(X, y, Xtest, options=None):
    from sklearn.svm import LinearSVC

    if options is None:
        options = {}

    K = options.get('K', 10)
    L = options.get('L', 25)
    lag = options.get('lag', 1)
    step = options.get('step', 2)

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
        D1test = Dtest[j].reshape(Ntest, p * winsize, order='F')
        D1 = D[j].reshape(N, p * winsize, order='F')

        classifier = LinearSVC(penalty = 'l2',
                        loss = 'hinge',
                        C = .5,
                        multi_class = 'ovr',
                        fit_intercept = True,
                        max_iter = 10000)
        classifier.fit(D1, y)
        yhat_win[:,j] = classifier.predict(D1test)

    return yhat_win


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
