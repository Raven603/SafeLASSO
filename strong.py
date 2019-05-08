"""Strong rules for feature elimination."""

import time
import numpy as np
import scipy as sp
import sklearn.linear_model as skl
import sklearn.preprocessing as skp
import sklearn.datasets as skd


# Time
def timeit(subr):
    start = time.process_time()
    result = subr()
    end = time.process_time()
    return (result, end - start)


# Check KKT conditions
def check_kkt(X, y, alpha, active_set):
    """Check if there are any violations on the KKT conditions."""
    lamb = X.shape[0] * alpha
    lhs = np.abs(X.T @ y).flatten()
    return np.logical_and(
        lhs > lamb,
        np.logical_not(active_set))


# Sequential Strong rules
def sequential_strong(X, y, *, this_alpha, last_alpha, last_w):
    this_lamb = X.shape[0] * this_alpha
    last_lamb = X.shape[0] * last_alpha
    lhs = np.abs(X.T @ (y - X @ last_w))
    rhs = 2 * this_lamb - last_lamb
    return (lhs >= rhs).flatten()


# Onetime Strong rules
def onetime_strong(X, y, *, this_alpha, **params):
    alpha0 = params.get("alpha0",
                        np.max(np.abs(X.T @ y)) * X.shape[0])
    w0 = params.get("w0",
                    sp.sparse.csr_matrix(np.zeros((X.shape[1], 1))))
    return sequential_strong(X, y,
                             last_w=w0,
                             last_alpha=alpha0,
                             this_alpha=this_alpha)


# Strong LASSO
def strong_lasso(clf, X, y, strong_type="sequential", **params):
    this_alpha = params["this_alpha"]

    if strong_type == "sequential":
        def select(X, y):
            return sequential_strong(X, y, this_alpha=this_alpha,
                                     last_alpha=params["last_alpha"],
                                     last_w=params["last_w"])
    else:
        def select(X, y):
            return onetime_strong(X, y, this_alpha=this_alpha,
                                  alpha0=params.get("alpha0"),
                                  w0=params.get("w0"))

    is_active = select(X, y)
    num_iter = 0

    if is_active.any():
        clf.fit(X[:, is_active], y)
        num_iter += clf.n_iter_

    is_incorrect = check_kkt(X, y, this_alpha, is_active)

    while np.any(is_incorrect):
        is_active = np.logical_or(is_active, is_incorrect)
        if is_active.any():
            clf.fit(X[:, is_active], y)
            num_iter += clf.n_iter_
        is_incorrect = check_kkt(X, y, this_alpha, is_active)

    total_iter = num_iter
    active_num = is_active.sum()

    coef = sp.sparse.lil_matrix(np.zeros((X.shape[1], 1)))

    if hasattr(clf, "coef_"):
        coef[is_active] += clf.sparse_coef_.T

    return (coef, active_num, total_iter)

# Experiments
if __name__ == "__main__":
    data = skd.make_regression(
        n_samples=200,
        n_features=100000,
        n_informative=500,
        effective_rank=1000,
        random_state=0,
        coef=True)
    alphas = np.logspace(-1.05, -2.5, num=200)

    X, y, *_ = data
    X, y, _, _, _ = skl.base._preprocess_data(
        X, y,
        fit_intercept=True,
        normalize=True)
    y = y.reshape((-1, 1))

    alpha0 = alphas[0]
    w0 = sp.sparse.csr_matrix(np.zeros((X.shape[1], 1)))

    # Bare LASSO
    lasso_path = []
    clf = skl.Lasso(fit_intercept=False)
    for alpha in alphas:
        clf.set_params(alpha=alpha)
        clf, proc_time = timeit(lambda: clf.fit(X, y))
        lasso_path.append((clf.sparse_coef_.count_nonzero(),
                           proc_time))

    # Basic Strong
    basic_strong_path = []
    clf = skl.Lasso(fit_intercept=False)
    for alpha in alphas:
        clf.set_params(alpha=alpha)
        (_, active_num, num_iter), proc_time = timeit(
            lambda: strong_lasso(
                clf, X, y,
                strong_type="basic",
                this_alpha=alpha,
                alpha0=alpha0,
                w0=w0))
        basic_strong_path.append((active_num, proc_time, num_iter))

    sequential_strong_path = []
    clf = skl.Lasso(fit_intercept=False)
    last_alpha = alpha0
    last_w = w0
    for alpha in alphas:
        clf.set_params(alpha=alpha)
        (coef, active_num, num_iter), proc_time = timeit(
            lambda: strong_lasso(
                clf, X, y,
                strong_type="sequential",
                this_alpha=alpha,
                last_alpha=last_alpha,
                last_w=last_w))
        sequential_strong_path.append((active_num, proc_time, num_iter))
        last_alpha = alpha
        last_w = coef

    results = [(bstrong[0], bstrong[1] / lasso[1], bstrong[2],
                sstrong[0], sstrong[1] / lasso[1], sstrong[2])
               for (lasso, bstrong, sstrong)
               in zip(lasso_path, basic_strong_path, sequential_strong_path)]
