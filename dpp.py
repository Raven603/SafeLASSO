"""Dual Polytope Projection Rules"""

import time
import numpy as np
import scipy as sp
import sklearn.linear_model as skl
import sklearn.preprocessing as skp
import sklearn.datasets as skd


# SDPP
def sdpp(X, y, *, this_alpha, last_alpha, last_w):
    this_lamb = X.shape[0] * this_alpha
    last_lamb = X.shape[0] * last_alpha
    lhs = np.abs(X.T @ (y - X @ last_w)) / this_lamb
    # NOTE: X has norm 1
    rhs = 1 - np.linalg.norm(y) * (1 / this_lamb - 1 / last_lamb)
    return (lhs >= rhs).flatten()


# DPP
def dpp(X, y, *, this_alpha, **params):
    alpha0 = params.get("alpha0",
                        np.max(np.abs(X.T @ y)) / X.shape[0])
    w0 = params.get("w0",
                    sp.sparse.csr_matrix(np.zeros((X.shape[1], 1))))
    return sdpp(X, y,
                last_w=w0,
                last_alpha=alpha0,
                this_alpha=this_alpha)


# DPP LASSO
def dpp_lasso(clf, X, y, dpp_type="sequential", **params):
    this_alpha = params["this_alpha"]

    if dpp_type == "sequential":
        def select(X, y):
            return sdpp(X, y, this_alpha=this_alpha,
                        last_alpha=params["last_alpha"],
                        last_w=params["last_w"])
    else:
        def select(X, y):
            return dpp(X, y, this_alpha=this_alpha,
                       alpha0=params.get("alpha0"),
                       w0=params.get("w0"))

    is_active = select(X, y)
    num_iter = 0

    if is_active.any():
        clf.fit(X[:, is_active], y)
        num_iter = clf.n_iter_

    coef = sp.sparse.lil_matrix(np.zeros((X.shape[1], 1)))
    if hasattr(clf, "coef_"):
        coef[is_active] += clf.sparse_coef_.T

    return (coef, is_active.sum(), num_iter)


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


    # Basic
    dpp_path = []
    clf = skl.Lasso(fit_intercept=False)
    for alpha in alphas:
        clf.set_params(alpha=alpha)
        (_, active_num, num_iter), proc_time = timeit(
            lambda: dpp_lasso(
                clf, X, y,
                dpp_type="basic",
                this_alpha=alpha,
                alpha0=alpha0,
                w0=w0))
        dpp_path.append((active_num, proc_time, num_iter))

    # Sequential
    sdpp_path = []
    clf = skl.Lasso(fit_intercept=False)
    last_alpha = alpha0
    last_w = w0
    for alpha in alphas:
        clf.set_params(alpha=alpha)
        (coef, active_num, num_iter), proc_time = timeit(
            lambda: dpp_lasso(
                clf, X, y,
                dpp_type="sequential",
                this_alpha=alpha,
                last_alpha=last_alpha,
                last_w=last_w))
        sdpp_path.append((active_num, proc_time, num_iter))
        last_alpha = alpha
        last_w = coef

    results = [(b[0], b[1] / l[1], b[2],
                s[0], s[1] / l[1], s[2])
               for (l, b, s)
               in zip(lasso_path, dpp_path, sdpp_path)]
