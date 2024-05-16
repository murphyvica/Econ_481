import numpy as np
import pandas as pd


# ### Exercise 0


def github() -> str:
    """
    Link to github repo
    """

    return "https://github.com/murphyvica/Econ_481/blob/main/PS02.py"


# ### Exercise 1


def simulate_data(seed: int = 481) -> tuple:
    """
    Generated matrices y and X
    """
    rng = np.random.default_rng(seed=seed)
    x1 = rng.normal(loc = 0, scale = np.sqrt(2), size = 1000)
    x2 = rng.normal(loc = 0, scale = np.sqrt(2), size = 1000)
    x3 = rng.normal(loc = 0, scale = np.sqrt(2), size = 1000)
    e = rng.standard_normal(size = 1000)

    X = np.c_[x1.reshape(-1, 1), x2.reshape(-1, 1), x2.reshape(-1, 1)]
    y = 5 + 3*x1 + 2*x2 + 6*x3 + e
    
    return (y, X)


# ### Exercise 2

import scipy as sp

def estimate_mle(y: np.array, X: np.array) -> np.array:
    """
    optimizes the coefficients b0, b1, b2, b3 to have the max log likelyhood
    """
    X = np.c_[np.ones(1000).reshape(-1, 1), X]

    betas = sp.optimize.minimize(
        fun=neg_mle, # the objective function
        x0=[0, 0, 0, 0], # starting guess
        args=(X,y), # additional parameters passed to neg_ll
        method = 'Nelder-Mead' # optionally pick an algorithm
        )
    print(betas)
    return betas



def neg_mle(b, X: np.array, y: np.array) -> float:
    """
    calculates the negative log likelyhood
    """
    residuals = y - np.dot(X, b) 
    nll = 0.5 * np.sum(residuals**2)  
    return nll



# ### Exercise 3


def estimate_ols(y: np.array, X: np.array) -> np.array:
    """
    Uses scipy library to estimate ols by minimizing the sum of squared residuals
    """
    X = np.c_[np.ones(1000).reshape(-1, 1), X]

    betas = sp.optimize.minimize(
        fun=sse, # the objective function
        x0=[0, 0, 0, 0], # starting guess
        args=(X,y), # additional parameters passed to neg_ll
        method = 'Nelder-Mead' # optionally pick an algorithm
        )
    return betas


def sse(b, X: np.array, y: np.array) -> float:
    """
    Calculated the sum of squares error
    """
    e = y - (X @ b)

    return np.sum(e**2)

### TESTS

# from sols_02 import *
import numpy as np
from scipy.optimize import minimize
import re
import requests

rng = np.random.default_rng(seed=1)

X = rng.normal(scale=np.sqrt(2), size = (1000, 3))
y = 5 + X @ np.array([3.,2.,6.]) + rng.standard_normal(1000)

def test_github():
    url = github()
    repo_url = re.search('github\\.com/(.+)/blob', url).group(1)
    req = requests.get(f'https://api.github.com/repos/{repo_url}/stats/participation')
    assert req.json()['all'][-1] > 0

def test_simulation_shapes():
    assert simulate_data()[0].shape in [(1000,), (1000,1)]
    assert simulate_data()[1].shape == (1000,3)

def test_simulation_means():
    assert np.allclose(np.mean(simulate_data()[0]), 5., atol=3*np.sqrt(((9+4+36)*2+1)/1000), rtol=0)
    assert np.allclose(np.mean(simulate_data()[1]), 0., atol=3*np.sqrt(2/3000), rtol=0)

def test_simulation_vars():
    assert np.allclose(np.var(simulate_data()[0]), (9+4+36)*2+1, rtol=.1, atol=0)

def test_mle_coef_shape():
    assert estimate_mle(y,X).shape in [(4,), (4,1)]

def test_mle_coef_vals():
    assert np.allclose(estimate_mle(y,X).flatten(), np.array([5., 3., 2., 6.]), atol=0, rtol=.25)

def test_mle_coef_vals_unseen():
    beta = np.array([6., 5, -5, 2])
    y_new = beta[0] + X @ beta[1:] + rng.normal(size=1000,scale=3)
    assert np.allclose(estimate_mle(y_new, X).flatten(), beta, atol=0, rtol=.25)

def test_mle_coef_vals_unseen_2():
    beta = np.array([1., -2, -1., 2])
    y_new = beta[0] + X @ beta[1:] + rng.normal(size=1000,scale=3)
    assert np.allclose(estimate_mle(y_new, X).flatten(), beta, atol=0, rtol=.25)

def test_ols_coef_shape():
    assert estimate_ols(y,X).shape in [(4,), (4,1)]

def test_ols_coef_vals():
    assert np.allclose(estimate_ols(y,X).flatten(), np.array([5., 3., 2., 6.]), atol=0, rtol=.25)

def test_ols_coef_vals_unseen():
    beta = np.array([6., 5, -5, 2])
    y_new = beta[0] + X @ beta[1:] + rng.normal(size=1000,scale=3)
    assert np.allclose(estimate_ols(y_new, X).flatten(), beta, atol=0, rtol=.25)

def test_ols_coef_vals_unseen_2():
    beta = np.array([1., -2, -1., 2])
    y_new = beta[0] + X @ beta[1:] + rng.normal(size=1000,scale=3)
    assert np.allclose(estimate_ols(y_new, X).flatten(), beta, atol=0, rtol=.25)


test_github()
test_simulation_shapes()
test_simulation_means()
test_simulation_vars()
estimate_mle(y,X)
#test_mle_coef_shape()
print("ran")
