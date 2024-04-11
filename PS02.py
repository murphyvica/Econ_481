import numpy as np
import pandas as pd


# ### Exercise 0


def github() -> str:
    """
    Link to github repo
    """

    return "https://"


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






