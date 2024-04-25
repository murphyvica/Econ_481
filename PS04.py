### Exercise 0

def github() -> str:
    """
    Some docstrings.
    """

    return "https://github.com/<user>/<repo>/blob/main/<filename.py>"


### Exercise 1


import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


def load_data() -> pd.DataFrame:
    """
    Some docstrings.
    """
    tesla = pd.read_csv('https://lukashager.netlify.app/econ-481/data/TSLA.csv', index_col='Date', parse_dates=True)

    return tesla


### Exercise 2


def plot_close(df: pd.DataFrame, start: str = '2010-06-29', end: str = '2024-04-15') -> None:
    """
    Some docstrings
    """
    btw = df[(df.index > start) & (df.index < end)]
    
    f = plt.figure()
    a = f.add_subplot()
    a.plot(btw.index, btw.Close)
    a.set_title(f"{start} to {end}")
    
    a.set_xticklabels([])
    a.set_xticks([])


### Exercise 3


def autoregress(df: pd.DataFrame) -> float:
    """
    Some docstrings.
    """
    df['Close_lag'] = df['Close'].shift(1, freq='B')
    df['diff'] = df['Close'] - df['Close_lag']
    df['diff_lag'] = df['diff'].shift(1, freq='B')

    data = df.dropna(subset=['diff_lag'])
    model = smf.ols(formula='diff ~ -1 + diff_lag', data=data)
    result = model.fit(cov_type='HC1')

    return result.tvalues['diff_lag']


### Exercise 4


def autoregress_logit(df: pd.DataFrame) -> float:
    """
    Some docstrings.
    """
    df['Close_lag'] = df['Close'].shift(1, freq='B')
    df['diff'] = df['Close'] - df['Close_lag']
    df['diff_lag'] = df['diff'].shift(1, freq='B')

    data = df.dropna(subset=['diff_lag'])
    data['diff_0'] = (data['diff'] > 0).astype(int)
    model = smf.logit(formula='diff_0 ~ diff_lag', data=data)
    result = model.fit()

    return result.tvalues['diff_lag']


### Exercise 5


def plot_delta(df: pd.DataFrame) -> None:
    """
    Some docstrings.
    """
    df['Close_lag'] = df['Close'].shift(1, freq='B')
    df['diff'] = df['Close'] - df['Close_lag']

    f = plt.figure()
    a = f.add_subplot()
    a.plot(df.index, df['diff'])