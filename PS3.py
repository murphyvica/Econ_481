### Exercise 0

def github() -> str:
    """
    Some docstrings.
    """

    return "https://github.com/murphyvica/Econ_481/blob/main/PS3.py"


### Exercise 1

import pandas as pd

def import_yearly_data(years: list) -> pd.DataFrame:
    """
    Some docstrings.
    """
    df = pd.DataFrame()
    for y in years:
        url = f"https://lukashager.netlify.app/econ-481/data/ghgp_data_{y}.xlsx"
        new = pd.read_excel(url, sheet_name = 'Direct Emitters', engine='openpyxl', skiprows=3)
        new['year'] = y
        df = pd.concat([df, new])
    return df


### Exercise 2

def import_parent_companies(years: list) -> pd.DataFrame:
    """
    Some docstrings.
    """
    df2 = pd.DataFrame()
    url = f"https://lukashager.netlify.app/econ-481/data/ghgp_data_parent_company_09_2023.xlsb"
    for y in years:
         new = pd.read_excel(url, sheet_name = str(y), engine='pyxlsb')
         new['year'] = y
         df2 = pd.concat([df2, new])
    df2 = df2.dropna(how='all') 
    return df2


### Exercise 3

def n_null(df: pd.DataFrame, col: str) -> int:
    """
    Some docstrings
    """
    count = pd.isna(df[col]).sum()

    return count


### Exercise 4

def clean_data(emissions_data: pd.DataFrame, parent_data: pd.DataFrame) -> pd.DataFrame:
    """
    Some docstrings.
    """
    cat = pd.merge(emissions_data, parent_data, left_on=['year', 'Facility Id'], right_on=['year', 'GHGRP FACILITY ID'])
    subset_cat = cat[['Facility Id', 'year', 'State', 'Industry Type (sectors)', 'Total reported direct emissions', 'PARENT CO. STATE', 'PARENT CO. PERCENT OWNERSHIP']]
    subset_cat.columns = subset_cat.columns.str.lower()

    return subset_cat


### Exercise 5

def aggregate_emissions(df: pd.DataFrame, group_vars: list) -> pd.DataFrame:
    """
    Some docstrings.
    """
    ag = df.groupby(group_vars, as_index=True)

    avg = ag[['total reported direct emissions', 'parent co. percent ownership']].mean()
    avg.columns = ['total reported direct emissions (mean)', 'parent co. percent ownership (mean)']
    
    med = ag[['total reported direct emissions', 'parent co. percent ownership']].median()
    med.columns = ['total reported direct emissions (median)', 'parent co. percent ownership (median)']
    
    minn = ag[['total reported direct emissions', 'parent co. percent ownership']].min()
    minn.columns = ['total reported direct emissions (min)', 'parent co. percent ownership (min)']
    
    maxx = ag[['total reported direct emissions', 'parent co. percent ownership']].max()
    maxx.columns = ['total reported direct emissions (max)', 'parent co. percent ownership (max)']

    return pd.concat([avg, med, minn, maxx], axis=1).sort_values(by='total reported direct emissions (mean)', ascending=False)
