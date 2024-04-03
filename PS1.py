### Exercise 0

def github() -> str:
    """
    Some docstrings.
    """

    return "https://github.com/<user>/<repo>/blob/main/<filename.py>"


### Exercise 1

import numpy as np
import pandas as pd
import scipy
import matplotlib
import seaborn

print("success")


### Exercise 2


def evens_and_odds(n: int) -> dict:
    """
    This function takes in an integer, and calculates the sum of the even numbers less than the 
    integers, and the sum of odd numbers less than the int. The returned value is a dict 
    """
    evens = []
    odds = []
    for i in range(1, n):
        if i % 2 == 0:
            evens = np.append(evens, i)
        else:
            odds = np.append(odds, i)
    return ({'evens': np.sum(evens), 'odds': np.sum(odds)})
print(evens_and_odds(4))


### Exercise 3


from typing import Union
from datetime import datetime


def time_diff(date_1: str, date_2: str, out: str) -> Union[str,float]:
    """
    The function takes in 2 days as strings, and computes the number of days between the two and 
    returns the result. The function also takes in the type that the returned value should be, 
    either string or float.
    """
    d1 = datetime.strptime(date_1, "%Y-%M-%d")
    d2 = datetime.strptime(date_2, "%Y-%M-%d")
    delta = d1 - d2
    diff = np.abs(delta.days)
    if out == 'string':
        return (f'There are {diff} days between the two dates')
    return diff

print(time_diff('2020-01-01', '2020-01-02', 'float'))
print(time_diff('2020-01-03', '2020-01-01', 'string'))


### Exercise 4


def reverse(in_list: list) -> list:
    """
    The function takes in a list, and returns the reverse order of the input list.
    """
    new = []
    for i in in_list:
        new = np.append(i, new)
    return new

print(reverse(['a','b']))


### Exercise 5


def fact(n):
    t = 1
    for i in range(1, n+1):
        t *= i
    return t
    
def prob_k_heads(n: int, k: int) -> float:
    """
    The function takes in the number of trials for coin toss as int n, and the number of heads
    as int k. The function will return the probability of getting k heads in n coin flips.
    """
    p = 0.5
    return (fact(n)/(fact(k)*(fact(n-k)))) * (p**k) * ((1 - p)**(n-k))

print(prob_k_heads(2,2))
print(prob_k_heads(1,1))



