### Exercise 0

def github() -> str:
    """
    Some docstrings.
    """

    return "https://github.com/<user>/<repo>/blob/main/<filename.py>"

### Exercise 1

import requests
from bs4 import BeautifulSoup
import re

def scrape_code(url: str) -> str:
    """
    Some docstrings.
    """
    req_obj = requests.get(url)
    soup = BeautifulSoup(req_obj.text)
    code = soup.find_all('code', class_=['sourceCode'])
    code_list = [[item.text for item in c] for c in code]

    s = ''
    for i in code_list:
        for j in i:
            if not re.match('^%+', j):
                if s == '':
                    s = j
                elif j != "\n":
                    s = s + "\n" + j
            
    return s
    

print(scrape_code('https://lukashager.netlify.app/econ-481/02_numerical_computing_in_python'))




