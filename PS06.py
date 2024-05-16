### Exercise 0 

def github() -> str:
    """
    Some docstrings.
    """

    return "https://github.com/murphyvica/Econ_481/blob/main/PS06.py"


### Exercise 1

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import inspect
import pandas as pd
from sqlalchemy.orm import Session

path = 'econ-481-jupyterhub/auctions.db'

class DataBase:
    def __init__(self, loc: str, db_type: str = "sqlite") -> None:
        """Initialize the class and connect to the database"""
        self.loc = loc
        self.db_type = db_type
        self.engine = create_engine(f'{self.db_type}:///{self.loc}')
    def query(self, q: str) -> pd.DataFrame:
        """Run a query against the database and return a DataFrame"""
        with Session(self.engine) as session:
            df = pd.read_sql(q, session.bind)
        return(df)

auctions = DataBase(path)

inspector = inspect(auctions.engine)
inspector.get_table_names()


def std() -> str:
    """
    Some docstrings.
    """
    q = """
    select bids.itemId, SQRT(SUM(POWER(bidAmount - bidavg, 2)) / (n-1)) as std
    FROM bids 
    JOIN (
    SELECT itemId, AVG(bidAmount) AS bidavg, COUNT(*) as n
    FROM bids
    GROUP BY itemId
    ) b
    ON bids.itemId = b.itemId
    WHERE n >= 2
    GROUP BY bids.itemId
    """

    return q



q = std()
print(auctions.query(q).head)


### Exercise 2

def bidder_spend_frac() -> str:
    """
    Some docstrings.
    """
    q = """
    WITH total_S AS (SELECT bidderName, SUM(bidAmount) as total_spend
        FROM bids as b
        WHERE bidAmount = (SELECT MAX(bidAmount) FROM bids WHERE itemId = b.itemID GROUP BY itemID)
        GROUP BY bidderName
    ),
    total_B AS (
    SELECT bidderName, SUM(bidAmount) as total_bid
    FROM (
        SELECT bidderName, MAX(bidAmount) AS bidAmount
        FROM bids
        GROUP BY bidderName, itemId
    )
    GROUP BY bidderName
    )
    
    SELECT s.total_spend, b.bidderName, b.total_bid, s.total_spend/b.total_bid as spend_frac
    FROM total_S s
    RIGHT JOIN total_B b ON s.bidderName = b.bidderName
    """

    return q


q = bidder_spend_frac()
print(auctions.query(q).head)


### Exercise 3


def min_increment_freq() -> str:
    """
    Some docstrings.
    """
    q = """
    SELECT COUNT(bidIncrement) * 1.0 /(SELECT COUNT(*) FROM items) as freq
    FROM items
    WHERE bidIncrement = (SELECT MIN(bidIncrement) FROM items)
    AND isBuyNowUsed != 1
    """

    return q


q = min_increment_freq()
print(auctions.query(q).head)


### Exercise 4


def win_perc_by_timestamp() -> str:
    """
    Some docstrings.
    """
    q = """
    WITH binned AS (
        SELECT (julianday(endtime) - julianday(starttime)) as length,
            NTILE(10) OVER (ORDER BY (julianday(endtime) - julianday(starttime))) AS bin,
            bidAmount,
            CASE
                WHEN bidAmount = (SELECT MAX(bidAmount) FROM bids WHERE itemId = items.itemID GROUP BY itemID)
                THEN TRUE
                ELSE FALSE
            END AS iswin
        FROM items
        JOIN bids ON bids.itemId = items.itemId
    )
    SELECT bin as timestamp_bin, SUM(iswin)*1.0/COUNT(*) as win_perc
    FROM binned
    GROUP BY bin
    
    """

    return q


q = win_perc_by_timestamp()
print(auctions.query(q).head)






