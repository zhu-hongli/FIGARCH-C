import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma


def get_sp500_data():
    path = 'data/sp500.csv'
    try:
        sp500 = pd.read_csv(path)
    except:
        print("There is no file, it is being downloaded from the Internet. . . .")
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime(2021, 1, 1)
        sp500 = web.DataReader('^GSPC', 'yahoo', start=start, end=end)
        sp500.to_csv(path)
    return sp500

def log_returns(data):
    log = np.log(data/data.shift(1))
    return log

def get_sp500_log_returns():
    sp500 = get_sp500_data()["Adj Close"]
    data = log_returns(sp500)
    data = data.dropna()
    return data

