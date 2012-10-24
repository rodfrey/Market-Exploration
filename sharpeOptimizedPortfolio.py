import numpy as np
from numpy import recfromcsv
from itertools import combinations
import os

portfolio_size=4
top_n_equities=10

files = [fi for fi in os.listdir('.') if fi.endswith(".csv")]
symbols = [os.path.splitext(fi)[0] for fi in files]
firstfile = recfromcsv(files[0])
datalength = len(firstfile['close'])
closes = np.recarray((datalength,), dtype=[(symbol, 'float') for symbol in symbols])
daily_ret = np.recarray((len(firstfile['close'])-1,), dtype=[(symbol, 'float') for symbol in symbols])
average_returns = np.zeros(len(files))
return_stdev = np.zeros(len(files))
sharpe_ratios = np.zeros(len(files))
cumulative_returns = np.recarray((datalength,), dtype=[(symbol, 'float') for symbol in symbols])
for i, file in enumerate(files):
    data = recfromcsv(file)
    if(len(data) != datalength):
        continue
    closes[symbols[i]] = data['close'][::-1]
    daily_ret[symbols[i]] = (closes[symbols[i]][1:]-closes[symbols[i]][:-1])/closes[symbols[i]][:-1]
    average_returns[i] = np.mean(daily_ret[symbols[i]])
    return_stdev[i] = np.std(daily_ret[symbols[i]])
    sharpe_ratios[i] = (average_returns[i] / return_stdev[i]) * np.sqrt(datalength) 

sorted_sharpe_indices = np.argsort(sharpe_ratios)[::-1][0:top_n_equities]
cov_data = np.zeros((datalength-1, top_n_equities))
for i, symbol_index in enumerate(sorted_sharpe_indices):
    cov_data[:,i] = daily_ret[symbols[symbol_index]]

cormat = np.corrcoef(cov_data.transpose())
portfolios = list(combinations(range(0, top_n_equities), portfolio_size))
total_corr = [sum([cormat[x[0]][x[1]] for x in combinations(p, 2)]) for p in portfolios]
best_portfolio=[symbols[sorted_sharpe_indices[i]] for i in portfolios[total_corr.index(np.nanmin(total_corr))]]
print(best_portfolio)

