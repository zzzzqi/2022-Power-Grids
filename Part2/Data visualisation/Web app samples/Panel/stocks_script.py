import panel as pn
import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.pandas  # noqa

pn.extension()

stocks = pd.read_csv('./stocks.csv', index_col='Date', parse_dates=True)
mean_daily_ret = stocks.pct_change(1).mean()
stocks.pct_change(1).corr()
stock_normed = stocks/stocks.iloc[0]
pn.Pane(stock_normed.hvplot()).servable()

stock_daily_ret = stocks.pct_change(1)
log_ret = np.log(stocks/stocks.shift(1))
pn.Pane(log_ret.hvplot.hist(bins=100, subplots=True, width=400, group_label='Ticker', grid=True).cols(2)).servable()

pn.Pane(log_ret.describe().transpose()).servable()

num_ports = 15000

all_weights = np.zeros((num_ports,len(stocks.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for ind in range(num_ports):

    # Create Random Weights
    weights = np.array(np.random.random(4))

    # Rebalance Weights
    weights = weights / np.sum(weights)
    
    # Save Weights
    all_weights[ind,:] = weights

    # Expected Return
    ret_arr[ind] = np.sum((log_ret.mean() * weights) *252)

    # Expected Variance
    vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))

    # Sharpe Ratio
    sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]

max_sr_ret = ret_arr[1419]
max_sr_vol = vol_arr[1419]

scatter = hv.Scatter((vol_arr, ret_arr, sharpe_arr), 'Volatility', ['Return', 'Sharpe Ratio'])
max_sharpe = hv.Scatter([(max_sr_vol,max_sr_ret)])

pn.Pane(scatter.opts(color='Sharpe Ratio', cmap='plasma', width=600, height=400, colorbar=True, padding=0.1) * max_sharpe.opts(color='red', line_color='black', size=10)).servable()