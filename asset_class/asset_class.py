#!/usr/bin/env python
# encoding: utf-8
"""
.. module:: asset_class.py
   :synopsis: Do fun things with Asset Classes

.. moduleauthor:: Benjamin M. Gross <benjaminMgross@gmail.com>
"""

import pandas
import numpy
import scipy.optimize as sopt
import pandas.io.data as web


def r_squared_adj(portfolio_prices, asset_prices, weights):
    """
    The Adjusted R-Squared that incorporates the number of independent variates
    using the `Formula Found of Wikipedia
    <http://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2>_`
    """
    asset_returns = asset_prices.pct_change()
    portfolio_returns = portfolio_prices.pct_change()
    estimate = numpy.dot(asset_returns, weights)
    sse = ((estimate - portfolio_returns)**2).sum()
    sst = ((portfolio_returns - portfolio_returns.mean())**2).sum()
    rsq = 1 - sse/sst
    p, n = weights.shape[0], asset_returns.dropna().shape[0]
    return rsq - (1 - rsq)*(float(p)/(n - p - 1))
    
def r_squared(portfolio_prices, asset_prices):
    """
    Univariate R-Squared
    """
    asset_returns = asset_prices.pct_change()
    portfolio_returns = portfolio_prices.pct_change()
    sse = ((asset_returns - portfolio_returns)**2).sum()
    sst = ((portfolio_returns - portfolio_returns.mean())**2).sum()
    return 1 - sse/sst


def get_all_classes(asset_series):
    asset_class = get_asset_classes(asset_series)
    sub_classes = get_sub_classes(asset_series, asset_class)
    return sub_classes.append(pandas.Series([asset_class], ['Asset Class']))

def get_asset_classes(asset_series):
    ac_dict = {'VTSMX':'US Equity', 'VBMFX':'Fixed Income', 'VGTSX':'Intl Equity',
               'IYR':'Alternative', 'GLD':'Alternative', 'GSG':'Alternative',
               'WPS':'Alternative'}
    data = web.DataReader(ac_dict.keys(), 'yahoo')['Adj Close']
    rsq_d = {}
    for ticker in data.columns:
        ind = asset_series.index & data[ticker].index
        rsq_d[ticker] = r_squared(asset_series[ind], data[ticker][ind])
    rsq = pandas.Series(rsq_d)
    return ac_dict[rsq.argmax()]
    
def get_sub_classes(asset_series, asset_class):
    
    fi_dict = {'US Inflation Protected':'TIP', 'Foreign Treasuries':'BWX',
               'Foreign High Yield':'PCY','US Investment Grade':'LQD',
               'US High Yield':'HYG', 'US Treasuries ST':'SHY',
               'US Treasuries LT':'TLT', 'US Treasuries MT':'IEF'}

    us_eq_dict = {'US Large Cap Growth':'JKE', 'US Large Cap Value':'JKF',
                  'US Mid Cap Growth':'JKH','US Mid Cap Value':'JKI',
                  'US Small Cap Growth':'JKK', 'US Small Cap Value':'JKL'}

    for_eq_dict = {'Foreign Developed Small Cap':'SCZ',
                   'Foreign Developed Large Growth':'EFG',
                   'Foreign Developed Large Value':'EFV',
                   'Foreign Emerging Market':'EEM'}

    alt_dict = {'Commodities':'GSG', 'US Real Estate':'IYR',
                'Foreign Real Estate':'WPS', 'US Preferred Stock':'PFF'}


    sub_dict = {'Fixed Income': fi_dict, 'US Equity': us_eq_dict,
                'Intl Equity': for_eq_dict, 'Alternative': alt_dict}

    pairs = sub_dict[asset_class]
    ac_prices = web.DataReader(pairs.values(), 'yahoo',
                               start = '01/01/2000')['Adj Close']
    ret_series = best_fitting_weights(asset_series, ac_prices)
    
    #change the column names to asset classes instead of tickers
    ret_series = ret_series.rename(index = {v:k for k, v in pairs.iteritems()})
    return ret_series
    

def load_asset_classes(asset_series):
    """
    Load the different prices that can be determined to find the "Broad Asset Class"
    into a :class:`pandas.Panel` and then pickle the data into ``../data/indexes``.
    """
    ac_dict = {'VTSMX':'US Equity', 'VBMFX':'Fixed Income', 'VGTSX':'Intl Equity',
               'IYR':'Alternative', 'GLD':'Alternative', 'GSG':'Alternative'}

    fi_dict = {'US Inflation Protected':'TIP', 'Foreign Treasuries':'BWX',
               'Foreign High Yield':'PCY','US Investment Grade':'LQD',
               'US High Yield':'HYG', 'US Treasuries ST':'SHY',
               'US Treasuries LT':'TLT', 'US Treasuries MT':'IEF'}

    us_eq_dict = {'U.S. Large Cap Growth':'JKE', 'U.S. Large Cap Value':'JKF',
                  'U.S. Mid Cap Growth':'JKH','U.S. Mid Cap Value':'JKI',
                  'U.S. Small Cap Growth':'JKK', 'U.S. Small Cap Value':'JKL'}

    for_eq_dict = {'Foreign Developed Small Cap':'SCZ',
                   'Foreign Developed Large Growth':'EFG',
                   'Foreign Developed Large Value':'EFV',
                   'Foreign Emerging Market':'EEM'}


    alt_dict = {'Commodities':'GSG', 'U.S. Real Estate':'IYR',
                'Foreign Real Estate':'WPS', 'U.S. Preferred Stock':'PFF'}
        
    asset_classes = ['US Equity', 'Fixed Income', 'Intl Equity', 'Alternative']
    tickers = ['VTSMX', 'VBMFX', 'VGTSX', 'PFF','IYR','GLD','GSG']
    data = web.DataReader(tickers, 'yahoo')['Adj Close']
    rsq_d = {}
    for ticker in data.columns:
        ind = asset_series.index & data[ticker].index
        rsq_d[ticker] = r_squared(asset_series[ind], data[ticker][ind])
    rsq = pandas.Series(rsq_d)
    return ac_dict[rsq.argmax()]

def best_fitting_weights(asset_prices, ac_prices):
    """
    :ARGS:

        asset_prices: m x 1 :class:`pandas.TimeSeries` of asset_prices

        ac_prices: m x n :class:`pandas.DataFrame` asset class ("ac") prices

    :RETURNS:

        :class:`pandas.TimeSeries` of nonnegative weights for each asset
        such that the r_squared from the regression of Y ~ Xw + e is maximized

    """

    def _r_squared_adj(weights):
        """
        The Adjusted R-Squared that incorporates the number of independent variates
        using the `Formula Found of Wikipedia
        <http://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2>_`
        """

        estimate = numpy.dot(asset_returns, weights)
        sse = ((estimate - portfolio_returns)**2).sum()
        sst = ((portfolio_returns - portfolio_returns.mean())**2).sum()
        rsq = 1 - sse/sst
        p, n = weights.shape[0], asset_returns.dropna().shape[0]
        return rsq - (1 - rsq)*(float(p)/(n - p - 1))

    def _obj_fun(weights):
        """
        To maximize the r_squared_adj minimize the negative of r_squared
        """
        return -_r_squared_adj(weights)

    asset_prices.dropna(inplace = True)
    ac_prices.dropna(inplace = True)
    if ac_prices.index.equals(asset_prices.index) == False:
        ind = ac_prices.index & asset_prices.index
        asset_returns = ac_prices.loc[ind, :].pct_change()
        portfolio_returns = asset_prices[ind].pct_change()
    else:
        asset_returns = ac_prices.pct_change()
        portfolio_returns = asset_prices.pct_change()

    num_assets = asset_returns.shape[1]
    guess = numpy.zeros(num_assets,)

    #ensure the boundaries of the function are (0, 1)
    ge_zero = [(0,1) for i in numpy.arange(num_assets)]

    #optimize to maximize r-squared, using the 'TNC' method (that uses the boundary
    #functionality)
    opt = sopt.minimize(_obj_fun, x0 = guess, method = 'TNC', bounds = ge_zero)
    normed = opt.x*(1./numpy.sum(opt.x))

    return pandas.TimeSeries(normed, index = asset_returns.columns)
