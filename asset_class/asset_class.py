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
import visualize_wealth.analyze as vwa

def clean_dates(arr_a, arr_b):
    """
    Return the intersection of two indexes
    """
    arr_a = arr_a.sort_index()
    arr_a.dropna(inplace = True)
    arr_b = arr_b.sort_index()
    arr_b.dropna(inplace = True)
    if arr_a.index.equals(arr_b.index) == False:
        return arr_a.index & arr_b.index
    else:
        return arr_a.index

def get_all_classes(asset_series):
    asset_class = get_asset_classes(asset_series)
    sub_classes = get_sub_classes(asset_series, asset_class)
    return sub_classes.append(pandas.Series([asset_class], ['Asset Class']))

def get_asset_class(asset_series):
    ac_dict = {'VTSMX':'US Equity', 'VBMFX':'Fixed Income', 'VGTSX':'Intl Equity',
               'IYR':'Alternative', 'GLD':'Alternative', 'GSG':'Alternative',
               'WPS':'Alternative'}
    data = web.DataReader(ac_dict.keys(), 'yahoo')['Adj Close']
    rsq_d = {}
    for ticker in data.columns:
        ind = asset_series.dropna().index & data[ticker].dropna().index
        rsq_d[ticker] = vwa.r_squared(asset_series[ind], data[ticker][ind])
    rsq = pandas.Series(rsq_d)
    return ac_dict[rsq.argmax()]
    
def get_sub_classes(series, asset_class):
    
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
    asset_class_prices = web.DataReader(pairs.values(), 'yahoo',
                               start = '01/01/2000')['Adj Close']
    ind = clean_dates(series, asset_class_prices)
    ret_series = best_fitting_weights(series[ind], asset_class_prices.loc[ind, :])
    
    #change the column names to asset classes instead of tickers
    return ret_series.rename(index = {v:k for k, v in pairs.iteritems()})

def asset_class_tickers(asset_class):
    """
    All of the ticker and asset class information stored in dictionary form for use
    by other functions
    """
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

    class_dict = {'US Equity': us_eq_dict, 'Foreign Equity': for_eq_dict,
                  'Alternative': alt_dict, 'Fixed Income': fi_dict}

    return class_dict[asset_class]

def all_asset_classes(asset_series):
    """
    Load the different prices that can be determined to find the "Broad Asset Class"
    into a :class:`pandas.Panel` and then pickle the data into ``../data/indexes``.
    """
    ac_dict = {'VTSMX':'US Equity', 'VBMFX':'Fixed Income', 'VGTSX':'Intl Equity',
               'IYR':'Alternative', 'GLD':'Alternative', 'GSG':'Alternative'}
        
    asset_classes = ['US Equity', 'Fixed Income', 'Intl Equity', 'Alternative']
    tickers = ['VTSMX', 'VBMFX', 'VGTSX', 'PFF','IYR','GLD','GSG']
    data = web.DataReader(tickers, 'yahoo')['Adj Close']
    rsq_d = {}
    for ticker in data.columns:
        ind = asset_series.index & data[ticker].index
        rsq_d[ticker] = r_squared(asset_series[ind], data[ticker][ind])
    rsq = pandas.Series(rsq_d)
    return ac_dict[rsq.argmax()]

def best_fitting_weights(series, asset_class_prices):
    """
    Return the best fitting weights given a :class:`pandas.Series` of asset prices
    and a :class:`pandas.DataFrame` of asset class prices.  Can be used with the
    :func:`clean_dates` function to ensure an intersection of the two indexes is
    being passed to the function
    
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

        estimate = numpy.dot(ac_rets, weights)
        sse = ((estimate - series_rets)**2).sum()
        sst = ((series_rets - series_rets.mean())**2).sum()
        rsq = 1 - sse/sst
        p, n = weights.shape[0], ac_rets.shape[0]
        return rsq - (1 - rsq)*(float(p)/(n - p - 1))

    def _obj_fun(weights):
        """
        To maximize the r_squared_adj minimize the negative of r_squared
        """
        return -_r_squared_adj(weights)

    #linear price changes to create a weighted return
    ac_rets = asset_class_prices.pct_change()
    series_rets = series.pct_change()

    #de-mean the sample
    ac_rets = ac_rets.sub(ac_rets.mean() )
    series_rets = series_rets.sub( series_rets.mean() )

    num_assets = ac_rets.shape[1]
    guess = numpy.zeros(num_assets,)

    #ensure the boundaries of the function are (0, 1)
    ge_zero = [(0,1) for i in numpy.arange(num_assets)]

    #optimize to maximize r-squared, using the 'TNC' method (that uses the boundary
    #functionality)
    opt = sopt.minimize(_obj_fun, x0 = guess, method = 'TNC', bounds = ge_zero)
    normed = opt.x*(1./numpy.sum(opt.x))

    return pandas.TimeSeries(normed, index = ac_rets.columns)

def asset_class_by_interval(series, interval, asset_class):
    """
    Return asset class weightings that explain the asset returns over interval
    periods of "interval."

    :ARGS:

        asset_prices: :class:`pandas.Series` of the asset for which attribution
        will be done

        asset_class_prices: :class:`pandas.DataFrame` of asset class prices

        interval :class:string of the frequency interval 'quarterly' or 'annual'

    :RETURNS:

        :class:`pandas.DataFrame` of proportions of each asset class that most
        explain the returns of the individual security
    """
    #import pdb
    #pdb.set_trace()
    ac_dict = asset_class_tickers(asset_class)
    benchmark = web.DataReader(ac_dict.values(), 'yahoo',
                               start = '01/01/2000')['Adj Close']
    
    ind = clean_dates(series, benchmark)
    dt_dict = {'quarterly': lambda x: x.quarter, 'annually': lambda x: x.year}
    dts = numpy.append(
        True, dt_dict[interval](ind[1:]) != dt_dict[interval](ind[:-1]) )
    weight_d  = {}
    for beg, fin in zip(ind[dts][:-1], ind[dts][1:]):
        weight_d[beg] = best_fitting_weights(
            series[beg:fin], benchmark.loc[beg:fin, :]).rename(
            index = {v:k for k, v in ac_dict.iteritems()})

    return pandas.DataFrame(weight_d).transpose()

    
