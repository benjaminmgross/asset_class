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
    p, n = weights.shape[0], asset_returns.shape[0]
    return rsq - (1 - rsq)*(float(p)/(n - p - 1))
    
def r_squared(portfolio_prices, asset_prices, weights):
    """
    The unadjusted R-Squared
    """
    asset_returns = asset_prices.pct_change()
    portfolio_returns = portfolio_prices.pct_change()
    estimate = numpy.dot(asset_returns, weights)
    sse = ((estimate - portfolio_returns)**2).sum()
    sst = ((portfolio_returns - portfolio_returns.mean())**2).sum()
    return 1 - sse/sst

def load_asset_classes():
    """
    Load the different prices that can be determined to find the "Broad Asset Class"
    into a :class:`pandas.Panel` and then pickle the data into ``../data/indexes``.
    """
    asset_classes = ['US Equity', 'Fixed Income', 'Intl Equity', 'Alternative']
    tickers = ['VTSMX', 'VBMFX', 'VGTSX', 'PFF','IYR','GLD','GSG']
    return None

def best_fitting_weights(asset_prices, asset_class_prices):
	"""
	:ARGS:
    
	    asset_prices: m x 1 :class:`pandas.TimeSeries` of asset_prices

        ac_prices: m x n :class:`pandas.DataFrame` asset_class prices

	:RETURNS:
	
	    :class:`pandas.TimeSeries` of nonnegative weights for each asset
	    such that the r_squared from the regression of Y ~ Xw + e is maximized

	"""
	def _r_squared(weights):
		"""
		The Adjusted R-Squared that incorporates the number of independent variates
		"""
		estimate = asset_returns.dot(weights)
		sse = ((estimate - portfolio_returns)**2).sum()
		sst = ((portfolio_returns - portfolio_returns.mean())**2).sum()
		return 1 - sse/sst

	def _obj_fun(weights):
		"""
		To maximize the r_squared, minimize the negative of r_squared
		"""	 
		return - _r_squared(weights)


	

	num_assets = asset_returns.shape[1]
	guess = numpy.zeros(num_assets,)
	#sum_to_one = lambda x: numpy.dot(numpy.tile(x, num_assets,),
    #numpy.ones(num_assets,)) - 1

	#ensure the boundaries of the function are (0, 1)
	ge_zero = [(0,1) for i in numpy.arange(num_assets)]

	#optimize to maximize r-squared, using the 'TNC' method (that uses the boundary
    #functionality)
	opt = sopt.minimize(_obj_fun, x0 = guess, method = 'TNC', bounds = ge_zero)

	normed = opt.x*(1./numpy.sum(opt.x))

	return pandas.TimeSeries(normed, index = asset_returns.columns)
    
