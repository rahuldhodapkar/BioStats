#!/usr/bin/env python
# -*- coding: utf-8 -*-

# regress.py - least squares regression and confidence plotting
#
# Adapted from a script by Tom Holderness, we add additional generality and
# CLI helpers to make this easier to interact with
#
# @author:  Rahul Dhodapkar
# @version: 2016-07-28
# @fork:    [https://tomholderness.wordpress.com/2013/01/10/confidence_intervals/]
#
# References:
# - Statistics in Geography by David Ebdon (ISBN: 978-0631136880)
# - Reliability Engineering Resource Website:
# - http://www.weibull.com/DOEWeb/confidence_intervals_in_simple_linear_regression.htm
# - University of Glascow, Department of Statistics:
# - http://www.stats.gla.ac.uk/steps/glossary/confidence_intervals.html#conflim
#
# Assumptions:
#   This package assumes that data is sampled from normally distributed source
#   populations, and uses Student's t-distribution for confidence interval
#   calculation
#

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy import stats
from optparse import OptionParser
import csv
import json

##################################################################
## CONFIGURATION
##################################################################

def poly_fit(x, y, degree):
    # fit a curve to the data using a least squares 1st order polynomial fit
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    return p

# define fit function from x and y samples
#   (x,y,metadata) -> ([x]) -> ([y])
# as required for generating predictions against the fit.
MODELS = {
    "poly": poly_fit
}

SUPPORTED_MODELS = MODELS.keys()

# assume header line and "x,y" format
def load_csv(input_file):
    data_iter = csv.reader(input_file,
                           delimiter=",",
                           quotechar='"',
                           )

    # ***NOTE*** does not check if header row exists before skipping
    next(data_iter)

    x, y = [], []
    for xval, yval in data_iter:
        x.append(float(xval))
        y.append(float(yval))

    return np.asarray(x), np.asarray(y)

LOAD_DATA_BY_TYPE = {
    "csv": load_csv
}

##################################################################
## INPUTS
##################################################################

parser = OptionParser()

parser.add_option("-t", "--inputType", dest="input_type",
                  help="input of format FORMAT", metavar="FORMAT")
parser.add_option("-i", "--inputFile", dest="input_file",
                  help="input file path FILE", metavar="FILE")
parser.add_option("-o", "--outputDir", dest="out_base",
                  help="output base DIR", metavar="DIR")
parser.add_option("-d", "--modelDist", dest="fit_dist",
                  help="run regression against model DIST", metavar="DIST")
parser.add_option("-m", "--modelMetadata", dest="model_metadata",
                  help="model metadata as JSON META", metavar="META")

(options, args) = parser.parse_args()

print(options)
print(args)

# initialize defaults - linear polynomial fit
input_type     = "csv"
input_file     = None
out_base       = None
fit_dist       = "poly"
model_metadata = {"degree": 1}

# input_type
if options.input_type is not None:
    if options.input_type == "csv":
        input_type = "csv"
    else:
        print("ERROR: unsupported inputType '{0}' provided".format(options.input_type))

# input_file
if options.input_file is None:
    print("ERROR: required input file not provided")
    sys.exit(1)
else:
    try:
        input_file = open(options.input_file, 'rU')
    except Exception as e:
        print("ERROR: problems opening input file '{0}'".format(options.input_file))
        sys.exit(1)

# out_base
if options.out_base is None:
    print("ERROR: required out_base not provided")
    sys.exit(1)
else:
    try:
        if not os.path.exists(options.out_base):
            os.makedirs(options.out_base)
        out_base = options.out_base
    except Exception as e:
        print("ERROR: {0} - problems checking base directory '{1}'".format(e, options.out_base))
        sys.exit(1)

# fit_dist
if options.fit_dist is not None:
    if options.fit_dist not in SUPPORTED_MODELS:
        print("ERROR: model '{0}' not currently supported".format(options.fit_dist))
        sys.exit(1)
    else:
        fit_dist = options.fit_dist

# model_metadata
if options.model_metadata is not None:
    try:
        model_metadata = json.loads(options.model_metadata)
    except ValueError as e:
        print("ERROR: unable to parse META JSON string - {0}".format(e))
        sys.exit(1)

# exit cleanly after test
# sys.exit(0)

##################################################################
## EXECUTION
##################################################################

# example data
#x = np.array([4.0,2.5,3.2,5.8,7.4,4.4,8.3,8.5])
#y = np.array([2.1,4.0,1.5,6.3,5.0,5.8,8.1,7.1])
(x, y) = LOAD_DATA_BY_TYPE[input_type](input_file)

print(x)
print(y)

# fit a curve to the data using a least squares 1st order polynomial fit
#z = np.polyfit(x,y,1)
#p = np.poly1d(z)
#fit = p(x)
p = MODELS[fit_dist](x, y, model_metadata)

print("completed fit of model as:")
print(p)

# predict y values of original data using the fit
p_y = [p(x_i) for x_i in x]

# get the coordinates for the fit curve
c_y = [np.min(p_y), np.max(p_y)]
c_x = [np.min(x),   np.max(x)  ]

# calculate the y-error (residuals)
y_err = y - p_y

# create series of new test x-values to predict for
p_x = np.arange(np.min(x), np.max(x)+1, 1)

# now calculate confidence intervals for new test x-series
mean_x = np.mean(x)     # mean of x
n = len(x)              # number of samples in origional fit

# use the t-distribution ppf (inverse cdf) to find t value at low percentile
t = stats.t.ppf(1 - 0.025, df=n)    # appropriate t value (two tailed 95%)
s_err = np.sum(np.power(y_err,2))   # sum of the squares of the residuals

confs = t * np.sqrt(
                (s_err/(n-2))*
                (1.0/n +
                    (np.power((p_x-mean_x),2)/
                        ((np.sum(np.power(x,2))) - n*(np.power(mean_x,2))))
                )
            )

# now predict y based on test x-values
p_y = [p(p_x_i) for p_x_i in p_x]

# get lower and upper confidence limits based on predicted y and confidence intervals
lower = p_y - abs(confs)
upper = p_y + abs(confs)

# set-up the plot
plt.axes().set_aspect('equal')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Linear regression and confidence limits')

# plot sample data
plt.plot(x,y,'bo',label='Sample observations')

# plot line of best fit
plt.plot(c_x,c_y,'r-',label='Regression line')

# plot confidence limits
plt.plot(p_x,lower,'b--',label='Lower confidence limit (95%)')
plt.plot(p_x,upper,'b--',label='Upper confidence limit (95%)')

# set coordinate limits
plt.xlim(0,11)
plt.ylim(0,11)

# configure legend
plt.legend(loc=0)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=10)

# show the plot
plt.show()

