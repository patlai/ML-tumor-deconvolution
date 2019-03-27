import numpy as np
import pandas as pd
import sys

from scipy.optimize import minimize
from plotting import generatePlots
from sklearn.metrics import mean_absolute_error

def runMix(sigMatrix, mixture):
    S = sigMatrix.T

    lossFunction = lambda x: np.sum(np.square((np.dot(x, S) - mixture)))
    constraints =  ({'type': 'eq', 'fun' : lambda x: np.sum(x) - 1.0})
    
    x0 = np.zeros(S.shape[0])
    res = minimize(
        lossFunction,
        x0,
        method='SLSQP',
        constraints=constraints,
        bounds=[(0, np.inf) for i in range(S.shape[0])]
    )

    return res.x

def runCls(sigMatrix, mixtures, expected, outputPath, numMixes):
    results = np.array([runMix(sigMatrix, mix) for mix in mixtures])

    print("reults: ", results.shape)
    print("expected: ", expected.shape)

    np.savetxt('%s/results.csv' %outputPath, np.array(results).T, delimiter=',')

    #error = mean_absolute_error(expected, results)
    # print("error: ", error.shape)
    # np.savetxt('%s/error.csv' %outputPath, error, delimiter=',')

    generatePlots(results.T, expected.T, "%s/plots" %outputPath, numMixes)

    meanAbsoluteError = mean_absolute_error(expected, results)
    print("Mean Absolute Error: %.4f" %meanAbsoluteError)