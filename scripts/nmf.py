import numpy as np
import pandas as pd

import sys
import preprocessing

from scipy import stats
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF, non_negative_factorization
from sklearn.metrics import mean_absolute_error

from plotting import generatePlots



# runs NMF but fixes W and calls nmf directly instead of instantiating a new NMF() object
# and calling fit_transform()
def runCustom(sig, mix):

    print("mix: ", mix.shape)
    print("sig: ", sig.shape)

    # the roles of W and H are reversed in this case
    # because sklearn nmf only lets us fix H whereas we want to fix W so we must reverse
    # the roles and transpose
    # H is now the signature matrix
    # W is now the mix matrix
    print ("running NMF with %d components" %sig.shape[1])
    W, H, n_iter = non_negative_factorization(
        mix.T,
        n_components=sig.shape[1],
        init='custom',
        solver='mu',
        beta_loss='kullback-leibler',
        max_iter=10000000,
        tol=1e-13,
        random_state=123456,
        update_H=False,
        H=sig.T)
    
    # sum to 1 for each row
    W = W/W.sum(axis=1, keepdims=1)
    
    return W.T , H.T


def runMix(sig, mix):
    # NMF needs all positive values
    sig[sig < 0] = 0
    mix[mix < 0] = 0
    sig[np.isnan(sig)] = 0
    mix[np.isnan(mix)] = 0

    print (np.min(sig))
    print (np.max(sig))

    print (np.min(mix))
    print (np.max(mix))

    print(sig.shape)
    print(mix.shape)

    W, H = runCustom(sig, mix)
    # take the average of the 3 columns in the mix
    return np.mean(W, axis=1)
    

def runMixes(mixes):
    results_raw = []
    results_zscore = []
    results_minmax = []
    for mix in mixes:
        z = zScorePreProcess(signatureMatrix, mix)
        mm = minMaxPreProcess(signatureMatrix, mix)
        results_raw.append(runMix(signatureMatrix, mix))
        results_zscore.append(runMix(z[:, 0:4], z[:, 4:7]))
        results_minmax.append(runMix(mm[:, 0:4], mm[:, 4:7]))
    
    return [results_raw, results_zscore, results_minmax]


def runLinearRegression(sig, mix, expectedWeights):
    sig_train, sig_test, weights_train, weights_test = train_test_split(sig, expectedWeights, train_size = 0.7, test_size = 0.3) 
    model = LinearRegression().fit(sig_train, weights_train)
    model.score(sig_test, weights_test)



def runNmf(sig, mixes, expected, outputPath, numMixes, outputPrefix = ""):
    results = np.array([runMix(sig, mix) for mix in mixes])

    print("output prefix: ", outputPrefix)
    print("reults: ", results.shape)
    print("expected: ", expected.shape)

    np.savetxt('%s/results%s.csv' %(outputPath, outputPrefix), np.array(results), delimiter=',')

    #error = mean_absolute_error(expected, results)
    # print("error: ", error.shape)
    # np.savetxt('%s/error.csv' %outputPath, error, delimiter=',')

    generatePlots(results, expected, "%s/plots" %outputPath, numMixes, outputPrefix)

    meanAbsoluteError = mean_absolute_error(expected, results)
    print("Mean Absolute Error: %.4f" %meanAbsoluteError)

    return meanAbsoluteError