import numpy as np
import sys

from nmf import runNmf
from cls import runCls

def main(args):
    mixFilePrefix = args[0]
    signatureFile = args[1]
    expectedFile = args[2]
    outputPath = args[3]
    numMixes = int(args[4])
    method = args[5]

    mixes = []
    for i in range(0, numMixes):
        mix = np.array([np.genfromtxt("%s_%d.csv" %(mixFilePrefix, i+1), delimiter=',')]).T
        mix[np.isnan(mix)] = 0
        mix[mix < 0] = 0
        mixes.append(mix)
    
    expected = np.genfromtxt(expectedFile, delimiter=',')
    signatureMatrix = np.genfromtxt(signatureFile, delimiter=',')

    print("num mixes: ", numMixes)
    print("mix: ", mix.shape)
    print("sig: ", signatureMatrix.shape)
    print("expected: ", expected.shape)

    signatureMatrix[np.isnan(signatureMatrix)] = 0

    print (method)

    if (method == 'NMF'):
        runNmf(signatureMatrix, mixes, expected.T, outputPath, numMixes)
    elif (method == 'CLS'):
        runCls(signatureMatrix, mixes, expected.T, outputPath, numMixes)
    

if __name__ == "__main__":
    main(sys.argv[1:])