import numpy as np
import pandas as pd
import sys
from scipy import stats

from IO import saveMatrix
from preprocessing import getOverlappingGenes

from plotting import multiHistogram
from nmf import runNmf
from cls import runCls

def generate(sig, cov, covTransformed, tcgaMean, tcgaStd, numMixtures, outputPath, scaleFactor, mu = None, weights = None):
    """
    generates cell mixtures data using a multivariate-normal distribution and randomly generated weights for the cell types
    .
    This will generate the following:
    - raw data
    - log2(x+1) transformed data (note: in the case of data that is already log2(x+1) transformed, this will be pretty useless)

    :sig: signature matrix

    :cov: gene-gene covariance matrix (should be square)

    :tcgaMean: mean column vector of patient data by gene
    :tcgaStd: standard deviation column vector of patient data by gene

    :numMixtures: how many mixtures to generate

    :outputPath: where to save the generated data

    :scaleFactor: how much to scale the covariance matrix by
    """

    gen = []
    for i in range(0, numMixtures):
        print("  generating mixture %d" %(i+1))
        # get mu from the epic signature by generating random weights and multiplying them with the sig
        # generate a random vector equal to the number of cell types in the epic signature (cols)
        # make sure the weights sum to one
        randomWeights = []
        if (weights is None):
            randomWeights = np.random.rand(sig.shape[1])
            randomWeights /= randomWeights.sum(axis = 0)
        else:
            randomWeights = weights

        weightedMu = mu if mu is not None else np.sum(randomWeights * sig, axis = 1)

        print("using weights:")
        print(weightedMu)

        # use the weighted mu from epic and the cov matrix from tcga to generate a multivariate distribution
        # 1 sample of the generated data should correspond to the (mixed) gene expression of 1 patient
        generatedData = np.random.multivariate_normal(weightedMu, cov, 1)
        generatedData[np.isnan(generatedData)] = 0
        
        # undo the z-score
        # Z = (x-m)/s 
        # sZ = x-m 
        # x = sZ + m
        # generatedDataZ = tcgaStd *  np.random.multivariate_normal(weightedMu, covTransformed, 1) + tcgaMean
        
        generatedDataTransformed = np.log2(generatedData + 1)
        generatedDataTransformed[np.isnan(generatedDataTransformed)] = 0
        # generatedDataZTransformed = np.log2(generatedDataZ + 1)

        print("generated data with shape %s" %str(generatedDataTransformed.shape))

        print("%d NaN values found out of %d values in %s" %(
            len([x for x in np.isnan(generatedDataTransformed) if x is True]),
            (generatedDataTransformed.shape[0] * generatedDataTransformed.shape[1]),
            "log2 transformed raw generated data"
        ))

        saveMatrix('%s/generated_raw_mixture_%d_%s.csv' %(outputPath, i+1, formatSf(scaleFactor)), generatedData)
        saveMatrix('%s/generated_log2_mixture_%d_%s.csv' %(outputPath, i+1, formatSf(scaleFactor)), generatedDataTransformed)
        
        # saveMatrix('%s/generated_zTransform_mixture_%d.csv' %(outputPath, i+1), generatedDataZ)
        # saveMatrix('%s/generated_zTransform_log2_mixture_%d.csv' %(outputPath, i+1), generatedDataZTransformed)
        
        saveMatrix('%s/mu_mixture_%d_%s.csv' %(outputPath, i+1, formatSf(scaleFactor)), weightedMu)
        saveMatrix('%s/weights_mixture_%d_%s.csv' %(outputPath, i+1, formatSf(scaleFactor)), randomWeights)
        
        gen.append({
            "raw": generatedData,
            # "zt": generatedDataZ,
            "raw_log2": generatedDataTransformed,
            # "zt_log2": generatedDataZTransformed,
            "weights": randomWeights,
            "mu": weightedMu,
        })
    
    return gen

def formatSf(sf):
    return str(sf).replace('.', '')


def getMapping(patientDataPath, signaturePath, mappingFilePath, outputPath):
    """
    Gets the overlapping genes between the patient data and cell signature and
    returns both filtered matrices and the covariance of the patient data
    """

    preProcessed = getOverlappingGenes(patientDataPath, signaturePath, mappingFilePath, outputPath)

    patientDataMatrix = preProcessed["patientData"]
    signatureMatrix = preProcessed["signature"]

    patientDataCov = np.cov(patientDataMatrix)

    return patientDataMatrix, signatureMatrix, patientDataCov


def generateWithScaling(patientDataPath, signaturePath, mappingFilePath, outputPath, scaleFactors):
    """
    Calculates the gene overlap, covariance matrix and then generates the data
    """

    patientDataMatrix, signatureMatrix, patientDataCov = getMapping(patientDataPath, signaturePath, mappingFilePath, outputPath)

    print("truncated signature matrix shape: " + str(signatureMatrix.shape))
    saveMatrix("%s/truncated-signature.csv" %outputPath, signatureMatrix)

    errors = []

    randomWeights = np.ones(signatureMatrix.shape[1]) #np.random.rand(signatureMatrix.shape[1])
    randomWeights /= randomWeights.sum(axis = 0)
    print(randomWeights)
    weightedMu = np.sum(randomWeights * signatureMatrix, axis = 1)

    for sf in scaleFactors:
        gen = generate(signatureMatrix, sf * patientDataCov, None, None, None, 4, outputPath, sf, weightedMu, randomWeights)

        nmfError = runNmf(
            signatureMatrix,
            np.array([g["raw_log2"].T for g in gen]),
            np.array([g["weights"] for g in gen]),
            outputPath,
            4,
            formatSf(sf),
        )

        clsErrors = runCls(
            signatureMatrix,
            np.array([g["raw_log2"].T for g in gen]),
            np.array([g["weights"] for g in gen]),
            outputPath,
            4,
            formatSf(sf),
        )

        #plotDataGeneration(gen, sf, outputPath)

        errors.append([sf, nmfError, clsErrors])

    saveMatrix("%s/scaling-results.csv" %outputPath, np.array(errors))


def generateWithMapping(patientDataPath, signaturePath, mappingFilePath, outputPath, scaleFactor = 1):
    """
    Maps the genes between patient data and a signature
    then generates data using a multi-variate normal distribtuion
    """

    print("=== Generating with patient data %s, signature %s, scale factor %s" %(patientDataPath, signaturePath, formatSf(scaleFactor)))
    
    preProcessed = getOverlappingGenes(patientDataPath, signaturePath, mappingFilePath)
    
    patientDataMatrix = preProcessed["patientData"]
    signatureMatrix = preProcessed["signature"]

    # z normalize the patient data one gene at a time across different samples
    # x2 = (x- meean(all x in the same row)) / std (all x in the same row)
    # also keep a column of the means and standard deviations to recover the original data later
    patientDataMean = np.mean(patientDataMatrix, axis = 1)
    patientDataStd = np.std(patientDataMatrix, axis = 1)
    patientDataMatrixTransformed = stats.zscore(patientDataMatrix, axis = 0)

    print("taking covariance...")
    # take the gene-gene covariance of tcga and multiply it by the scaling factor
    patientDataCov = scaleFactor * np.cov(patientDataMatrix)
    patientDataCovTransformed = scaleFactor * np.cov(patientDataMatrixTransformed)

    print("generating data...")
    gen = generate(signatureMatrix, patientDataCov, patientDataCov, patientDataMean, patientDataStd, 4, outputPath)

    # generate histograms for the generated data
    numMixtures = len(gen)
    rawLabels = [("mix %d" %x) for x in range (1, numMixtures + 1)]
    log2Labels = [("mix %d: log2(x+1) transformed" %x) for x in range (1, numMixtures + 1)]

    multiHistogram([g["raw"].flatten() for g in gen],
        10,
        "Generated raw mixtures, scale factor = %.2f" %scaleFactor,
        "%s/plots/data-generation-raw-sf-%s.png" %(outputPath, formatSf(scaleFactor)),
        labels=None
    )

    multiHistogram(
        [g["raw_log2"].flatten() for g in gen],
        10,
        "Generated log2(x+1) transformed mixtures, scale factor = %.2f" %scaleFactor,
        "%s/plots/data-generation-log2-sf-%s.png" %(outputPath, formatSf(scaleFactor)),
        labels=None
    )

    return gen, signatureMatrix


def plotDataGeneration(gen, scaleFactor, outputPath):
    # generate histograms for the generated data
    numMixtures = len(gen)
    rawLabels = [("mix %d" %x) for x in range (1, numMixtures + 1)]
    log2Labels = [("mix %d: log2(x+1) transformed" %x) for x in range (1, numMixtures + 1)]

    multiHistogram([g["raw"].flatten() for g in gen],
        10,
        "Generated raw mixtures, scale factor = %.2f" %scaleFactor,
        "%s/plots/data-generation-raw-sf-%s.png" %(outputPath, formatSf(scaleFactor)),
        labels=None
    )

    multiHistogram(
        [g["raw_log2"].flatten() for g in gen],
        10,
        "Generated log2(x+1) transformed mixtures, scale factor = %.2f" %scaleFactor,
        "%s/plots/data-generation-log2-sf-%s.png" %(outputPath, formatSf(scaleFactor)),
        labels=None
    )


def main(args):
    """
    args:
    [0] = patient data path
    [1] = signature path
    [2] = mapping file path
    [3] = output path
    """
    print(args)

    patientDataPath = args[0]
    signaturePath = args[1]
    mappingFilePath = args[2]
    outputPath = args[3]

    # try different scaling factors on the covariance matrix to examine the effect of adding noise
    # on the data generation
    scaleFactors = [round(sf, 2) for sf in np.arange(0, 1.1, 0.1)]

    generateWithScaling(patientDataPath, signaturePath, mappingFilePath, outputPath, scaleFactors)

if __name__ == "__main__":
    main(sys.argv[1:])

