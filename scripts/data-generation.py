import numpy as np
import pandas as pd
import sys
from scipy import stats

from IO import saveMatrix
from preprocessing import getOverlappingGenes

def generate(sig, cov, covTransformed, tcgaMean, tcgaStd, numMixtures, outputPath):
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
    """

    gen = []
    for i in range(0, numMixtures):
        print("  generating mixture %d" %(i+1))
        # get mu from the epic signature by generating random weights and multiplying them with the sig
        # generate a random vector equal to the number of cell types in the epic signature (cols)
        # make sure the weights sum to one
        randomWeights = np.random.rand(sig.shape[1])
        randomWeights /= randomWeights.sum(axis = 0)
        weightedMu = np.sum(randomWeights * sig, axis = 1)

        # use the weighted mu from epic and the cov matrix from tcga to generate a multivariate distribution
        # 1 sample of the generated data should correspond to the (mixed) gene expression of 1 patient
        generatedData = np.random.multivariate_normal(weightedMu, cov, 1)
        
        # undo the z-score
        # Z = (x-m)/s 
        # sZ = x-m 
        # x = sZ + m
        generatedDataZ = tcgaStd *  np.random.multivariate_normal(weightedMu, covTransformed, 1) + tcgaMean
        
        generatedDataTransformed = np.log2(generatedData + 1)
        generatedDataZTransformed = np.log2(generatedDataZ + 1)
        
        saveMatrix('%s/generated_raw_mixture_%d.csv' %(outputPath, i+1), generatedData)
        saveMatrix('%s/generated_log2_mixture_%d.csv' %(outputPath, i+1), generatedDataTransformed)
        saveMatrix('%s/generated_zTransform_mixture_%d.csv' %(outputPath, i+1), generatedDataZ)
        saveMatrix('%s/generated_zTransform_log2_mixture_%d.csv' %(outputPath, i+1), generatedDataZTransformed)
        
        saveMatrix('%s/mu_mixture_%d.csv' %(outputPath, i+1), weightedMu)
        saveMatrix('%s/weights_mixture_%d.csv' %(outputPath, i+1), randomWeights)
        
        gen.append({
            "raw": generatedData,
            "zt": generatedDataZ,
            "raw_log2": generatedDataTransformed,
            "zt_log2": generatedDataZTransformed,
        })
    
    return gen

def generateWithMapping(patientDataPath, signaturePath, mappingFilePath, outputPath):
    """
    Maps the genes between patient data and a signature
    then generates data using a multi-variate normal distribtuion
    """
    
    preProcessed = getOverlappingGenes(patientDataPath, signaturePath, mappingFilePath)
    
    patientDataMatrix = preProcessed["patientData"]
    signatureMatrix = preProcessed["signature"]

    # z normalize the patient data one gene at a time across different samples
    # x2 = (x- meean(all x in the same row)) / std (all x in the same row)
    # also keep a column of the means and standard deviations to recover the original data later
    patientDataMean = np.mean(patientDataMatrix, axis = 1)
    patientDataStd = np.std(patientDataMatrix, axis = 1)
    patientDataMatrixTransformed = stats.zscore(patientDataMatrix, axis = 0)

    print("  taking cov")
    # take the gene-gene covariance of tcga
    patientDataCov = np.cov(patientDataMatrix)
    patientDataCovTransformed = np.cov(patientDataMatrixTransformed)

    print("  generating data")
    gen = generate(signatureMatrix, patientDataCov, patientDataCov, patientDataMean, patientDataStd, 4, outputPath)

def main(args):
    """
    args:
    [0] = patient data path
    [1] = signature path
    [2] = mapping file path
    [3] = output path
    """
    print(args)
    generateWithMapping(args[0], args[1], args[2], args[3])

if __name__ == "__main__":
    main(sys.argv[1:])

