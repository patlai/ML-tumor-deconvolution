import numpy as np
import pandas as pd
from scipy import stats
from IO import csvToDf

def zScorePreProcess(sigMatrix, mixMatrix):
    """
    Pre-process a signature matrix and mix matrix by using z-score

    z = (x - mu) / sigma
    sigma = POPULATION stdev (divide by N) and NOT SAMPLE stdev (divide by N-1)
    """
    X = np.concatenate((sigMatrix, mixMatrix), axis=1)
    X = np.log2(X)
    X = stats.zscore(X, axis=1)
    X = X / np.max(np.absolute(X), axis = 1)[:,None]
    X = X / 2.0 + 0.5
    return X


def minMaxPreProcess(sigMatrix, mixMatrix):
    """
    Pre-process by subtracing min then dividing my (max-min) from each row
    """

    X = np.concatenate((sigMatrix, mixMatrix), axis=1)
    X = np.apply_along_axis(lambda x : (x - np.min(x)) / (np.max(x) - np.min(x)), 1, X)
    return X


def getOverlappingGenes(patientDataPath, sigMatrixPath, mappingFilePath, outputPath):
    """
    Gets all the overlapping genes between a patient data matrix and a signature matrix and truncates
    both matrices according to the overlap

    :patientDataPath: The path to the patient data csv file. This should contain an index column listing the genes
    with each column corresponding to a different patient/mixture and each row representing a different gene

    :sigMatrixPath: path to the signature matrix in .csv format with an index column listing the gene names
    Each row corresponds to a different gene and each column represents a different cell type

    :mappingFilePath: The path to the mapping text file. This should have exactly 2 columns with the first 
    column containing the format of the patient data and the 2n containing the format of the signature matrix

    :return: the truncated patient data and signature matrices
    """

    mappingArray = np.genfromtxt(mappingFilePath, delimiter = '\t', dtype='|U')
    print(mappingArray)
    mapping = dict(mappingArray)

    patientDataDf = csvToDf(patientDataPath)
    sigDf = csvToDf(sigMatrixPath)

    # truncate the patient data to the top 30k genes by variance
    # we want to eliminate the genes of low variance
    patientDataDf = patientDataDf.reindex(patientDataDf.var(axis = 1).sort_values(ascending=False).index).head(30000)

    # log2 transform the sig matrix and take the top 5k genes by variance
    sigDf = np.log2(sigDf)
    sigDf = sigDf.reindex(sigDf.var(axis = 1).sort_values(ascending=False).index).head(5000)
    
    tpmToEnsMapping = {v: k for k, v in mapping.items()}

    # go through the gene names in the epic signature (TPM) and find the corresponding ENS name
    # if the TPM name exists, keep the corresponding row in the TCGA data and epic signature        
    ensNames = set([tpmToEnsMapping[name] for name in sigDf.index if name in tpmToEnsMapping])

    patientRowsToKeep = set([name for name in patientDataDf.index if name in ensNames])
    sigRowsToKeep = set([mapping[name] for name in patientRowsToKeep])

    genesToKeep = np.array(list(sigRowsToKeep))
    np.savetxt("%s/genesToKeep.txt" %outputPath, genesToKeep, delimiter=",", fmt="%s")

    # go through the patient data and keep only the rows/genes that are still in the signature
    patientDataDf = patientDataDf[patientDataDf.index.map(lambda x: x in patientRowsToKeep)]

    # go through the signature data and keep only the rows/genes that are still in the patient data
    sigDf = sigDf[sigDf.index.map(lambda x: x in sigRowsToKeep)]

    # both data frames should have the exact same genes to keep now
    return {
        "patientData": patientDataDf,
        "signature": sigDf,
    }

