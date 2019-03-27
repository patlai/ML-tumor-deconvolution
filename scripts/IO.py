import pandas as pd
import numpy as np
import sys

def parseMixSheetFromExcel(path):
    """
    reads an excel sheet into a numpy array
    """
    
    WS = pd.read_excel(path)
    WS_np = np.array(WS)
    return np.array(WS_np[:, 1:], dtype=float)

def saveMatrix(path, matrix):
    """
    Saves a matrix to a .csv file
    """

    np.savetxt(path, matrix, delimiter=',')
    print("Saved matrix to %s" %(path))

def csvToDf(path, index=0, delimiter=','):
    """
    converts a .csv file to a pandas data frame using the first column as an index
    """

    print("Convering %s to a data frame" %path)
    return pd.read_csv(path, index_col=index, sep=delimiter)