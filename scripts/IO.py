import pandas as pd
import numpy as np
import sys

def parseMixSheetFromExcel(path):
    WS = pd.read_excel(path)
    WS_np = np.array(WS)
    return np.array(WS_np[:, 1:], dtype=float)

def saveMatrix(path, matrix):
    np.savetxt(path, matrix, delimiter=',')
    print("Saved matrix to %s" %(path))

def csvToDf(path, index=0, delimiter=','):
    print("Convering %s to a data frame" %path)
    return pd.read_csv(path, index_col=index, sep=delimiter)