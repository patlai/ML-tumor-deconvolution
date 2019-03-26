
def spearman(S, W):    
    S_W_corr=[]
    for i in range(0,4): # 4 is a hardcoded number since we know there are 4 cell types
        new = []
        for j in range(0,4):
            corr, p_value = stats.spearmanr(S[:,i], W[:,j])
            new.append(corr)
        S_W_corr.append(new)
    return S_W_corr
    
    
def printStats(x, name):
    print("--------- %s ---------" %name)
    print("  Min: %f" %x.min())
    print("  Max: %f" %x.max())
    print("  Mean: %f" %x.mean())
    print("  Median: %f" %np.median(x))
    print("  Variance: %f" %x.var())
    

def getNegativeRatio(x, name):
    negativeRatio = len([i for i in x[0] if i < 0]) / len(x[0])
    print("  Negative ratio for %s: %f" %(name, negativeRatio))
