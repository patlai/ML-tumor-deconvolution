import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import sys

def histogram(x, n, min = -50, max = 50):    
    fig = plt.clf()
    fig, ax = plt.subplots(figsize=(15, 8))

    # flatten the array in the case of multi-dimensonial arrays
    arrayToPlot = np.array(x).flatten()

    dist = sns.distplot(arrayToPlot, kde=False, rug=False, bins=int(len(arrayToPlot) / n))
    
    plt.xlim(min, max)
    plt.show()

def multiHistogram(arraysToPlot, n, title, outputPath, labels=None):
    """
    generates histograms given multiple arrays to plot
    """ 
    fig = plt.clf()
    fig, ax = plt.subplots(figsize=(8, 5))
    # flatten the array in the case of multi-dimensonial arrays
    for i in range (0, len(arraysToPlot)):
        x = arraysToPlot[i]
        arrayToPlot = np.array(x).flatten()
        numBins = int(len(arrayToPlot) / n)
        
        dist = sns.distplot(
            arrayToPlot,
            kde=False,
            rug=False,
            bins=numBins,
            axlabel = "values",
            label = "mix %s" %(i+1) if labels is None else labels[i],
        )

    minimum = min([min(x) for x in arraysToPlot]) - 1
    maximum = max([max(x) for x in arraysToPlot]) + 1

    print(minimum)
    print(maximum)

    plt.xlim(minimum, maximum)
    plt.legend()
    plt.title(title)
    plt.savefig(outputPath)

def generatePlots(actual, expected, prefix, numMixes, suffix = ""):
    """
    Generates scatter plots for actual vs expected cell fractions
    for each mixture and all mixtures together
    """

    print("actual: ", actual.shape)
    print("expected: ", expected.shape)

    # make sure the dimensions are (num mixes x cell types per mix)
    if actual.shape[0] != numMixes:
        actual = actual.T

    if expected.shape != actual.shape:
        expected = expected.T

    # SEPARATE PLOTS FOR EACH MIX
    colors = ['red', 'green', 'purple', 'orange']

    # iterate through the mixtures
    for k in range(0, numMixes):
        compare = []
        print("hello world")
        print(actual.shape)
        print(expected.shape)
        
        # iterate over each cell fraction in the mixture: mixture k, cell type j
        for j in range (0, actual[k].shape[0]):
            compare.append((actual[k][j], expected[k][j]))

        # x is ground truth, y is estimate
        x = [c[1] for c in compare]
        y = [c[0] for c in compare]
        area = 40
        title = 'Mixture ' + str(k + 1)

        limit = max(max(x,y))

        plt.figure(figsize=(4, 4))
        plt.plot([0, limit], [0, limit])
        plt.scatter(x, y, s=area, color=colors[k], alpha=0.5)
        plt.ylabel('Estimaed')
        plt.xlabel('Ground truth')
        # need to call save before show or else it will export as a blank
        plt.title(prefix + ': ' + title)
        plt.savefig(prefix + title + '_%d_components_%s.png' % (len(x), suffix) )
        #plt.show()

    # ALL MIXES TOGETHER  
    colors = ['red', 'green', 'purple', 'orange']
    markers = ['o', '^', 'h', 'D']
    compare = []

    for k in range(0, numMixes):
        compare.append([])
        for j in range (0, actual.T[k].shape[0]):
            compare[k].append((actual.T[k][j], expected.T[k][j]))

    # x is ground truth, y is estimate
    x = [[mixResult[1] for mixResult in c] for c in compare]
    y = [[mixResult[0] for mixResult in c] for c in compare]
    area = 40
    #title = 'Mixture ' + str(k + 1)

    limit = max(max([max(t) for t in x]), max([max(t) for t in y]))

    plt.figure(figsize=(4, 4))
    plt.plot([0, limit], [0, limit])
    for i in range(0, numMixes):
        #print (x[i])
        #print (y[i])
        plt.scatter(x[i],
                    y[i],
                    s=area,
                    label = 'mix ' + str(i+1),
                    color=colors[i],
                    alpha=0.7,
                    #marker = markers[i]
                   )
    plt.ylabel('Estimaed')
    plt.xlabel('Ground truth')
    plt.legend()
    plt.grid(True)
    # need to call save before show or else it will export as a blank
    plt.title(prefix + ': ' + 'All mixes: Truth vs. Estimated ' +suffix)
    plt.savefig(prefix + 'All_mixes_%d_components_%s.png' %(len(x[i]), suffix))
    #plt.show()

    print("plots saved to %s" %prefix)
    
def getErrorTable(expected, actual, prefix):
    # calculate the spearman and error for each mix
    correlations = []
    errors = []
    for i in range(0, len(actual)):
        spearman = stats.spearmanr(expected[i], actual[i]).correlation
        meanAbsError = mean_absolute_error(expected[i], actual[i])
        correlations.append(spearman)
        errors.append(meanAbsError)
        #print(spearman)
    
    y_pos = np.arange(len(correlations))
    mixArray = ['mix ' + str(i+1) for i in range(0, len(correlations))]
    plt.barh(y_pos, correlations, align='center', alpha=0.5, height=0.4, color='blue')
    plt.yticks(y_pos, mixArray)
    plt.xlabel('spearman correlation')
    
    plt.title(prefix + ': Spearman Correlation between actual and expected')
    plt.show()
    
    fileName = prefix + '_performance.csv'
        
    np.savetxt(fileName,
               np.array([[i + 1 for i in range (0, len(correlations))], correlations, errors]).T,
               delimiter=",", fmt = "%10d, %1.3f, %1.3f",
               header = "Mix #,Spearman Correlation, Mean Abs. Error",
               comments = '')


def main(args):
    a = np.genfromtxt(args[0], delimiter=',').T
    a[np.isnan(a)] = 0
    arraysToPlot = a.tolist()
    labels = np.genfromtxt(args[1], delimiter=',', dtype="U32").T.tolist()
    title = args[2]
    outputPath = args[3]

    multiHistogram(arraysToPlot, 10, title, outputPath, labels = labels)

if __name__ == "__main__":
    main(sys.argv[1:])