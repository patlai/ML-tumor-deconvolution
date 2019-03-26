import matplotlib.pyplot as plt
import seaborn as sns

def histogram(x, n, min = -50, max = 50):    
    fig = plt.clf()
    fig, ax = plt.subplots(figsize=(15, 8))

    # flatten the array in the case of multi-dimensonial arrays
    arrayToPlot = np.array(x).flatten()

    dist = sns.distplot(arrayToPlot, kde=False, rug=False, bins=int(len(arrayToPlot) / n))
    
    plt.xlim(min, max)
    plt.show()

def generatePlots(actual, expected, prefix, numMixes):

    print("actual: ", actual.shape)
    print("expected: ", expected.shape)

    # SEPARATE PLOTS FOR EACH MIX
    colors = ['red', 'green', 'purple', 'orange']
    for k in range(0, numMixes):
        compare = []
        print("hello world")
        print(actual.shape)
        print(expected.shape)
        for j in range (0, actual.T[k].shape[0]):
            compare.append((actual.T[k][j], expected.T[k][j]))

        # x is ground truth, y is estimate
        x = [c[1] for c in compare]
        y = [c[0] for c in compare]
        area = 40
        title = 'Mixture ' + str(k + 1)

        limit = 1#max(max(x,y))

        plt.figure(figsize=(4, 4))
        plt.plot([0, limit], [0, limit])
        plt.scatter(x, y, s=area, color=colors[k], alpha=0.5)
        plt.ylabel('Estimaed')
        plt.xlabel('Ground truth')
        # need to call save before show or else it will export as a blank
        plt.title(prefix + ': ' + title)
        plt.savefig(prefix + title + '_%d_components.png' % len(x))
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
    plt.title(prefix + ': ' + 'All mixes: Truth vs. Estimated')
    plt.savefig(prefix + 'All_mixes_%d_components.png' %len(x[i]))
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