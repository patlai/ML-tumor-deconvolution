import numpy as np
from scipy.optimize import minimize
import pandas as pd

df = pd.read_excel("./epic_signature_selected.xlsx", usecols="B:L")

S = np.array(df.as_matrix()[:, 0:7]).transpose()

for i in range(7,11):

    y = np.array(df.as_matrix()[:, i:i+1]).flatten()


    def loss(x):
        return np.sum(np.square((np.dot(x, S) - y)))

    cons = ({'type': 'eq',
             'fun' : lambda x: np.sum(x) - 1.0})

    x0 = np.zeros(S.shape[0])
    res = minimize(loss, x0, method='SLSQP', constraints=cons,
                   bounds=[(0, np.inf) for i in range(S.shape[0])])

    print(res.x)




