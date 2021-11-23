
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from apyori import apriori

DataFrame=pd.read_csv('/Users/nouromran/Documents/SKlearn/Data/Market_Basket_Optimisation.csv',header=None)

transactions=[]
for i in range(len(DataFrame)):
    transactions.append([str(DataFrame.values[i,j]) for  j in range (0,20)])
# ex min len at least 2 project together
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
results = list(rules)
print(type(results))