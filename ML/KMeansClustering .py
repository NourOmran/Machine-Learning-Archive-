
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from  matplotlib import pyplot as plt
import pandas as pd
def Load_Data():
    # load the data from csv file
    DataFrame = pd.read_csv('/Users/nouromran/Documents/SKlearn/Data/income.csv')

    #plt.scatter(DataFrame['Age'], DataFrame['Income($)'])
    #plt.show()
    return DataFrame
def preprocessing ():
    # preprocessing using MinMax Scaler to my feture from 0 to 1
    DataFrame=Load_Data()

    scaler = MinMaxScaler()
    scaler.fit(DataFrame[['Income($)']])
    # scaler will scale icome features
    DataFrame['Income($)'] = scaler.transform(DataFrame[['Income($)']])

    scaler.fit(DataFrame[['Age']])
    # scaler will scale Age features
    DataFrame['Age'] = scaler.transform(DataFrame[['Age']])

    return DataFrame
def train_show():
    df=preprocessing()
    km = KMeans(n_clusters=3)
    ylabel = km.fit_predict(df[['Age','Income($)']])
    print(ylabel)
    df['Label']=ylabel
    df1 = df[df.Label == 0]
    df2 = df[df.Label == 1]
    df3 = df[df.Label == 2]
    plt.scatter(df1.Age, df1['Income($)'], color='green')
    plt.scatter(df2.Age, df2['Income($)'], color='red')
    plt.scatter(df3.Age, df3['Income($)'], color='black')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker='*', label='centroid')
    plt.legend()
    plt.show()
train_show()