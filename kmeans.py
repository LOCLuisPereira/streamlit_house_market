import os
if 'Data' not in os.listdir() :
    os.mkdir('Data')

from sklearn import preprocessing, cluster
import pandas as pd
import math

df = pd.read_csv('homeData.csv')

x_train = df[['lat','long']].to_numpy()
scaler = preprocessing.MinMaxScaler()
x_trained_scaled = scaler.fit_transform(x_train)

for n in range(1, 10+1) :
    print(f'Calculating for {n}...')

    kmeans = cluster.KMeans( n_clusters=n )

    kmeans = kmeans.fit( x_trained_scaled )

    df['cluster'] = pd.Series( kmeans.predict( x_trained_scaled ) )

    df.to_csv(f'Data/pos_processing_{n}.csv')