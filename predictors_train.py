import os
if 'Data' not in os.listdir() :
    os.mkdir('Data')
if 'Predictors' not in os.listdir() :
    os.mkdir('Predictors')

from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn import preprocessing
from pickle import load, dump
import pandas as pd
import math

df = pd.read_csv('homeData.csv')

y = df.price
x = df[[
    'lat',
    'long',

    'sqft_living',
    'sqft_lot',
    'sqft_above',
    'sqft_basement',

    'floors',
    'bedrooms',
    'bathrooms',
    'waterfront',

    'condition',
    'grade',

    'yr_built',
    'yr_renovated',

]]

xs = x.to_numpy()
scaler = preprocessing.MinMaxScaler().fit( xs )
xs_scaled = scaler.transform( xs )


bayesianRidgePredictor = BayesianRidge()
bayesianRidgePredictor.fit( xs_scaled, y.to_numpy() )

pred = bayesianRidgePredictor.predict( scaler.transform(xs) )
print( r2_score( pred, y.to_numpy() ) )




randomForest = RandomForestClassifier(n_estimators= 150, max_depth=10)
randomForest.fit( xs_scaled, y.to_numpy() )

pred = randomForest.predict( scaler.transform(xs) )
print( r2_score( pred, y.to_numpy() ) )




with open('Predictors/scaler', 'wb') as handler :
    dump(scaler, handler)

with open('Predictors/bayesianRidgePredictor', 'wb') as handler :
    dump(bayesianRidgePredictor, handler)

with open('Predictors/randomForest', 'wb') as handler :
    dump(randomForest, handler)