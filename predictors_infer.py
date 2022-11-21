from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn import preprocessing
from pickle import load, dump
import pandas as pd
import math

with open('Predictors/scaler', 'rb') as handler :
    scaler = load(handler)

with open('Predictors/bayesianRidgePredictor', 'rb') as handler :
    bayesianRidgePredictor = load(handler)

with open('Predictors/randomForest', 'rb') as handler :
    randomForest = load(handler)

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

pred_linear = bayesianRidgePredictor.predict( scaler.transform(xs) )
pred_random = randomForest.predict( scaler.transform(xs) )

print(f'Linear score: {r2_score(pred_linear, y.to_numpy())}')
print(f'Random score: {r2_score(pred_random, y.to_numpy())}')