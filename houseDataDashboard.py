def string_formatter( x ) :
    y = int(x['month'])
    y = '0' + str(y) if y < 10 else str(y)
    return f'{x["year"]}-{y}'

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score
from sklearn import preprocessing
from pickle import load, dump
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

st.set_page_config(
    page_title = 'House Data EDA',
    layout='wide'
)


# EDA Component
# print(df.head())
# print(df.describe())
# print(df.info())

df = pd.read_csv('homeData.csv')

with open('Predictors/scaler', 'rb') as handler :
    scaler = load(handler)

with open('Predictors/bayesianRidgePredictor', 'rb') as handler :
    bayesianRidgePredictor = load(handler)

with open('Predictors/randomForest', 'rb') as handler :
    randomForest = load(handler)



st.title('Housing Data EDA Showcase')




st.text('Welcome to streamlit showcase!')




# Total houses, avg price and avg condition score
col1, col2, col3 = st.columns(3)
col1.metric(
    label='Houses',
    value=df.price.count(),
    #delta=0
)
col2.metric(
    label='Average House Price ($)',
    value=round(df['price'].mean(),2),
    #delta=0
)
col3.metric(
    label='Average House Condition Score',
    value=round(df['condition'].mean(),2),
    #delta=0
)





# avg floor and space and floor nm
col1, col2, col3 = st.columns(3)
col1.metric(
    label='Average Size Floor Space (m²)',
    value=round(df['sqft_living'].mean(),2),
    #delta=0
)
col2.metric(
    label='Average Size Lot Space (m²)',
    value=round(df['sqft_lot'].mean(),2),
    #delta=0
)
col3.metric(
    label='Average number of floors',
    value=round(df['floors'].mean(),2),
    #delta=0
)



# avg bedrooms, avg badroom, nm w waterfront
col1, col2, col3 = st.columns(3)
col1.metric(
    label='Average Number of Bedrooms 🛏',
    value=round(df['bedrooms'].mean(),2),
    #delta=0
)
col2.metric(
    label='Average Number of Bathrooms 🚽',
    value=round(df['bathrooms'].mean(),2),
    #delta=0
)
col3.metric(
    label='Houses with waterfront',
    value=df[ df.waterfront != 0 ].waterfront.count(),
    #delta=0
)




# MAP
df_map = pd.DataFrame({'lat':df['lat'], 'lon':df['long']})
st.header('House Distribution.')
st.map(df_map, zoom=10)




st.header('Select the number of clusters')
num = st.slider('Select number of clusters', min_value=1, max_value=10, value=6, step=1)
df_cluster = pd.read_csv(f'Data/pos_processing_{num}.csv')




col1, col2 = st.columns(2)
with col1 :
    # ML. Slipting the house into several groups
    points = alt.Chart(df_cluster).mark_circle(opacity=0.7).encode(
        longitude='long:Q',
        latitude='lat:Q',
        size=alt.value(10),
        color='cluster',
        # tooltip=['brand', 'vicinity'])
    #).project("albersUsa")
    ).project('equirectangular')
    st.header('House Clusters')
    st.altair_chart(points, use_container_width=True)
    st.caption('The map show how the houses distribute themselves other the available clusters')




with col2 :
    # POS ML. Price of the group
    df_cluster = df_cluster[['cluster', 'price']].groupby('cluster').price.mean().reset_index()
    df_cluster.rename(columns={0:'value'}, inplace=True)
    df_cluster.set_index('cluster')
    st.header('Cluster-Price')
    st.dataframe(df_cluster)




# Columns for bar chart showing Condition and Grade
col1, col2 = st.columns(2)
with col1 :
    df_condition = df.groupby(['condition']).size().reset_index()
    df_condition.rename( columns={0:'count'}, inplace=True )
    st.header('Condition')
    st.bar_chart(df_condition, x='condition', y='count')

with col2 :
    st.header('Grade')
    df_grade = df.groupby(['grade']).size().reset_index()
    df_grade.rename( columns={0:'count'}, inplace=True )
    st.bar_chart(df_grade, x='grade', y='count')




# Heatmap
st.header('Condition Score - Grade Relationship.')
st.caption('From this two different metrics, which one is more relavant when considering the house price.')

df_gradecondition = df.groupby(['grade', 'condition']).size().reset_index()
df_gradecondition.rename( columns={0:'count'}, inplace=True )

xbin = df_gradecondition.grade.unique().shape[0] * 2
ybin = df_gradecondition.condition.unique().shape[0] * 2

heatmap = alt.Chart(df_gradecondition).mark_circle().encode(
    alt.X('grade', scale=alt.Scale(zero=False)),
    alt.Y('condition', scale=alt.Scale(zero=False, padding=1)),
    color='count',
    size='count'
)

st.altair_chart(heatmap, use_container_width=True)




# renovated houses, percentage, years till renovation
col1, col2, col3 = st.columns(3)
col1.metric(
    label='Renovated Houses',
    value=df[df['yr_renovated'] != 0].price.count(),
    #delta=0
)
col2.metric(
    label='Percentage of Houses that had renovations (%)',
    value=round(df[df['yr_renovated'] != 0].price.count() / df.price.count() * 100, 2),
    #delta=0
)
def tillRenovation(r) :
    return r['yr_renovated'] - r['yr_built']
col3.metric(
    label='Average year passed from construction to renovation',
    value=round(df[df['yr_renovated'] != 0].apply(tillRenovation, axis='columns').mean(), 2),
    #delta=0
)




# Line Chart With Sales
st.title('Sales History.')
df['date'] = pd.to_datetime(df['date'], format='%Y-%m')
df['month'] = df.date.dt.month
df['year'] = df.date.dt.year

df_ = df.groupby(['year', 'month']).size().reset_index()
df_.rename( columns={0:'count'}, inplace=True )
df_.sort_values(by=['year','month'], inplace=True)
df_['date_string'] = df_.apply(string_formatter, axis='columns')

st.line_chart(df_, x='date_string', y='count')




# Heatmap
st.header('The bedroom-bathroom count.')
st.caption('In the dataset, there was a sneaky outlier. To keep the graph elegant, we eliminated it from the dataframe.')

df_bedbath = df[df['bedrooms']<16].groupby(['bedrooms', 'bathrooms']).price.mean().reset_index()
df_bedbath.rename( columns={0:'mean_price'}, inplace=True )
print(df_bedbath)

xbin = df_bedbath.bathrooms.unique().shape[0]  / 2
ybin = df_bedbath.bedrooms.unique().shape[0]

heatmap = alt.Chart(df_bedbath).mark_rect().encode(
    alt.X('bedrooms', bin=alt.Bin(maxbins=xbin)),
    alt.Y('bathrooms', bin=alt.Bin(maxbins=ybin)),
    color='price',
)

st.altair_chart(heatmap, use_container_width=True)





# Prediction
st.header('House Price Automatic Estimation')
with st.form("prediction_form"):
    mn_lat, mx_lat = float(df.lat.min()), float(df.lat.max())
    mn_long, mx_long = float(df.long.min()), float(df.long.max())

    c1, c2 = st.columns(2)
    with c1 :
        lat = st.slider('Latitude', mn_lat, mx_lat, (mx_lat+mn_lat)/2, 0.05)
    with c2 :
        long = st.slider('Longitude', mn_long, mx_long, (mx_long+mn_long)/2, 0.05)



    c1, c2 = st.columns(2)
    with c1 :
        sqft_living = st.number_input('Living Area (m²)', min_value=0, step=10)
    
    with c2:
        sqft_lot = st.number_input('Lot Area (m²)', min_value=0, step=10)

    c1, c2 = st.columns(2)
    with c1 :
        sqft_above = st.number_input('Above Area (m²)', min_value=0, step=10)
    
    with c2:
        sqft_basement = st.number_input('Basement Area (m²)', min_value=0, step=10)



    c1, c2 = st.columns(2)
    with c1 :
        floors = st.number_input('Floor number', min_value=0, step=1)
    
    with c2:
        waterfront = st.number_input('Waterfront', min_value=0, max_value=1)

    c1, c2 = st.columns(2)
    with c1 :
        bedrooms = st.number_input('Bedrooms number', min_value=0, step=1)
    
    with c2:
        bathrooms = st.number_input('Bathrooms', min_value=0, step=1)



    c1, c2 = st.columns(2)
    with c1 :
        condition = st.slider('Condition', min_value=0, max_value=5, step=1)
    
    with c2:
        grade = st.slider('Grade', min_value=0, max_value=12, step=1)

    c1, c2 = st.columns(2)
    with c1 :
        yr_built = st.number_input('Construction Year', min_value=1900, step=1)
    
    with c2:
        yr_renovated = st.number_input('Renovation Year (0 - No Renovation)', min_value=0, step=1)



    submitted = st.form_submit_button('Submit')

if submitted :

    xs = np.array([[
        lat, long,
        sqft_living, sqft_lot, sqft_above, sqft_basement,
        floors, bedrooms, bathrooms, waterfront,
        condition, grade,
        yr_built, yr_renovated
    ]])
    xs = scaler.transform( xs )

    c1, c2 = st.columns(2)

    with c1 :
        c1.title('Bayesian Ridge')
        c1.caption('R2 score: 0.542')
        c1.text( f'Estimated value: {round(bayesianRidgePredictor.predict( xs )[0] )} $' )
        c1.caption('The model is so good, that he tells us that some business can lead to getting a house and recieve money to keep the house...')

    with c2 :
        c2.title('Random Forest')
        c2.caption('R2 score: 0.867')
        c2.text( f'Estimated value: {round(randomForest.predict( xs )[0] )} $' )




# DataFrame
st.header('DataFrame.')
st.dataframe(df)
st.caption('Table with the entries on the dataframe.')




# JSON/ DICT
st.header('Dataset\'s Structure')
st.json(df.loc[0].to_json())
st.caption('Originally the dataset was in the form of a CSV. This is the JSON dump.')