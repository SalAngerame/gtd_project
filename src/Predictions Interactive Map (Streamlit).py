# This adds an interactive map of the Model Predictions Map for it to be a link within a PowerPoint presentation.

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title='Prediction Accuracy Map', layout='wide')

@st.cache_data
def load_data():
    gtd_df = pd.read_csv('../data/GTD.csv', encoding='latin-1')
    gtd_df = gtd_df[gtd_df['doubtterr'] == 0]
    gtd_df = gtd_df.rename(columns={
        'region_txt': 'Region Name',
        'country_txt': 'Country Name',
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'all killed': 'Total Killed',
        'attacktype1': 'Attack Type',
        'all wounded': 'Total Wounded',
        'gname': 'Group Name'
    })
    gtd_df = gtd_df.fillna({
        'Total Killed': 0,
        'Total Wounded': 0,
        'Group Name': 'Unknown',
        'Attack Type': 'Unknown'
    })
    attacktype_mapping = {
        1: 'ASSASSINATION',
        2: 'ARMED ASSAULT', 
        3: 'BOMBING/EXPLOSION',
        4: 'HIJACKING',
        5: 'HOSTAGE TAKING (BARRICADE INCIDENT)',
        6: 'HOSTAGE TAKING (KIDNAPPING)',
        7: 'FACILITY / INFRASTRUCTURE ATTACK',
        8: 'UNARMED ASSAULT',
        9: 'UNKNOWN'
    }
    gtd_df['Attack Type'] = gtd_df['Attack Type'].replace(attacktype_mapping)
    gtd_df['Total Wounded'] = pd.to_numeric(gtd_df['Total Wounded'], errors='coerce').fillna(0)
    return gtd_df

gtd_df = load_data()

feature_cols = ['Total Killed', 'Total Wounded', 'Attack Type', 'any nationality similarities', 'targtype1', 'targsubtype1', 'Group Name', 'related', 'corp1']
for col in ['Attack Type', 'Group Name', 'related', 'corp1']:
    le = LabelEncoder()
    gtd_df[col] = le.fit_transform(gtd_df[col].astype(str))

region_le = LabelEncoder()
region_le.fit(gtd_df['Region Name'])

X = gtd_df[feature_cols]
y = region_le.transform(gtd_df['Region Name'])

forest_model = joblib.load('../model/forest_model.joblib')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

forest_pred = forest_model.predict(X_test)

X_test = X_test.copy()
X_test['Latitude'] = gtd_df.loc[X_test.index, 'Latitude']
X_test['Longitude'] = gtd_df.loc[X_test.index, 'Longitude']

X_test['Actual Region'] = region_le.inverse_transform(y_test)
X_test['Predicted Region'] = region_le.inverse_transform(forest_pred)
X_test['Correct Prediction'] = (y_test == forest_pred)
X_test['Prediction Status'] = X_test['Correct Prediction'].map({True: 'Correct', False: 'Incorrect'})

st.title('Model Prediction Accuracy on World Map')

fig = px.scatter_mapbox(
    X_test,
    lat='Latitude',
    lon='Longitude',
    color='Prediction Status',
    color_discrete_map={'Correct': '#00FF00', 'Incorrect': '#FF0000'},
    hover_name='Actual Region',
    hover_data={
        'Predicted Region': True,
        'Latitude': ':.3f',
        'Longitude': ':.3f',
        'Prediction Status': True
    },
    opacity=0.7,
    size_max=8,
    zoom=2,
    center={'lat': 20, 'lon': 0},
    mapbox_style='carto-darkmatter',
    title='Machine Learning Model Prediction Accuracy on World Map',
    template='plotly_dark',
    width=1400,
    height=1200
)

fig.update_layout(
    title={
        'text': 'Machine Learning Model Prediction Accuracy on World Map',
        'font': dict(size=28),
        'x': 0.5,
        'y': 0.95,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    font=dict(size=12),
    margin=dict(t=100),
    legend={
        'yanchor': 'top',
        'y': 1,
        'xanchor': 'right',
        'x': 1,
        'bgcolor': '#282828',
        'bordercolor': '#1B1B1B',
        'borderwidth': 3,
        'font': {
            'color': '#FFFFFF',
            'size': 14
        }
    }
)

st.plotly_chart(fig, use_container_width=True)