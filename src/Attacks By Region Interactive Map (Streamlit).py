# This adds an interactive map of the Attacks By Region Map for it to be a link within a PowerPoint presentation.

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title='Terrorism Map', layout='wide')

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
    
    return gtd_df

gtd_df = load_data()

st.title('Global Terrorism Database - Interactive Map')

original_colors = px.colors.qualitative.Dark24
unique_regions = gtd_df['Region Name'].unique()

region_colors = {}
for i, region in enumerate(unique_regions):
    if region == 'North America':
        region_colors[region] = '#E9FA32'
    else:
        region_colors[region] = original_colors[i % len(original_colors)]

fig = px.scatter_mapbox( 
    data_frame=gtd_df,
    lat='Latitude',
    lon='Longitude',
    color='Region Name',
    hover_name='Country Name',
    hover_data=['Attack Type', 'Group Name'],
    size='Total Killed',
    opacity=.5,
    size_max=30,
    zoom=2,
    center={'lat': 20, 'lon': 0},
    mapbox_style='carto-darkmatter',
    title='Map of the Global Terrorism Database By Region',
    color_discrete_map=region_colors,
    template='plotly_dark',
    width=1400,
    height=1200)

# This sets the title location and size.
fig.update_layout(
    title={
        'text': 'Map of the Global Terrorism Database By Region',
        'font': dict(size=28),
        'x': 0.5,
        'y': 0.95,
        'xanchor': 'center',
        'yanchor': 'top'
    } 
)

st.plotly_chart(fig, use_container_width=True)

# This adds neat little information.
col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Total Attacks', f'{len(gtd_df):,}')
with col2:
    st.metric('Countries Affected', f'{gtd_df['Country Name'].nunique()}')
with col3:
    st.metric('Total Deaths', f'{gtd_df['Total Killed'].sum():,.0f}')