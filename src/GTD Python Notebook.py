import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MultipleLocator
from matplotlib.patches import Patch
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import statsmodels.formula.api as smf
import statsmodels.api as sm
import joblib
import kaleido
from math import ceil
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import os
os.environ['OMP_NUM_THREADS'] = '1'

# 'encoding='latin-1'' is needed because the default UTF-8 encoding couldn't inerpret either non-English characters, special symbols, or data exported from older software systems.
# The dataframe was also editted prior to being converted to a CSV file. This was due to excel not being able to convert it into a CSV file in its original size, so the dataframe had to been skimmed down in excel.
def load_and_describe_gtd(filepath='../data/GTD.csv'):
    gtd_df = pd.read_csv(filepath, encoding='latin-1')
    print(gtd_df.describe())
    print(gtd_df.info())
    pd.set_option('display.max_columns', None)
    print(gtd_df.head())
    return gtd_df
gtd_df = load_and_describe_gtd()

# THE CODE BELOW EDITS THE DATAFRAME

def clean_gtd_dataframe(gtd_df):
# The doubtterr column consisted of three numbers:
# 1 equalled 'Yes' and stated 'There is doubt as to whether the incident is an act of terrorism.'
# 0 equalled 'No' and stated 'There is essentially no doubt as to whether the incident is an act of terrorism.'
# -9 basically meant 'Unknown', the system didn't fully take place until 1997, so data before that time didn't always have the doubtterr column values.
# To ensure the code represented acts of terrorism, all the values that were other than 0 were dropped.
    gtd_df = gtd_df[gtd_df['doubtterr'] == 0]
    gtd_df['doubtterr'].size

# This renames some of the columns to better make sense on what they do and have them look cleaner on certain graphs.
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

# This was originally a column of its own call 'attacktype1_txt', but was mistakenly deleted before it was converted into a CSV. Using the codebook provided by the University of Maryland, I changed the numbers into their proper naming conventions.
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

# This fills in nulls.
# Fill multiple columns with different values in one line
    gtd_df = gtd_df.fillna({
        'Total Killed': 0,
        'all wounded': 'Unknown',
        'gname': 'Unknown',
        'attacktype1': 'Unknown'
    })
    
    return gtd_df

########
########

def plot_terrorism_region_map(gtd_df, save_path=None):
# This map model represent terroristic attacks by region. When hovering over the points it displays data from the gtd_df from the following columns: Country Name, Region Name, Total Killed, Latitude and Longitude, Attack Type, and Group Name.
    original_colors = px.colors.qualitative.Dark24
    unique_regions = gtd_df['Region Name'].unique()  # This preserves the original order when using Dark24. This line prevents moving the dark blue color onto another region on the map.

# This changes the color of North America on the map to a color more easily visible while keeping rest of the countries as Dark24.
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
        zoom=1,
        center={'lat': 20, 'lon': 0},
        mapbox_style='carto-darkmatter',
        title='Map of the Global Terrorism Database By Region',
        color_discrete_map=region_colors,
        template='plotly_dark',
        width=1400,
        height=1200)

# This sets the title locatoin and size.
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
    fig.show()
    if save_path:
        fig.write_image('Map of the Global Terrorism Database By Region.png',  width=1600, height=1200, scale=3)

#########
#########

def plot_terrorism_bar_charts(gtd_df, region_colors):
# This applies a dark theme for better visual appearance.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.patch.set_facecolor('#191919')
    ax1.set_facecolor('#262626')
    ax2.set_facecolor('#262626')

# This configures text colors to be readable for the dark theme.
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'lightgray'
    plt.rcParams['ytick.color'] = 'lightgray'

    region_counts = gtd_df['Region Name'].value_counts().sort_values(ascending=False)
    country_counts = gtd_df['Country Name'].value_counts().sort_values(ascending=False)
    colors = [region_colors[region] for region in region_counts.index] # Sets the colors of the bars to the same colors represented for regions in the scatter_mapbox.
    country_region_colors = [region_colors[gtd_df[gtd_df['Country Name'] == country]['Region Name'].iloc[0]] for country in country_counts.index[0:21]] # Sets the colors of the bars to the same colors represented for regions in the scatter_mapbox.
    represented_regions = gtd_df[gtd_df['Country Name'].isin(country_counts.index[0:21])]['Region Name'].unique() # Only displays the regions represented within the second graph's legend.
    legend_elements = [Patch(facecolor=region_colors[region], label=region) for region in represented_regions] # Sets the legend to correlate with region colors for the second graph.

# Bar chart for terrorist attacks by region.
    ax1.bar(region_counts.index, region_counts.values, color=colors, linewidth=0.8)
    ax1.set_xlabel('By Region', fontsize=12)
    ax1.set_ylabel('Number of Attacks', fontsize=12)
    ax1.set_title('Terrorist Attacks by Region', fontsize=16, fontweight='bold')
    ax1.tick_params(axis='x', rotation=35)
    ax1.locator_params(axis='y', nbins = 10)

# Bar chart for terrorist attacks by top 20 countries.
    ax2.bar(country_counts.index[0:21], country_counts.values[0:21], color=country_region_colors, linewidth=0.8)
    ax2.set_xlabel('By Country', fontsize=12)
    ax2.set_ylabel('Number of Attacks', fontsize=12)
    ax2.set_title('Top 20 countries With The Most Terrorist Attacks', fontsize=16, fontweight='bold')
    ax2.tick_params(axis='x', rotation=35)
    ax2.locator_params(axis='y', nbins = 12)
    ax2.legend(handles=legend_elements, loc='best', fontsize=10, title='Regions', 
            title_fontproperties={'weight': 'bold'}, facecolor='#191919', 
            edgecolor='lightgray', framealpha=0.9)

    plt.setp(ax1.xaxis.get_majorticklabels(), ha='right')
    plt.setp(ax2.xaxis.get_majorticklabels(), ha='right')
    plt.tight_layout()
    plt.show()

########
########

def train_region_classifier(gtd_df):
# This predicts terrorist attacks by region.

# This fills in missing features in Total Wounded.
    gtd_df['Total Wounded'] = pd.to_numeric(gtd_df['Total Wounded'], errors='coerce').fillna(0)

# This encodes both region and attack into numeric data.
    encoder_region = LabelEncoder()
    gtd_df['Region Name'] = encoder_region.fit_transform(gtd_df['Region Name'])
    encoder_attack = LabelEncoder()
    gtd_df['Attack Type'] = encoder_attack.fit_transform(gtd_df['Attack Type'].fillna('UNKNOWN'))
    encoder_gname = LabelEncoder()
    gtd_df['Group Name'] = encoder_gname.fit_transform(gtd_df['Group Name'])
    encoder_related = LabelEncoder()
    gtd_df['related'] = encoder_related.fit_transform(gtd_df['related'])
    encoder_corp = LabelEncoder()
    gtd_df['corp1'] = encoder_corp.fit_transform(gtd_df['corp1'])

    gtd_df['Any_Deaths'] = (gtd_df['Total Killed'] > 0).astype(int) # This converts the data into zeros and ones. Zero if no deaths occured, and one if any deaths occured.
    gtd_df['Any_Wounded'] = (gtd_df['Total Wounded'] > 0).astype(int) # This converts the data into zeros and ones. Zero if no injuries occured, and one if any injuries occured.
    gtd_df['Mass_Casualty'] = ((gtd_df['Total Killed'] + gtd_df['Total Wounded']) > 10).astype(int) # If the combined number of Total Wounded and Total Killed is greater than 10 it is considered a mass casualty event.

    X = gtd_df[['Total Killed', 'Total Wounded', 'Attack Type', 'any nationality similarities', 'targtype1', 'targsubtype1', 'Group Name', 'related', 'corp1']]
    y = gtd_df['Region Name']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    forest_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    forest_model.fit(X_train, y_train)
    forest_pred = forest_model.predict(X_test)

# Evaluate the enhanced model
    accuracy = accuracy_score(y_test, forest_pred)
    print(f'\nEnhanced Model Accuracy: {accuracy:.3f}')

# Gives analysis on which feature affects the model the most from decending order.
    feature_names = ['Total Killed', 'Total Wounded', 'Attack Type', 'any nationality similarities', 'targtype1', 'targsubtype1', 'Group Name', 'related', 'corp1']
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': forest_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print('\nFeature Importance (Geographic Enhanced):')
    print(importance_df)

    print('\nClassification Report:')
    print(classification_report(y_test, forest_pred, target_names=encoder_region.classes_))

# This trains the model 5 times and calcuates the average accuracy.
    cv_scores = cross_val_score(forest_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f'\nCross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})')
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': forest_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    return forest_model, encoder_region, X_test, y_test, forest_pred, importance_df

########
########

def plot_prediction_accuracy_map(X_test, y_test, forest_pred, encoder_region, save_path=None):
# Interactive Map with Correct and Incorrect for Region Predictions.

    X_test['Latitude'] = gtd_df.loc[X_test.index, 'Latitude']
    X_test['Longitude'] = gtd_df.loc[X_test.index, 'Longitude']
    X_test['Actual Region'] = encoder_region.inverse_transform(y_test)
    X_test['Predicted Region'] = encoder_region.inverse_transform(forest_pred)

# This calculates prediction accuracy.
    X_test['Correct Prediction'] = (y_test.values == forest_pred)
    correct_count = X_test['Correct Prediction'].sum()
    total_count = len(X_test)
    accuracy_pct = correct_count / total_count

# This maps Correct and Incorrect Predictions.
    X_test['Prediction Status'] = X_test['Correct Prediction'].map({True: 'Correct', False: 'Incorrect'})

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
        zoom= 1.3,
        center={'lat': 20, 'lon': 0},
        mapbox_style='carto-darkmatter',
        title=f"Machine Learning Model Prediction Accuracy on World Map<br>Model correctly predicted {correct_count:,} out of {total_count:,} locations<br>Geography Enables {accuracy_pct:.1%} Accuracy<br><span style='color:green'>Green = Correct Predictions</span> | <span style='color:red'>Red = Incorrect Predictions</span>",
        template='plotly_dark',
        width=1400,
        height=1200
    )

# This sets the title and legend position.
    fig.update_layout(
        title_font_size=18,
        title_x=0.5,
        title_y=.98,
        font=dict(size=12),
        margin=dict(t=100),
        legend=dict(
            yanchor='top',
            y=1,
            xanchor='right',
            x=1,
            bgcolor='#282828',
            bordercolor='#1B1B1B',
            borderwidth=3,
            font=dict(color='#FFFFFF', size=14)
        )
    )

    fig.show()
    if save_path:
        fig.write_image('prediction_map.png', width=1400, height=1200, scale=2)

########
########

def plot_feature_importance_bar(model, gtd_df, save_path=None):
# This is a horizontal bar graph that displays which column affects the model the most.
    bar_df = gtd_df.copy()[['Total Killed', 'Total Wounded', 'Attack Type', 'any nationality similarities', 'targtype1', 'targsubtype1', 'Group Name', 'related', 'corp1']]

    bar_df = bar_df.rename(columns={
        'any nationality similarities': 'Nationality Similarities',
        'targtype1': 'Target Type',
        'targsubtype1': 'Target Subtype',
        'Group Name': 'Terrorist Group',
        'related' : 'Related',
        'corp1': 'Corp/Gov Involved'
    })

    feature_names = ['Total Killed', 'Total Wounded', 'Attack Type', 'Nationality Similarities', 'Target Type', 'Target Subtype', 'Terrorist Group', 'Related', 'Corp/Gov Involved']
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': forest_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor('#191919')
    ax.set_facecolor('#262626')
    ax.barh(importance_df['Feature'], importance_df['Importance'])
    ax.set_xlabel('Importance', fontsize=16)
    ax.set_ylabel('Columns', fontsize=16)
    ax.set_title('Columns that Affect Scores the Most', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0)) # Changes the x-axis to percentage format.
    ax.xaxis.set_major_locator(MultipleLocator(0.05)) # Sets the x-axis major ticks to increments of 5%.
    ax.invert_yaxis()
    plt.show()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor())