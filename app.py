#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 02:52:43 2023

@author: kayttaja
"""
!pip install sklearn

import sklearn
import preprocess_function
import streamlit as st
import pickle
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


st.title(' House Price Prediction ')
html_temp = """
<div style="background-color:blue;padding:10px">
<h2 style="color:white;text-align:center;"> Streamlit ML Cloud App </h2>
</div>"""
st.markdown(html_temp, unsafe_allow_html=True)


neighborhood = option = st.selectbox(
        "variables of neighborhood column",
        ('OLD CITY II', 'KALORAMA', 'MOUNT PLEASANT', 'COLUMBIA HEIGHTS',
       'LEDROIT PARK', 'BROOKLAND', 'GEORGETOWN', 'BURLEITH', 'MALL',
       'R.L.A. SW', 'OLD CITY I', 'FOGGY BOTTOM', 'CENTRAL',
       'S. ROCK CREEK PARK', 'BERKLEY', 'WESLEY HEIGHTS', 'KENT',
       'SPRING VALLEY', 'AMERICAN UNIV. PARK', 'CHEVY CHASE', 'PALISADES',
       'FORT DRIVE', 'GLOVER-ARCHBOLD PWY', 'FOREST HILLS',
       '16TH ST. HEIGHTS', 'BRIGHTWOOD', 'PETWORTH', 'SHEPHERD PARK',
       'CHILLUM', 'TAKOMA', 'ECKINGTON', 'BRENTWOOD', 'WOODRIDGE',
       'TRINIDAD', 'RIGGS PARK', 'N. ANACOSTIA PARK', 'MICHIGAN PARK',
       'NATIONAL ARBORETUM', 'CAPITOL HILL', 'WASHINGTON NAVY YARD',
       'S. ANACOSTIA PARK', 'LILY PONDS', 'DEANWOOD', 'MARSHALL HEIGHTS',
       'FORT DUPONT PARK', 'HILLCREST', 'ANACOSTIA', 'RANDLE HEIGHTS',
       'CONGRESS HEIGHTS', 'BOLLING AFB & NAVAL RES', 'D.C. VILLAGE'),
        
    )



sub_neighborhood= option = st.selectbox(
        "variables of sub_neighborhood column",
        ('D', 'F', 'E', 'G', 'B', 'C', 'A', 'I', 'J', 'H', 'L', 'K', 'M'),
        
    )


use_code = option = st.selectbox(
        "variables of use_code column",
        ('23', '11', '42', '13', '12', '49', '24', '64', '216', '16', '18',
       '26', '51', '21', '217', '81', '29', '84', '17', '22', '27', '34',
       '41', '93', '89', '45', '91', '365', '52', '62', '48', '58', '67',
       '46', '87', '191', '39', '1', '66', '194', '68', '15', '79', '14',
       '75', '59', '94', '19', '214', '82', '28', '192', '61', '83',
       '165', '57', '74', '25', '32', '37', '85', '47', '126', '56', '65',
       '36', '86', '193', '88', '117', '69', '35', '44', '63', '38', '2',
       '96', '196', '31', '92', '53', '189', '43', '316', '195', '73',
       '465', '95', '78', '33', '71', '265', '127'),
        
    )



my_dict = {
    "neighborhood": neighborhood,
    "sub_neighborhood": sub_neighborhood,
    "use_code": use_code,

}

df = pd.DataFrame.from_dict([my_dict])




st.header("The plot of the most important 10 features with respect to, assessment for XGBoost model is below")


XGB_model = pickle.load(open('model.pkl', 'rb'))



data = pd.read_csv('https://query.data.world/s/7do5jwejce7drzdqorrzhpjtn6hrb3?dws=00000' ,encoding='latin1')
feature_names = preprocess_function.preprocess_data(data)
feature_imp = pd.Series( XGB_model.feature_importances_,
                            index=feature_names ).sort_values(ascending=False)

fig, ax = plt.subplots()
ax = sns.barplot(x=feature_imp.head(10), y=feature_imp.index[:10])
plt.xlabel('Variable İmportance Scores')
plt.ylabel('Variables')
plt.title("Variable Significance Levels")
st.pyplot(fig)

