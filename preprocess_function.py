#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 01:58:37 2023

@author: kayttaja
"""

# Libraries

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split 

import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns


# Function that deletes the first element of the data in the column (we will use it when reading data)

def remove_from_beginning(x):
            return x[1:]
        
        
def preprocess_data (data):
    
    # Removing duplicates
    data.duplicated(subset=None, keep="first")

    # Removing the $ sign in the 2014_assessment column
    data["2014_assessment"] = data["2014_assessment"].map(remove_from_beginning)

    # Removing the dots and commas in the currency in the 2014 assessment colum
    data["2014_assessment"] = data["2014_assessment"].str.replace(',','')
    data["2014_assessment"] = data["2014_assessment"].str.replace('.','')
    
    # Dropping rows containing 'ot Available'
    data = data[ ~ (data["2014_assessment"] == 'ot Available') ]
    
    # Converting currency data to numeric value
    data["2014_assessment"] = data["2014_assessment"].astype(int)
    
    # Although use_code appears as an integer, it actually represents a categorical variable
    data["use_code"] = data["use_code"].astype(str)
    
    # Dropping unnecessary columns
    data = data.drop( columns= ['address', 'owner', "ssl"])
    
    # Missing values treatment for sub_neighborhood column
    data.dropna(subset=['sub_neighborhood'], inplace=True)
    
    # preparing to model
    X = data.drop('2014_assessment', axis=1)  
    y = data["2014_assessment"]
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3, random_state = 0)
    
    # one hot encoder
    column_trans = make_column_transformer((OneHotEncoder(handle_unknown="ignore", sparse=False), Xtrain.columns ), 
                                       remainder='passthrough')
    Xtrain = column_trans.fit_transform(Xtrain)
    Xtest = column_trans.transform(Xtest)
    feature_names = column_trans.get_feature_names_out()

    return feature_names
    