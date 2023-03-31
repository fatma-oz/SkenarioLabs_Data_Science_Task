## Introduction

* This project focuses on building a web application to display the most important 10 features with respect to, assessment for house price prediction project. This application can be improved by adding price prediction results.

* First, this project was deployed with streamlit and the application was created. Streamlit application screenshot is in app · Streamlit.pdf file.
 
* As a bonus, the streamlit application was dockerized. To run project ; 

     * install docker compose

     * run in the project root

            docker compose up

* Data pulled from this website: https://data.world/codefordc/dc-real-property-assessment-data

* Here are our variables about houses in this data:

      * ssl
    
      * address
    
      * owner
    
      * neighborhood
    
      * sub_neighborhood
    
      * use_code
    
      * 2014_assessment


* Data manipulation was done.

* The manipulation operations on the notebook were turned into functions (preprocess_function)  and imported into the app.py file.

* Three different regression models were used in the model development process:

      * RandomForestRegression
    
      * DecisionTree Regression
    
      * XGboost Regression (BEST Result) (r2 score)


* The model was tuned and the best parameters were determined using GridsearchCV.

* In this data, there is only location data that we can use for our model (neighborhood, sub_neighborhood, use_code). The value of a house is more than just location. Houses have several features that make up it's value. The more information we have about the properties of the houses, the more likely we are to make an accurate price estimation. By adding more features, higher accuracy can be achieved.


## New Achievements

* Update Macos version catalina to monterey

* Update python version (all libraries need to install again)

* Model saving (joblib, pickle)

* Model deployment (streamlit, flask , fastapi)

* BONUS Create streamlit docker app (dockerfile, docker image, docker container, docker-compose.yaml file)

