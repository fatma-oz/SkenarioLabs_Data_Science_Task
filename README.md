## Introduction

* This project focuses on building a web application to display the most important 10 features with respect to, assessment for house price prediction project. This application can be improved by adding price prediction results.

* Here are our variables about houses in this data:

      * ssl
    
      * address
    
      * owner
    
      * neighborhood
    
      * sub_neighborhood
    
      * use_code
    
      * 2014_assessment


* Data preprocessing was done by pulling from this website: https://data.world/codefordc/dc-real-property-assessment-data

* Three different regression models were used in the model development process:

      * RandomForestRegression
    
      * DecisionTree Regression
    
      * XGboost Regression (BEST Result)


* The model was tuned and the best parameters were determined using GridsearchCV.

* In this data, there is only location data that we can use for our model. The value of a house is more than just location. Houses have several features that make up it's value. The more information we have about the properties of the houses, the more likely we are to make an accurate price estimation. By adding more features, higher accuracy can be achieved.
