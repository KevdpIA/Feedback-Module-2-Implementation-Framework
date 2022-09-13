- Kevin Joan Delgado Pérez A01706328
- 12/09/22

Implementation of a Machine Learning technique with any framework, this is a part from an activity with feedback.

In this repository i will public the second activity from module of Machine Learning, this activity consist in implement a method of ML with any framework.

The study case consists in the management of the same dataset of the previous activity, that is the information of the demand of bikes in Seoul, South Korea. 

The Variables used in this activity are:

 *  Dependent variable.
 
  - Rented Bike count : Count of bikes rented at each hour
  
 *  Independent variables.
 
  - Hour : Hour of he day
  - Temperature : Temperature in Celsius
  - Windspeed : m/s
  - Season : season of the year (this categoric variable will be converted in dummie variables: Spring, Summer, Autumn and Winter)

In this code, are an implementation of a Neural Network with Tensorflow, ths NN is trained with the 75% of information of the dataframe and validated with predicting the rest of the information. 

The objetive from use this method with the dataset of rented bikes in Seoul is learn the process of the Machine Learning to predict values with the learning from a NN.

Finally, the print shows the performance of the model with the errors acumulated from the trained data and the validation data.
