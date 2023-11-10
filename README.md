# deep-learning-challenge
Overview of the analysis (Purpose of the analysis): 

A nonprofit foundation - Alphabet Soup - wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. To address this request, we used our knowledge of machine learning and neural networks. 

We received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. The features in the provided dataset are used to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

Results:
1.	Data Preprocessing
The charity_data.csv is read into a Pandas DataFrame:

o	What variable(s) are the target(s) for your model?

The target variable for our model is ‘IS_SUCCESSFUL’. That is, y = IS_SUCCESSFUL. This variable captures whether the money was used effectively by the organizations that have received funding from Alphabet Soup

![image](https://github.com/YeFiseha/deep-learning-challenge/assets/135238511/f4ee752d-9592-4be5-b5d2-799215b4eace)

o	What variable(s) are the features for your model?

The remaining variables are the features (‘X’s) for our model. These are:
•	APPLICATION_TYPE—Alphabet Soup application type
•	AFFILIATION—Affiliated sector of industry
•	CLASSIFICATION—Government organization classification
•	USE_CASE—Use case for funding
•	ORGANIZATION—Organization type
•	STATUS—Active status
•	INCOME_AMT—Income classification
•	SPECIAL_CONSIDERATIONS—Special considerations for application
•	ASK_AMT—Funding amount requested

o	What variable(s) should be removed from the input data because they are neither targets nor features?

EIN and NAME—Identification columns removed from the input data because they are neither targets nor features.

![image](https://github.com/YeFiseha/deep-learning-challenge/assets/135238511/a8f72a32-2571-4007-bcf9-ed34e1e02b50)

The number of unique values for each column are determined. For columns that have more than 10 unique values, the number of data points for each unique value is determined. The number of data points for each unique value are used to pick a cutoff point to bin "rare" categorical variables together in a new value, Other. The binning was successful.

pd.get_dummies() is used to encode categorical variables. 

![image](https://github.com/YeFiseha/deep-learning-challenge/assets/135238511/4f6f5017-2ccc-4f80-ae0c-77423fbc3755)

We then split the preprocessed data into a features array, X, and a target array, y. These arrays and the train_test_split function are used to split the data into training and testing datasets. The training and testing features datasets are scaled by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

![image](https://github.com/YeFiseha/deep-learning-challenge/assets/135238511/1a0874c1-7656-4ef0-a3e0-a2380d38bd3d)

2.	Compiling, Training, and Evaluating the Model

In this step, we designed a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. We first considered the number of inputs before determining the number of neurons and layers in the model. Then, we compiled, trained, and evaluated the binary classification model to calculate the model’s loss and accuracy.

o	How many neurons, layers, and activation functions did you select for your neural network model, and why?

The first model is developed with 2 hidden layers having 80, and 30 input nodes. We applied “relu” as our hidden layer activation function. For the output layer, “sigmoid” activation function was used as the model output is binary classification between 0 and 1 (indicating ‘success’ or ‘no success’) if funded by Alphabet Soup. 

![image](https://github.com/YeFiseha/deep-learning-challenge/assets/135238511/1b643773-682b-4a2b-8a5e-57f293eda4d5)

A callback was created that saves the model’s weights every five epochs and the model was trained with 100 epochs.

![image](https://github.com/YeFiseha/deep-learning-challenge/assets/135238511/bb7d9612-b768-4e81-9a51-17da94d3e395)

o	Were you able to achieve the target model performance?

The model did not achieve targeted performance as the prediction accuracy was 72% (i.e., less than the targeted 75% accuracy level).

![image](https://github.com/YeFiseha/deep-learning-challenge/assets/135238511/463c4e96-5b66-4f44-a50b-c1856be2738c)
 
o	What steps did you take in your attempts to increase model performance?

To increase the model performance and reach the targeted 75% prediction accuracy level, a second model is developed with three hidden layers – with neurons 80, 30 and 10. “Relu” and “Sigmoid” were used as the activation functions for the hidden layers and for the output layer “sigmoid” was used. The number of epochs were also reduced and tested. But even with these changes, the model did not achieve the targeted prediction accuracy level. The model’s prediction accuracy remained below 75% (at 72.9%).

Automated neural network optimization was also used allowing kerastuner to decide number of hidden layers, number of neurons in the hidden layers and which activation function to be used. The kerastuner search with 50 epochs showed the best model to be the one with 5 hidden layers having 21, 16, 6, 6, and 21 neurons activated with “relu” function. The prediction accuracy of this model still fell short of the targeted 0.75 (remaining at 0.729 with 0.55 loss).

![image](https://github.com/YeFiseha/deep-learning-challenge/assets/135238511/843f203d-da4e-44e8-a46b-1661b849d680)

3.	Summary: 

Overall, the model predicts whether applicants will be successful if funded by Alphabet Soup with 72.9% accuracy and with 0.55 loss. To improve the performance of the model (i.e., its prediction accuracy level) additional hidden layers, neurons, and epochs should be tried.

