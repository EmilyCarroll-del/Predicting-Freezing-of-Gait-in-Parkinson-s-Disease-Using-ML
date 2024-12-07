# Michael J. Fox Foundation - Predicting Freezing of Gait in Parkinson's Disease Using Machine Learning
This project uses various machine learning models to predict three triggers of freezing of gait - turn, walking, and start hesitation - in Parkinson's disease (PD) using the DeFOG and tDCSFOG datasets. 

### Datasets
https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/data

### Introduction
Freezing of gait is a detrimental symptom of PD that lowers independence and increases risk of falls. Using labeled accelerometer data collected from a lower back sensor with vertical, mediolateral, and anteroposterior sensors, this project aims to detect the start and stop of freezing of gait episodes. 

This project tests nine models on two datasets to attempt to find a suitable model for this problem. It was created in collaboration with the Break Through Tech AI program sponsored by Cornell University and overseen by the Michael J. Fox Foundation, who can use this work for future PD research and evaluation. 

The key benefits of this project are its contribution to PD research through analysis and quantification of episodes of freezing of gait. The large number of models trained provides insight into what does or does not have potential to accurately evaluate freezing of gait going forward. While traditional models such as Gradient Boosting or SVM may have potential applications, a neural network so as a Bidirectional-GRU shows great promise for this problem.

### Features
- Preprocessing used for traditional machine learning (ML) models
- Preprocessing used for the test data sets
- Group of traditional ML models -> KNN, Logistic Regression, Decision Tree, Naive Bayes, and Gradient Boosting
- SVM model
- Random Forest model
- LSTM model
- ResNet model
- Bi-GRU model
- Additional code to be used for future improvements

### Methods
Preprocessing pipeline:
- Remove missing values
- Remove small amount of outliers 
- Filter the accelerometer data
- Combination of oversampling using BorderlineSMOTE and undersampling using AllKNN in imblearn
- Extract features using seglearn
- Add label columns to feature dataframe, use the mode of each window
- Drop low variance columns
- Eliminate features with little or no correlation with training data
- Eliminate one feature of a highly correlated pair
- Dimensionality reduction + Rescaling the data

Traditional Models:
- Trained using preprocessed training and test datasets
- Use libraries for model architecture
- Hyperparameter tuning performed using gridsearch

Neural Networks:
- Preprocessed by removing missing values, removing outliers, and filtering
- Windowed in 3 second chunks with a 1 second stride
- Trained over several epochs, more suggested

### Results
Traditional Models:
- Similar results
- Unable to capture the
- complexity of the datasets

Overall:
 - SVM and Gradient Boosting -> greatest potential
   - Sensitive to hyperparameter tuning + many parameters
   - Difficult to fine-tune
 - For some labels, Naive Bayes and Decision Tree performed well
 - Little potential for improvement
 -
Neural Networks
- Incredibly computationally intensive
- Undertrained -> limited amount of epochs
- Bi-GRU and ResNet -> greatest potential
  - Less validation loss
  - More straightforward architectures
- LSTM -> complex and difficult to tune hyperparameters

Evaluated using the weighted average of precision, recall, and f1 score for each label in each dataset.

DeFOG Turn

Model          Precision  Recall   F1

SVM               0.82     0.68   0.67

Gradient Boosting 0.82     0.68   0.67

tDCSFOG Turn

Model    Precision  Recall   F1

ResNet     0.76      0.59   0.44

Bi-GRU     0.88     0.88   0.88

### Potential Next Steps
Preprocessing methods
- Undersampling using clustering
- Spectrogram, useful for ResNet
  
Models
- Convolutional Neural Networks (CNNs)
- Temporal Convolutional Networks (TCNs)
- Transformers (e.g. Time Series Transformer)
- Autoencoders and Variational Autoencoders (VAEs)

Evaluation techniques
- Grid search for neural networks
- Validate models on unlabeled data
- Stratified k-fold to tune imbalanced data

Collect start hesitation FoG events within the DeFOG environment

### Installation
Download the data from the kaggle link above.

For traditional ML models:
- Downlaoad preprocessing and test_preprocessing to transform training and test datasets
- Download one of the model files
- Use the output of the preprocessing files to train a model

For neural networks:
- Preprocessing is included in the model file
- Download a model and train it using the raw dataset

### License
This project is licensed under the Apache License 2.0. - see the LICENSE file for details.

### Acknowledgements
Thank you to the Michael J. Fox Foundation, our challenge advisor, Barbara Marebwa, our TA, Mako Ohara, and the Break Through Tech AI program for making this project possible. We also took some inspiration for models from the Kaggle competition linked above.

Libraries:
- tsflex
- seglearn
- imblearn
- tensorflow
- scikit learn
- pandas
- matplotlib
- seaborn
- numpy
- scipy
- os
