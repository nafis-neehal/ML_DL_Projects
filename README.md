# ML_PROJ_RPI
This is the repo for ML Projects Course at RPI (Fall 2021). I finished five independent mini-projects using Applied Machine Learning and Deep Learning models. 

Project 1: Logistic Regression on a Binary Classification Task
  - Describe an ML problem -  Stroke Prediction (Kaggle: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
  - Reasoning why Logistic Regression (LGR) would be a good choice to solve this
  - Pick public dataset
  - Perform Exploratory Data Analysis with reasoning
  - Implement LGR from scratch + write clear cost functions and derivatives
  - Use Mini-Batch/Batch Gradient Descent 
  - Optimize using RMSprop and ADAM
  - Performance Comparison

Project 2: Decision Tree Classifier for the same task as project 1
  - Implement DTC using Scikit-Learn
  - Hyperparameter tuning and exploration + optimization
  - Bagging + Boosting methods with K-Fold Cross Validation
  - Compare 3 models using appropriate metrics 
  
Project 3: MNIST Classification with 2-Layer DNN
  - MNIST data exploration
  - Train-Test-Dev split
  - Forward Prop + Activations, hyperparameters
  - Final cost function
  - Mini-Batch Gradient Descent to train model + Dropout regularization
  - Adam optimizer
  - Performance presentation on test data 
  
Project 4: Sequence Models
  - We use French to English Translation dataset for this problem. There are around 3.5k translated sentence pairs from french to english language. We use Pyter3 for TER Score calculate to evaluate machine translation quality besides BLEU score implementation in torchtext
  - Implemented RNN seq-to-seq and GRU 
  - GloVe embedding using Torchtext
  - Measure cosine similarity and euclidean distance 
  
Project 5: Exploration of CNN, AE, GAN
  - Custom CNN + retrain MobileNetV2 on CIFAR 10 dataset with Data Augmentation. Performed detailed exploration, analysis, comparison and evaluation
  - Convolutional VAE for reproducing the Fashion MNIST data samples
  - DCGAN for producing fake datasamples similar to CELEBA dataset (View in Colab to see the full notebook)
  - Reinforcement Learning to build Tic-Tac-Toe: Trained with Q-Learning and save policy, Training evaluation, Interface to play with humans with decisions based on Q Learning
